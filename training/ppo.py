"""PPO baseline for Project MissingNo (Phase 1).

A single-file implementation in the style of CleanRL, written to be read.
The whole algorithm is a loop of two halves:

  1. PLAY:  run the current policy in N games in parallel for a few hundred
            steps, remembering what it saw, did, and got.
  2. LEARN: use that experience to nudge the network toward actions that turned
            out better than expected, without changing it too much at once
            (that "not too much" is the "Proximal" in PPO).

Usage (from the repository root):
    uv run python training/ppo.py --total-steps 1_000_000 --run-name smoke
    uv run python training/ppo.py --total-steps 100_000_000 --seed 1 --run-name ppo_s1
    uv run python training/ppo.py --games red,blue,yellow --num-envs 6 --run-name gen1_s1

Watch the curves:
    uv run tensorboard --logdir runs

Watch one of the games live while it learns:
    uv run python training/ppo.py ... --show
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import deque
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AutoresetMode
from highlights import Highlights
from missingno_agents import build_network
from missingno_envs import COMPONENTS, EnvConfig, PokemonEnv, RewardConfig
from missingno_envs.exploration import ExplorationHub
from missingno_games import ADAPTERS, default_rom, default_state, goal_index
from torch import nn
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Config:
    # --- experiment
    run_name: str = "ppo"
    seed: int = 1
    game: str = "red"
    rom: str = ""  # default: roms/pokemon_<game>.gb (or .gbc)
    state: str = ""  # default: states/<game>_start.state
    # Several games in the same run, one model: e.g. "red,blue,yellow". The parallel
    # games are shared out in turn (6 games and 3 titles = 2 each), so num_envs should
    # be a multiple of the number of titles. ROMs and start states are taken from the
    # default paths. Overrides --game, --rom and --state.
    games: str = ""
    # Crystal is the held-out test game (docs/research.md): training on it is refused
    # unless this is set, so the test cannot be contaminated by mistake.
    allow_held_out: bool = False
    total_steps: int = 100_000_000  # agent steps, summed over all parallel games
    # Length of one episode. One step = 24 frames = 0.4 s of game time, so 8192 steps
    # are ~55 minutes of play. Longer episodes give the agent time to go further.
    episode_steps: int = 8192
    reward_version: str = "v2.2"  # see docs/rewards.md; "v1" = the original Phase 1 baseline
    # End an episode after this many steps without progress. -1 = the reward version's
    # default (v2, v2.1: 2000; v2.2: 5000), 0 = never.
    stagnation_steps: int = -1
    # End an episode after this many steps of one battle without progress (catches the
    # "flee loop" in trainer battles). -1 = the reward version's default (v2.2: 1000), 0 = never.
    battle_stagnation_steps: int = -1
    # --- exploration (docs/rewards.md, docs/SETUP.md). All off by default.
    # Go-Explore style archive: probability that an episode starts from a state the
    # agent itself reached earlier, instead of the start state. Try 0.3-0.5.
    archive_prob: float = 0.0
    archive_progress_weight: float = 1.0  # prefer archived states where learning happens
    # Backward curriculum from the agent's own first-ever successes: probability that
    # an episode starts close to the end of a past success, then further back. Try 0.3.
    curriculum_prob: float = 0.0
    # One archive and one curriculum for all the parallel games of a title, kept in
    # this process and saved next to the checkpoints (exploration_<game>.pkl), so a
    # new session continues with them. Off = each parallel game keeps its own, lost at
    # every resume (the behaviour of runs before October 2026).
    shared_exploration: bool = True
    archive_max_cells: int = 5000
    # Self-imitation learning: replay the agent's best past actions (those that turned
    # out better than expected) so that rare successes are not forgotten. Try 1.0.
    sil_coef: float = 0.0
    sil_capacity: int = 10_000  # transitions kept (~17 KB each)
    sil_store_frac: float = 0.05  # store the top 5% of each batch, by advantage
    sil_updates: int = 2  # replay minibatches per learning phase
    sil_value_coef: float = 0.01
    # --- highlights: GIFs of the episodes that went furthest (runs/<name>/gifs)
    record_every: int = 0  # keep one frame every N steps of each episode; 0 = off. Try 4
    gif_every_steps: int = 10_000  # save the best episode of each window of this many steps
    gif_keep: int = 20  # periodic GIFs kept; record-breaking GIFs are always kept
    # --- memory
    # "none": the agent decides from the last 3 frames (about one second of game).
    # "gru": a short-term memory carried from step to step and wiped at every new
    # episode, so the agent can remember what is not on the screen (e.g. that it
    # is carrying Oak's Parcel). It learns what to remember over num_steps steps.
    memory: str = "none"
    memory_size: int = 256
    # --- what the agent sees, and its eyes (the "v3" model: --downscale 1 --network impala)
    # 2 = half resolution (80x72): fast, but the 8x8 letters become unreadable. 1 = the
    # full Game Boy screen (160x144): the agent can read dialogues and menus.
    downscale: int = 2
    network: str = "atari"  # "atari" (3 convolutions) or "impala" (residual, deeper)
    # Scale rewards by the running spread of the discounted return, so that the critic
    # sees values of similar size whatever the reward version or the discount: needed
    # with a long horizon (--gamma 0.999). Logged returns stay unscaled.
    norm_rewards: bool = False
    # --- final goal
    # Stop training as soon as an episode from the start state reaches the last
    # milestone (for Red: the Boulder Badge). Its replay is saved in
    # runs/<name>/replays, the model in checkpoints/winner.pt. With several games,
    # training stops when every game has been won at least once; the model at the
    # first win of each game is saved as checkpoints/winner_<game>.pt.
    stop_at_goal: bool = False
    # The goal milestone, e.g. "M12" (first badge) or "M40" (the whole of Red).
    # "" = the game's own goal for the current phase (Red: M12).
    goal: str = ""
    # --- parallelism
    num_envs: int = 6  # parallel games: leave a core or two free for the OS
    num_steps: int = 256  # steps per game before each learning phase
    sync_envs: bool = False  # run games in the main process (slower, easier to debug)
    # Open a window on game 0 to watch training live. It runs at training speed
    # (fast), and costs a little speed. Stop training with Ctrl+C in the terminal,
    # not by closing the window.
    show: bool = False
    # --- PPO hyperparameters
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True  # linearly decrease the learning rate to 0
    gamma: float = 0.998  # discount: how much the future matters (high: long game)
    gae_lambda: float = 0.95  # bias/variance trade-off of the advantage estimate
    num_minibatches: int = 4
    update_epochs: int = 4  # how many times we reuse each batch of experience
    clip_coef: float = 0.1  # how far the new policy may move from the old one
    ent_coef: float = 0.01  # bonus for staying uncertain: encourages exploration
    vf_coef: float = 0.5  # weight of the value (critic) loss
    max_grad_norm: float = 0.5  # gradient clipping, for stability
    # --- logging and saving
    checkpoint_every: int = 50  # in learning updates
    device: str = "auto"  # "auto", "cpu" or "cuda"
    # --- long runs split across sessions (e.g. Kaggle)
    resume: str = ""  # path to a checkpoint (.pt) to continue from
    time_limit_hours: float = 0.0  # stop cleanly and save after this many hours (0 = no limit)
    # CPU threads for PyTorch. Keep num_envs + torch_threads <= your logical cores,
    # otherwise the emulators and the network fight for the CPU and everything slows down.
    torch_threads: int = 2


def game_list(cfg: Config) -> list[str]:
    """The titles played in this run: --games if given, otherwise --game."""
    games = [g.strip() for g in cfg.games.split(",") if g.strip()] if cfg.games else [cfg.game]
    unknown = [g for g in games if g not in ADAPTERS]
    if unknown:
        raise SystemExit(f"Unknown game(s) {unknown}: choose from {sorted(ADAPTERS)}")
    return games


def game_of(cfg: Config, index: int) -> str:
    """Which title the parallel game number `index` plays (shared out in turn)."""
    games = game_list(cfg)
    return games[index % len(games)]


def game_files(cfg: Config, game: str) -> tuple[str, str]:
    """ROM and start state of a title: --rom/--state for a single game, else the defaults."""
    if cfg.games:
        return default_rom(game), default_state(game)
    return cfg.rom or default_rom(game), cfg.state or default_state(game)


def make_env(cfg: Config, index: int) -> gym.Env:
    """Build one game. Defined at top level so subprocesses can create it."""
    game = game_of(cfg, index)
    rom, state = game_files(cfg, game)
    state = state if Path(state).exists() else None
    rewards = RewardConfig.preset(cfg.reward_version)
    if cfg.stagnation_steps >= 0:
        rewards = rewards.with_weights(stagnation_steps=cfg.stagnation_steps)
    if cfg.battle_stagnation_steps >= 0:
        rewards = rewards.with_weights(battle_stagnation_steps=cfg.battle_stagnation_steps)
    env_cfg = EnvConfig(
        rom_path=rom,
        start_state_path=state,
        max_steps=cfg.episode_steps,
        rewards=rewards,
        archive_prob=cfg.archive_prob,
        archive_max_cells=cfg.archive_max_cells,
        shared_exploration=cfg.shared_exploration,
        goal=cfg.goal or None,
        archive_progress_weight=cfg.archive_progress_weight,
        curriculum_prob=cfg.curriculum_prob,
        record_every=cfg.record_every,
        record_dir=str(Path("runs") / cfg.run_name / "recordings"),
        record_tag=f"{game}_env{index}",
        stop_at_goal=cfg.stop_at_goal,
        downscale=cfg.downscale,
        replay_dir=str(Path("runs") / cfg.run_name / "replays"),
    )
    if cfg.show and index == 0:
        # emulation_speed=0: the watched game must not slow down the other games,
        # which would all wait for it at every step.
        return PokemonEnv(env_cfg, ADAPTERS[game](), render_mode="human", emulation_speed=0)
    return PokemonEnv(env_cfg, ADAPTERS[game]())


def train(cfg: Config) -> None:
    # ------------------------------------------------------------------ setup
    games = game_list(cfg)
    multi = len(games) > 1
    held_out = [g for g in games if getattr(ADAPTERS[g], "held_out", False)]
    if held_out and not cfg.allow_held_out:
        raise SystemExit(
            f"{', '.join(held_out)} is held out: it is only used to test generalization, "
            "never for training (docs/research.md). Evaluate on it with scripts/evaluate.py. "
            "To train on it anyway, as a declared separate experiment, add --allow-held-out."
        )
    for game in games:
        rom, state = game_files(cfg, game)
        if not Path(rom).exists():
            raise SystemExit(f"ROM for {game} not found at {rom} (see roms/README.md).")
        if not Path(state).exists():
            print(
                f"\n!!! WARNING: start state {state} not found: every {game} episode starts "
                "from the power-on screen (title screen, new game, names), not from the "
                f"bedroom. Create it with: scripts/make_start_state.py --game {game}\n"
            )
    if cfg.num_envs % len(games):
        print(
            f"NOTE: {cfg.num_envs} parallel games for {len(games)} titles: some titles get "
            "more games than others. Use a multiple of the number of titles."
        )
    env_games = [game_of(cfg, i) for i in range(cfg.num_envs)]
    if cfg.reward_version not in COMPONENTS:
        raise SystemExit(
            f"Unknown --reward-version {cfg.reward_version!r}: choose from {sorted(COMPONENTS)}"
        )
    run_dir = Path("runs") / cfg.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    torch.set_num_threads(cfg.torch_threads)

    # N games in parallel. SAME_STEP autoreset: when a game ends, the observation
    # we get back is already the first frame of the next episode, and the final
    # statistics of the finished episode arrive in info["final_info"].
    factories = [partial(make_env, cfg, i) for i in range(cfg.num_envs)]
    if cfg.sync_envs:
        envs = gym.vector.SyncVectorEnv(factories, autoreset_mode=AutoresetMode.SAME_STEP)
    else:
        envs = gym.vector.AsyncVectorEnv(
            factories, autoreset_mode=AutoresetMode.SAME_STEP, context="spawn"
        )
    vec = envs  # the bare vector env: call() and set_attr() reach the games
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    milestones = {g: ADAPTERS[g].milestones for g in games}

    obs_shape = envs.single_observation_space.shape
    n_actions = envs.single_action_space.n
    agent = build_network(obs_shape, n_actions, cfg.memory, cfg.memory_size, cfg.network).to(device)
    scaler = RewardScaler(cfg.num_envs, cfg.gamma) if cfg.norm_rewards else None
    recurrent = agent.recurrent
    if recurrent and cfg.num_envs % min(cfg.num_minibatches, cfg.num_envs):
        raise SystemExit("With --memory gru, --num-envs must be a multiple of --num-minibatches.")
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    batch_size = cfg.num_envs * cfg.num_steps
    sil = (
        SelfImitationBuffer(cfg.sil_capacity, tuple(obs_shape), cfg.memory_size if recurrent else 0)
        if cfg.sil_coef > 0
        else None
    )
    # One set of highlights per title: their milestones are not comparable.
    highlights = (
        {
            g: Highlights(
                run_dir / "gifs" / (g if multi else ""), cfg.gif_every_steps, cfg.gif_keep
            )
            for g in games
        }
        if cfg.record_every > 0
        else None
    )
    minibatch_size = batch_size // cfg.num_minibatches
    num_updates = cfg.total_steps // batch_size

    # Buffers for one round of experience: (num_steps, num_envs, ...)
    obs_buf = torch.zeros(
        (cfg.num_steps, cfg.num_envs, *obs_shape), dtype=torch.uint8, device=device
    )
    act_buf = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.long, device=device)
    logp_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    rew_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    done_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    val_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    # With memory: the memory state before each step (after the reset of a new episode).
    mem_buf = (
        torch.zeros((cfg.num_steps, cfg.num_envs, cfg.memory_size), device=device)
        if recurrent
        else None
    )

    # Rolling statistics over the last finished episodes, per title, for the logs.
    recent_milestones = {g: deque(maxlen=50) for g in games}
    best_reached = dict.fromkeys(games, 0)
    # Per title: (replay path, step) of the first episode from the start that reached the goal.
    goals: dict[str, tuple[str, int]] = {}
    global_step = 0
    first_update = 1

    # Resume: restore the network, the optimizer and the counters, so a long run can be
    # split across several sessions and continue as if it had never stopped.
    if cfg.resume:
        saved = torch.load(cfg.resume, map_location=device)
        defaults = {"memory": "none", "network": "atari", "downscale": 2}
        for option, default in defaults.items():
            was, now = saved.get("config", {}).get(option, default), getattr(cfg, option)
            if was != now:
                flag = "--" + option.replace("_", "-")
                raise SystemExit(
                    f"The checkpoint was trained with {flag} {was}, not {now}: a different "
                    f"network. Resume it with {flag} {was}."
                )
        if scaler is not None and "reward_scaler" in saved:
            scaler.load_state_dict(saved["reward_scaler"])
        agent.load_state_dict(saved["model"])
        if "optimizer" in saved:
            optimizer.load_state_dict(saved["optimizer"])
        global_step = saved.get("global_step", 0)
        first_update = saved.get("update", 0) + 1
        saved_best = saved.get("best_reached", 0)
        if isinstance(saved_best, int):  # checkpoints from single-game runs
            saved_best = {saved.get("config", {}).get("game", games[0]): saved_best}
        for g in games:
            best_reached[g] = saved_best.get(g, 0)
        goals.update({g: tuple(v) for g, v in saved.get("goals", {}).items() if g in games})
        if goals:
            print("Already won: " + ", ".join(f"{g} at step {v[1]:,}" for g, v in goals.items()))
        print(f"Resumed from {cfg.resume}: step {global_step:,}, update {first_update - 1}")
        # Checkpoints saved before reward versions existed were trained with v1.
        saved_version = saved.get("config", {}).get("reward_version", "v1")
        if saved_version != cfg.reward_version:
            print(
                f"NOTE: the model was trained with reward {saved_version}, it now continues with "
                f"{cfg.reward_version}. Fine for a new experiment started from this model (use a "
                f"new --run-name); to continue the same run, add --reward-version {saved_version}."
            )
    # purge_step drops any log entries written after the checkpoint by an interrupted session.
    writer = SummaryWriter(str(run_dir), purge_step=global_step if cfg.resume else None)

    # Shared exploration: one hub per title (archive, curriculum, lineage), see exploration.py.
    explore = cfg.shared_exploration and (cfg.archive_prob > 0 or cfg.curriculum_prob > 0)
    hubs: dict[str, ExplorationHub] = {}
    if explore:
        for g in games:
            hubs[g] = ExplorationHub(
                archive_prob=cfg.archive_prob,
                curriculum_prob=cfg.curriculum_prob,
                archive_max_cells=cfg.archive_max_cells,
                archive_progress_weight=cfg.archive_progress_weight,
            )
            saved_hub = Path(cfg.resume).parent / f"exploration_{g}.pkl" if cfg.resume else None
            if saved_hub is not None and saved_hub.exists():
                hubs[g].load(saved_hub)
                st = hubs[g].stats()
                print(
                    f"Exploration of {g} restored: {st['archive_cells']} archived states, "
                    f"{st['demos_active']} demos"
                )
    root_states, replay_meta = {}, {}
    for i, (state, meta) in enumerate(
        zip(vec.call("start_state"), vec.call("replay_meta"), strict=True)
    ):
        root_states.setdefault(env_games[i], state)
        replay_meta.setdefault(env_games[i], meta)
    rng = np.random.default_rng(cfg.seed)

    def sync_exploration() -> None:
        """Collect what every game found, hand out the next starting points."""
        if not explore:
            return
        for i, report in enumerate(vec.call("drain")):
            hubs[env_games[i]].absorb(report)
        vec.set_attr("pending_starts", [hubs[g].starts(4, rng) for g in env_games])
        vec.set_attr("known_maps", [set(hubs[g].maps_ever) for g in env_games])

    def save_exploration() -> None:
        for g, hub in hubs.items():
            hub.save(run_dir / "checkpoints" / f"exploration_{g}.pkl")

    def save(path: Path, update: int) -> None:
        if not path.name.startswith("step_"):
            save_exploration()  # with latest.pt and the winners, not with every snapshot
        _save(
            path,
            agent,
            optimizer,
            cfg,
            obs_shape,
            n_actions,
            global_step,
            update,
            best_reached,
            goals,
            scaler,
        )

    print(
        f"Run '{cfg.run_name}': {cfg.num_envs} games ({', '.join(env_games)}), "
        f"{num_updates} updates of {batch_size} steps, "
        f"device {device}, {cfg.torch_threads} torch threads, reward {cfg.reward_version}, "
        f"archive {cfg.archive_prob:.0%}, curriculum {cfg.curriculum_prob:.0%}, "
        f"SIL {cfg.sil_coef}, memory {cfg.memory}, network {cfg.network}, "
        f"screen 1/{cfg.downscale}, gamma {cfg.gamma}, stop at goal: {cfg.stop_at_goal}, "
        f"{sum(p.numel() for p in agent.parameters()):,} parameters"
    )

    next_obs, _ = envs.reset(seed=cfg.seed)
    sync_exploration()  # after a resume, the restored archive is used from the first episodes
    next_obs = torch.as_tensor(next_obs, device=device)
    next_done = torch.zeros(cfg.num_envs, device=device)
    memory = agent.initial_state(cfg.num_envs, device) if recurrent else None
    start, start_step = time.time(), global_step
    deadline = start + cfg.time_limit_hours * 3600 if cfg.time_limit_hours > 0 else None
    completed = first_update - 1  # last fully completed update

    all_won = False
    try:
        play_time = learn_time = 0.0
        for update in range(first_update, num_updates + 1):
            t_play = time.time()
            if cfg.anneal_lr:
                optimizer.param_groups[0]["lr"] = cfg.learning_rate * (
                    1 - (update - 1) / num_updates
                )

            # ------------------------------------------------------------- 1. PLAY
            for t in range(cfg.num_steps):
                global_step += cfg.num_envs
                obs_buf[t] = next_obs
                done_buf[t] = next_done
                with torch.inference_mode():
                    if recurrent:
                        # A game that just ended starts a new episode: wipe its memory.
                        memory = memory * (1.0 - next_done).unsqueeze(-1)
                        mem_buf[t] = memory
                        action, logp, _, value, memory = agent.act(next_obs, memory)
                    else:
                        action, logp, _, value = agent.act(next_obs)
                act_buf[t], logp_buf[t], val_buf[t] = action, logp, value

                obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
                done_now = np.logical_or(terminated, truncated)
                if scaler is not None:
                    reward = scaler(reward, done_now)
                rew_buf[t] = torch.as_tensor(reward, device=device, dtype=torch.float32)
                # Simplification: a time-limit end is treated like a real end.
                # Pokémon never "ends", so all our episodes end by time limit.
                next_done = torch.as_tensor(
                    np.logical_or(terminated, truncated), device=device, dtype=torch.float32
                )
                next_obs = torch.as_tensor(obs, device=device)

                if "final_info" in info:
                    for i in np.flatnonzero(info["_final_info"]):
                        final = info["final_info"]
                        game = env_games[i]
                        won = "goal_reached" in final and final["goal_reached"][i]
                        if won and game not in goals:
                            replay = str(final["replay"][i])
                            if not replay and game in hubs:
                                # Shared mode: this process has the lineage, it writes the replay.
                                report = final["goal_report"][i]
                                replay = hubs[game].write_replay(
                                    run_dir / "replays" / f"goal_{game}_step{global_step}.npz",
                                    root_states[game],
                                    report["node"],
                                    report["actions"],
                                    **replay_meta[game],
                                )
                            goals[game] = (replay, global_step)
                            _announce_goal(
                                game,
                                goals[game],
                                milestones[game][goal_index(ADAPTERS[game], cfg.goal or None)],
                                multi,
                                run_dir,
                                update,
                            )
                            if multi:
                                save(run_dir / "checkpoints" / f"winner_{game}.pt", update)
                        if highlights is not None:
                            _add_highlight(highlights[game], final, i, global_step)
                        best_reached[game] = _log_episode(
                            writer,
                            global_step,
                            info,
                            i,
                            milestones[game],
                            recent_milestones[game],
                            best_reached[game],
                            game if multi else "",
                            local_stats=not explore,
                        )
                all_won = cfg.stop_at_goal and len(goals) == len(games)
                if all_won:
                    break  # every final goal is reached: no need to finish this rollout

            play_time += time.time() - t_play
            if all_won:
                save(run_dir / "checkpoints" / "winner.pt", update)
                save(run_dir / "checkpoints" / "latest.pt", update)
                won = ", ".join(f"{g} at step {goals[g][1]:,}" for g in games)
                print(f"\n*** EVERY GOAL REACHED ({won}) ***")
                print(f"    model: {run_dir / 'checkpoints' / 'winner.pt'}")
                break
            if highlights is not None:
                for h in highlights.values():
                    h.maybe_flush(global_step)
            sync_exploration()
            for g, hub in hubs.items():
                prefix = f"{g}/" if multi else ""
                for name, value in hub.stats().items():
                    section = "archive" if name.startswith(("archive", "lineage")) else "curriculum"
                    writer.add_scalar(f"{section}/{prefix}{name}", value, global_step)
            t_learn = time.time()

            # ------------------------------------- 2. ADVANTAGES: how good was each action?
            # GAE: compare what actually happened (rewards) with what the critic
            # expected (values). Positive advantage = better than expected.
            with torch.no_grad():
                if recurrent:
                    next_value = agent.value(next_obs, memory, next_done)
                else:
                    next_value = agent.value(next_obs)
                advantages = torch.zeros_like(rew_buf)
                last_gae = torch.zeros(cfg.num_envs, device=device)
                for t in reversed(range(cfg.num_steps)):
                    if t == cfg.num_steps - 1:
                        not_done, next_val = 1.0 - next_done, next_value
                    else:
                        not_done, next_val = 1.0 - done_buf[t + 1], val_buf[t + 1]
                    delta = rew_buf[t] + cfg.gamma * next_val * not_done - val_buf[t]
                    last_gae = delta + cfg.gamma * cfg.gae_lambda * not_done * last_gae
                    advantages[t] = last_gae
                returns = advantages + val_buf

            # ----------------------------------------------------------- 3. LEARN
            b_obs = obs_buf.reshape((-1, *obs_shape))
            b_act, b_logp = act_buf.reshape(-1), logp_buf.reshape(-1)
            b_adv, b_ret, b_val = advantages.reshape(-1), returns.reshape(-1), val_buf.reshape(-1)

            clipfracs = []
            for _ in range(cfg.update_epochs):
                for idx, new_logp, entropy, new_value in _minibatches(
                    agent, cfg, obs_buf, act_buf, done_buf, mem_buf, minibatch_size, device
                ):
                    ratio = (new_logp - b_logp[idx]).exp()  # new policy / old policy
                    with torch.no_grad():
                        clipfracs.append(((ratio - 1).abs() > cfg.clip_coef).float().mean().item())

                    adv = b_adv[idx]
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    # Policy loss: push up actions with positive advantage, but clip the
                    # ratio so one update cannot change the policy too much.
                    pg_loss = torch.max(
                        -adv * ratio, -adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)
                    ).mean()
                    # Value loss: teach the critic to predict the actual returns.
                    v_loss = 0.5 * ((new_value - b_ret[idx]) ** 2).mean()
                    # Entropy bonus: keep some randomness, so the agent keeps exploring.
                    ent_loss = entropy.mean()

                    loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent_loss
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                    optimizer.step()

            # ------------------------------------------- 4. SELF-IMITATION (optional)
            # Keep the actions that turned out much better than expected, and keep
            # practicing them later: a rare success (e.g. leaving the lab once) is
            # not forgotten after one update.
            sil_stats = None
            if sil is not None:
                k = max(1, int(cfg.sil_store_frac * batch_size))
                best = torch.topk(b_adv, k).indices
                best = best[b_adv[best] > 0]
                b_mem = mem_buf.reshape(batch_size, -1)[best] if recurrent else None
                sil.add(b_obs[best], b_act[best], b_ret[best], b_mem)
                sil_stats = _sil_update(agent, optimizer, sil, cfg, minibatch_size, device)

            learn_time += time.time() - t_learn

            # ------------------------------------------------------------- logging
            sps = int((global_step - start_step) / (time.time() - start))
            explained_var = 1 - torch.var(b_ret - b_val) / (torch.var(b_ret) + 1e-8)
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("losses/policy", pg_loss.item(), global_step)
            writer.add_scalar("losses/value", v_loss.item(), global_step)
            writer.add_scalar("losses/entropy", ent_loss.item(), global_step)
            writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
            writer.add_scalar("losses/explained_variance", explained_var.item(), global_step)
            if sil is not None:
                writer.add_scalar("sil/buffer", len(sil), global_step)
                if sil_stats:
                    writer.add_scalar("sil/still_better_frac", sil_stats[0], global_step)
                    writer.add_scalar("sil/loss", sil_stats[1], global_step)

            if update % 10 == 0 or update == num_updates:
                share = play_time / (play_time + learn_time)
                best = ", ".join(
                    (f"{g} " if multi else "") + f"{best_reached[g]}/{len(milestones[g])}"
                    for g in games
                )
                print(
                    f"update {update}/{num_updates}  steps {global_step:,}  SPS {sps}  "
                    f"(playing {share:.0%} of the time)  best milestones {best}"
                )
            completed = update
            if update % cfg.checkpoint_every == 0 or update == num_updates:
                save(run_dir / "checkpoints" / f"step_{global_step}.pt", update)
                save(run_dir / "checkpoints" / "latest.pt", update)
            if deadline and time.time() > deadline:
                print(f"Time limit of {cfg.time_limit_hours} h reached: saving and stopping.")
                save(run_dir / "checkpoints" / "latest.pt", update)
                print(f"Resume with: --resume {run_dir / 'checkpoints' / 'latest.pt'}")
                break

    except KeyboardInterrupt:
        # Ctrl+C: save what we have, so the run can still be watched or resumed.
        # The experience of the interrupted update is discarded: we save the last complete one.
        global_step = completed * batch_size
        print(f"\nInterrupted: saving the model at step {global_step:,} (last complete update)...")
        save(run_dir / "checkpoints" / "latest.pt", completed)

    envs.close()
    if highlights is not None:
        for h in highlights.values():
            h.close()
    writer.close()
    print(f"Done. Checkpoints in {run_dir / 'checkpoints'}")


def _minibatches(agent, cfg, obs_buf, act_buf, done_buf, mem_buf, minibatch_size, device):
    """Yield (flat indices, log-prob, entropy, value) for each minibatch of one epoch.

    Without memory, steps are independent: any random subset will do. With memory,
    each game's steps must be replayed in order, from the memory it had at the start
    of the rollout, so the gradient can reach back to what was remembered: the
    minibatches are random groups of whole games.
    """
    steps, n_envs = act_buf.shape
    if not agent.recurrent:
        b_obs = obs_buf.reshape((-1, *obs_buf.shape[2:]))
        b_act = act_buf.reshape(-1)
        for idx in torch.randperm(steps * n_envs, device=device).split(minibatch_size):
            _, logp, entropy, value = agent.act(b_obs[idx], b_act[idx])
            yield idx, logp, entropy, value
        return
    groups = min(cfg.num_minibatches, n_envs)
    for envs in torch.randperm(n_envs, device=device).chunk(groups):
        logp, entropy, value = agent.act_sequence(
            obs_buf[:, envs], mem_buf[0, envs], done_buf[:, envs], act_buf[:, envs]
        )
        # Flat index of (step t, game e) in the (steps * n_envs) batch: t * n_envs + e.
        idx = (torch.arange(steps, device=device).unsqueeze(1) * n_envs + envs).reshape(-1)
        yield idx, logp.reshape(-1), entropy.reshape(-1), value.reshape(-1)


def _announce_goal(game, goal, milestone, multi, run_dir, update) -> None:
    replay, step = goal
    title = f"{game.upper()}: " if multi else ""
    print(f"\n*** {title}FINAL GOAL REACHED at step {step:,}: {milestone.name}! ***")
    print(f"    replay: {replay}")
    print(f"    watch it: uv run python scripts/replay.py {replay}")
    if multi:
        print(f"    model:  {run_dir / 'checkpoints' / f'winner_{game}.pt'} (update {update})")
    else:
        print(f"    model:  {run_dir / 'checkpoints' / 'winner.pt'}")


def _add_highlight(highlights, final, i, step) -> None:
    """Hand a finished, recorded episode to the highlights (if it was recorded)."""
    path = final.get("recording", [""] * (i + 1))[i]
    if not path:
        return
    reached = int((np.asarray(final["milestone_step"][i]) >= 0).sum())
    highlights.add(
        str(path), reached, int(final["maps_visited"][i]), int(final["tiles_visited"][i]), step
    )


class RewardScaler:
    """Divide rewards by the running standard deviation of the discounted return.

    Like the usual "reward normalization" of PPO implementations, written out for our
    setting: every episode ends by a time limit (all "dones" restart the return).
    """

    def __init__(self, n_envs: int, gamma: float, epsilon: float = 1e-8) -> None:
        self.gamma, self.epsilon = gamma, epsilon
        self.returns = np.zeros(n_envs)
        self.count, self.mean, self.var = 1e-4, 0.0, 1.0

    def __call__(self, reward: np.ndarray, done: np.ndarray) -> np.ndarray:
        self.returns = self.returns * self.gamma + reward
        self._update(self.returns)
        self.returns[done] = 0.0
        return reward / np.sqrt(self.var + self.epsilon)

    def _update(self, x: np.ndarray) -> None:
        batch_mean, batch_var, n = float(np.mean(x)), float(np.var(x)), len(x)
        delta, total = batch_mean - self.mean, self.count + n
        self.mean += delta * n / total
        m2 = self.var * self.count + batch_var * n + delta**2 * self.count * n / total
        self.var, self.count = m2 / total, total

    def state_dict(self) -> dict:
        return {"count": self.count, "mean": self.mean, "var": self.var}

    def load_state_dict(self, state: dict) -> None:
        self.count, self.mean, self.var = state["count"], state["mean"], state["var"]


class SelfImitationBuffer:
    """Past (observation, action, return) where the action did better than expected.

    Kept on the CPU as uint8, so 10,000 transitions take ~170 MB of RAM.
    """

    def __init__(self, capacity: int, obs_shape: tuple[int, ...], memory_size: int = 0) -> None:
        self.obs = torch.zeros((capacity, *obs_shape), dtype=torch.uint8)
        self.act = torch.zeros(capacity, dtype=torch.long)
        self.ret = torch.zeros(capacity)
        # With memory: the memory state the agent had at that moment (as it was then).
        self.mem = torch.zeros((capacity, memory_size)) if memory_size else None
        self.capacity, self.size, self.ptr = capacity, 0, 0

    def __len__(self) -> int:
        return self.size

    def add(self, obs, act, ret, mem=None) -> None:
        for i, (o, a, r) in enumerate(zip(obs.cpu(), act.cpu(), ret.cpu(), strict=True)):
            self.obs[self.ptr], self.act[self.ptr], self.ret[self.ptr] = o, a, r
            if self.mem is not None:
                self.mem[self.ptr] = mem[i].cpu()
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int):
        idx = torch.randint(0, self.size, (n,))
        mem = self.mem[idx] if self.mem is not None else None
        return self.obs[idx], self.act[idx], self.ret[idx], mem


def _sil_update(agent, optimizer, sil, cfg, minibatch_size, device):
    """Self-imitation learning (Oh et al., 2018).

    For each replayed action, compare the return it led to with what the critic
    expects *now*. Only if the past action was still better than expected
    (R > V) does the agent imitate it, weighted by how much better. Actions the
    agent has meanwhile learned to beat are ignored automatically.
    """
    if len(sil) < minibatch_size:
        return None
    better_frac = losses = 0.0
    for _ in range(cfg.sil_updates):
        obs, act, ret, mem = sil.sample(minibatch_size)
        obs, act, ret = obs.to(device), act.to(device), ret.to(device)
        if agent.recurrent:
            # The stored memory is the one the agent had back then: an approximation
            # (the network has changed since), the usual one for replayed experience.
            _, logp, _, value, _ = agent.act(obs, mem.to(device), action=act)
        else:
            _, logp, _, value = agent.act(obs, act)
        gap = (ret - value).clamp(min=0)  # how much better than expected, 0 if not
        policy_loss = -(logp * gap.detach()).mean()
        value_loss = 0.5 * (gap**2).mean()
        loss = cfg.sil_coef * (policy_loss + cfg.sil_value_coef * value_loss)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
        optimizer.step()
        better_frac += (gap > 0).float().mean().item() / cfg.sil_updates
        losses += loss.item() / cfg.sil_updates
    return better_frac, losses


def _log_episode(
    writer, step, info, i, milestones, recent, best_reached, game="", local_stats=True
) -> int:
    """Log one finished episode: return, exploration and milestones.

    With several titles, curves are kept per title: "episode/red/return", ...
    """
    final = info["final_info"]

    def tag(section: str, name: str) -> str:
        return f"{section}/{game}/{name}" if game else f"{section}/{name}"

    if local_stats and "archive_cells" in final:
        writer.add_scalar("archive/cells", float(final["archive_cells"][i]), step)
    if local_stats and "demos_active" in final:
        writer.add_scalar("curriculum/demos_active", float(final["demos_active"][i]), step)
        writer.add_scalar("curriculum/demos_completed", float(final["demos_completed"][i]), step)
    if "from_demo" in final and final["from_demo"][i]:
        # Started from a demo: not comparable with episodes from the start state.
        writer.add_scalar(tag("curriculum", "success"), float(final["demo_success"][i]), step)
        writer.add_scalar(
            tag("curriculum", "demo_progress"), float(final["demo_progress"][i]), step
        )
        return best_reached
    if "from_archive" in final and final["from_archive"][i]:
        # Started from an archived state: useful for learning, but not comparable
        # with episodes from the start state, so kept out of the main curves.
        writer.add_scalar(tag("archive", "episode_return"), info["episode"]["r"][i], step)
        writer.add_scalar(tag("archive", "maps_visited"), float(final["maps_visited"][i]), step)
        writer.add_scalar(tag("archive", "tiles_visited"), float(final["tiles_visited"][i]), step)
        return best_reached
    writer.add_scalar(tag("episode", "return"), info["episode"]["r"][i], step)
    writer.add_scalar(tag("episode", "length"), info["episode"]["l"][i], step)
    writer.add_scalar(tag("episode", "tiles_visited"), final["tiles_visited"][i], step)
    writer.add_scalar(tag("episode", "maps_visited"), final["maps_visited"][i], step)
    for key in ("items_seen", "pokedex_owned", "pokedex_seen", "stagnated", "battle_loop"):
        if key in final:
            writer.add_scalar(tag("episode", key), float(final[key][i]), step)
    # Return of each reward component: if one dominates, watch the agent before
    # trusting the curves. That is how reward hacking is caught.
    totals = final.get("reward_totals", {})
    for name in totals:
        if not name.startswith("_"):
            writer.add_scalar(tag("reward", name), float(totals[name][i]), step)

    reached_at = np.asarray(final["milestone_step"][i])
    recent.append(reached_at >= 0)
    rates = np.mean(recent, axis=0)
    for m, at, rate in zip(milestones, reached_at, rates, strict=True):
        writer.add_scalar(tag("milestones", f"{m.id}_rate"), rate, step)
        if at >= 0:
            writer.add_scalar(tag("milestones", f"{m.id}_step"), at, step)

    n_reached = int((reached_at >= 0).sum())
    if n_reached > best_reached:
        names = ", ".join(m.name for m, at in zip(milestones, reached_at, strict=True) if at >= 0)
        where = f" in {game}" if game else ""
        print(f"  new best{where} at step {step:,}: {n_reached} milestones ({names})")
    return max(best_reached, n_reached)


def _save(
    path,
    agent,
    optimizer,
    cfg,
    obs_shape,
    n_actions,
    global_step,
    update,
    best_reached,
    goals,
    scaler=None,
):
    """Everything needed both to watch the agent and to resume training."""
    torch.save(
        {
            "model": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "obs_shape": tuple(obs_shape),
            "n_actions": int(n_actions),
            "config": dataclasses.asdict(cfg),
            "global_step": global_step,
            "update": update,
            "best_reached": best_reached,
            "goals": goals,  # per title: (replay, step) of the first win
            "reward_scaler": scaler.state_dict() if scaler is not None else None,
        },
        path,
    )


def parse_args() -> Config:
    """Every Config field becomes a command-line option: --total-steps, --seed, ..."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    for f in dataclasses.fields(Config):
        flag = "--" + f.name.replace("_", "-")
        if f.type == "bool":
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            caster = {"int": lambda v: int(v.replace("_", "")), "float": float}.get(f.type, str)
            parser.add_argument(flag, type=caster, default=f.default)
    return Config(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_args())
