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
from missingno_agents import CnnActorCritic
from missingno_envs import COMPONENTS, EnvConfig, PokemonEnv, RewardConfig
from missingno_games import ADAPTERS
from torch import nn
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Config:
    # --- experiment
    run_name: str = "ppo"
    seed: int = 1
    game: str = "red"
    rom: str = "roms/pokemon_red.gb"
    state: str = "states/red_start.state"
    total_steps: int = 100_000_000  # agent steps, summed over all parallel games
    # Length of one episode. One step = 24 frames = 0.4 s of game time, so 8192 steps
    # are ~55 minutes of play. Longer episodes give the agent time to go further.
    episode_steps: int = 8192
    reward_version: str = "v2"  # see docs/rewards.md; "v1" = the original Phase 1 baseline
    # End an episode after this many steps without progress. -1 = the reward version's
    # default (v2: 2000), 0 = never.
    stagnation_steps: int = -1
    # --- exploration (docs/rewards.md, docs/SETUP.md). All off by default.
    # Go-Explore style archive: probability that an episode starts from a state the
    # agent itself reached earlier, instead of the start state. Try 0.3-0.5.
    archive_prob: float = 0.0
    archive_progress_weight: float = 1.0  # prefer archived states where learning happens
    # Backward curriculum from the agent's own first-ever successes: probability that
    # an episode starts close to the end of a past success, then further back. Try 0.3.
    curriculum_prob: float = 0.0
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
    # --- final goal
    # Stop training as soon as an episode from the start state reaches the last
    # milestone (for Red: the Boulder Badge). Its replay is saved in
    # runs/<name>/replays, the model in checkpoints/winner.pt.
    stop_at_goal: bool = False
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


def make_env(cfg: Config, index: int) -> gym.Env:
    """Build one game. Defined at top level so subprocesses can create it."""
    state = cfg.state if Path(cfg.state).exists() else None
    rewards = RewardConfig.preset(cfg.reward_version)
    if cfg.stagnation_steps >= 0:
        rewards = rewards.with_weights(stagnation_steps=cfg.stagnation_steps)
    env_cfg = EnvConfig(
        rom_path=cfg.rom,
        start_state_path=state,
        max_steps=cfg.episode_steps,
        rewards=rewards,
        archive_prob=cfg.archive_prob,
        archive_progress_weight=cfg.archive_progress_weight,
        curriculum_prob=cfg.curriculum_prob,
        record_every=cfg.record_every,
        record_dir=str(Path("runs") / cfg.run_name / "recordings"),
        record_tag=f"env{index}",
        stop_at_goal=cfg.stop_at_goal,
        replay_dir=str(Path("runs") / cfg.run_name / "replays"),
    )
    if cfg.show and index == 0:
        # emulation_speed=0: the watched game must not slow down the other games,
        # which would all wait for it at every step.
        return PokemonEnv(env_cfg, ADAPTERS[cfg.game](), render_mode="human", emulation_speed=0)
    return PokemonEnv(env_cfg, ADAPTERS[cfg.game]())


def train(cfg: Config) -> None:
    # ------------------------------------------------------------------ setup
    if not Path(cfg.state).exists():
        print(
            f"\n!!! WARNING: start state {cfg.state} not found: every episode starts from "
            "the power-on screen (title screen, new game, names), not from the bedroom.\n"
        )
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
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    milestones = ADAPTERS[cfg.game].milestones

    obs_shape = envs.single_observation_space.shape
    n_actions = envs.single_action_space.n
    agent = CnnActorCritic(obs_shape, n_actions).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    batch_size = cfg.num_envs * cfg.num_steps
    sil = SelfImitationBuffer(cfg.sil_capacity, tuple(obs_shape)) if cfg.sil_coef > 0 else None
    highlights = (
        Highlights(run_dir / "gifs", cfg.gif_every_steps, cfg.gif_keep)
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

    # Rolling statistics over the last finished episodes, for the logs.
    recent_milestones: deque[np.ndarray] = deque(maxlen=50)
    best_reached = 0
    global_step = 0
    first_update = 1

    # Resume: restore the network, the optimizer and the counters, so a long run can be
    # split across several sessions and continue as if it had never stopped.
    if cfg.resume:
        saved = torch.load(cfg.resume, map_location=device)
        agent.load_state_dict(saved["model"])
        if "optimizer" in saved:
            optimizer.load_state_dict(saved["optimizer"])
        global_step = saved.get("global_step", 0)
        first_update = saved.get("update", 0) + 1
        best_reached = saved.get("best_reached", 0)
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

    def save(path: Path, update: int) -> None:
        _save(path, agent, optimizer, cfg, obs_shape, n_actions, global_step, update, best_reached)

    print(
        f"Run '{cfg.run_name}': {cfg.num_envs} games, {num_updates} updates of {batch_size} steps, "
        f"device {device}, {cfg.torch_threads} torch threads, reward {cfg.reward_version}, "
        f"archive {cfg.archive_prob:.0%}, curriculum {cfg.curriculum_prob:.0%}, "
        f"SIL {cfg.sil_coef}, stop at goal: {cfg.stop_at_goal}, "
        f"{sum(p.numel() for p in agent.parameters()):,} parameters"
    )

    next_obs, _ = envs.reset(seed=cfg.seed)
    next_obs = torch.as_tensor(next_obs, device=device)
    next_done = torch.zeros(cfg.num_envs, device=device)
    start, start_step = time.time(), global_step
    deadline = start + cfg.time_limit_hours * 3600 if cfg.time_limit_hours > 0 else None
    completed = first_update - 1  # last fully completed update

    goal = None  # (replay path, step) once an episode from the start reaches the goal
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
                    action, logp, _, value = agent.act(next_obs)
                act_buf[t], logp_buf[t], val_buf[t] = action, logp, value

                obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
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
                        if "goal_reached" in final and final["goal_reached"][i]:
                            goal = (str(final["replay"][i]), global_step)
                        if highlights is not None:
                            _add_highlight(highlights, final, i, global_step)
                        best_reached = _log_episode(
                            writer,
                            global_step,
                            info,
                            i,
                            milestones,
                            recent_milestones,
                            best_reached,
                        )

                if goal is not None:
                    break  # the final goal is reached: no need to finish this rollout

            play_time += time.time() - t_play
            if goal is not None:
                replay, step = goal
                save(run_dir / "checkpoints" / "winner.pt", update)
                save(run_dir / "checkpoints" / "latest.pt", update)
                print(f"\n*** FINAL GOAL REACHED at step {step:,}: {milestones[-1].name}! ***")
                print(f"    replay: {replay}")
                print(f"    model:  {run_dir / 'checkpoints' / 'winner.pt'}")
                print(f"    watch it: uv run python scripts/replay.py {replay}")
                break
            if highlights is not None:
                highlights.maybe_flush(global_step)
            t_learn = time.time()

            # ------------------------------------- 2. ADVANTAGES: how good was each action?
            # GAE: compare what actually happened (rewards) with what the critic
            # expected (values). Positive advantage = better than expected.
            with torch.no_grad():
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
                for idx in torch.randperm(batch_size, device=device).split(minibatch_size):
                    _, new_logp, entropy, new_value = agent.act(b_obs[idx], b_act[idx])
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
                sil.add(b_obs[best], b_act[best], b_ret[best])
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
                print(
                    f"update {update}/{num_updates}  steps {global_step:,}  SPS {sps}  "
                    f"(playing {share:.0%} of the time)  best milestones {best_reached}/{len(milestones)}"
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
        highlights.close()
    writer.close()
    print(f"Done. Checkpoints in {run_dir / 'checkpoints'}")


def _add_highlight(highlights, final, i, step) -> None:
    """Hand a finished, recorded episode to the highlights (if it was recorded)."""
    path = final.get("recording", [""] * (i + 1))[i]
    if not path:
        return
    reached = int((np.asarray(final["milestone_step"][i]) >= 0).sum())
    highlights.add(
        str(path), reached, int(final["maps_visited"][i]), int(final["tiles_visited"][i]), step
    )


class SelfImitationBuffer:
    """Past (observation, action, return) where the action did better than expected.

    Kept on the CPU as uint8, so 10,000 transitions take ~170 MB of RAM.
    """

    def __init__(self, capacity: int, obs_shape: tuple[int, ...]) -> None:
        self.obs = torch.zeros((capacity, *obs_shape), dtype=torch.uint8)
        self.act = torch.zeros(capacity, dtype=torch.long)
        self.ret = torch.zeros(capacity)
        self.capacity, self.size, self.ptr = capacity, 0, 0

    def __len__(self) -> int:
        return self.size

    def add(self, obs: torch.Tensor, act: torch.Tensor, ret: torch.Tensor) -> None:
        for o, a, r in zip(obs.cpu(), act.cpu(), ret.cpu(), strict=True):
            self.obs[self.ptr], self.act[self.ptr], self.ret[self.ptr] = o, a, r
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int):
        idx = torch.randint(0, self.size, (n,))
        return self.obs[idx], self.act[idx], self.ret[idx]


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
        obs, act, ret = (t.to(device) for t in sil.sample(minibatch_size))
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


def _log_episode(writer, step, info, i, milestones, recent, best_reached) -> int:
    """Log one finished episode: return, exploration and milestones."""
    final = info["final_info"]
    if "archive_cells" in final:
        writer.add_scalar("archive/cells", float(final["archive_cells"][i]), step)
    if "demos_active" in final:
        writer.add_scalar("curriculum/demos_active", float(final["demos_active"][i]), step)
        writer.add_scalar("curriculum/demos_completed", float(final["demos_completed"][i]), step)
    if "from_demo" in final and final["from_demo"][i]:
        # Started from a demo: not comparable with episodes from the start state.
        writer.add_scalar("curriculum/success", float(final["demo_success"][i]), step)
        writer.add_scalar("curriculum/demo_progress", float(final["demo_progress"][i]), step)
        return best_reached
    if "from_archive" in final and final["from_archive"][i]:
        # Started from an archived state: useful for learning, but not comparable
        # with episodes from the start state, so kept out of the main curves.
        writer.add_scalar("archive/episode_return", info["episode"]["r"][i], step)
        writer.add_scalar("archive/maps_visited", float(final["maps_visited"][i]), step)
        writer.add_scalar("archive/tiles_visited", float(final["tiles_visited"][i]), step)
        return best_reached
    writer.add_scalar("episode/return", info["episode"]["r"][i], step)
    writer.add_scalar("episode/length", info["episode"]["l"][i], step)
    writer.add_scalar("episode/tiles_visited", final["tiles_visited"][i], step)
    writer.add_scalar("episode/maps_visited", final["maps_visited"][i], step)
    for key in ("items_seen", "pokedex_owned", "pokedex_seen", "stagnated"):
        if key in final:
            writer.add_scalar(f"episode/{key}", float(final[key][i]), step)
    # Return of each reward component: if one dominates, watch the agent before
    # trusting the curves. That is how reward hacking is caught.
    totals = final.get("reward_totals", {})
    for name in totals:
        if not name.startswith("_"):
            writer.add_scalar(f"reward/{name}", float(totals[name][i]), step)

    reached_at = np.asarray(final["milestone_step"][i])
    recent.append(reached_at >= 0)
    rates = np.mean(recent, axis=0)
    for m, at, rate in zip(milestones, reached_at, rates, strict=True):
        writer.add_scalar(f"milestones/{m.id}_rate", rate, step)
        if at >= 0:
            writer.add_scalar(f"milestones/{m.id}_step", at, step)

    n_reached = int((reached_at >= 0).sum())
    if n_reached > best_reached:
        names = ", ".join(m.name for m, at in zip(milestones, reached_at, strict=True) if at >= 0)
        print(f"  new best at step {step:,}: {n_reached} milestones ({names})")
    return max(best_reached, n_reached)


def _save(path, agent, optimizer, cfg, obs_shape, n_actions, global_step, update, best_reached):
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
