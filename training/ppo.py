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
from missingno_agents import CnnActorCritic
from missingno_envs import EnvConfig, PokemonEnv
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
    episode_steps: int = 8192  # length of one episode (~15 minutes of game time)
    # --- parallelism
    num_envs: int = 6  # parallel games: leave a core or two free for the OS
    num_steps: int = 256  # steps per game before each learning phase
    sync_envs: bool = False  # run games in the main process (slower, easier to debug)
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


def make_env(cfg: Config, index: int) -> gym.Env:
    """Build one game. Defined at top level so subprocesses can create it."""
    state = cfg.state if Path(cfg.state).exists() else None
    env_cfg = EnvConfig(rom_path=cfg.rom, start_state_path=state, max_steps=cfg.episode_steps)
    return PokemonEnv(env_cfg, ADAPTERS[cfg.game]())


def train(cfg: Config) -> None:
    # ------------------------------------------------------------------ setup
    run_dir = Path("runs") / cfg.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    writer = SummaryWriter(str(run_dir))

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

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

    print(
        f"Run '{cfg.run_name}': {cfg.num_envs} games, {num_updates} updates of {batch_size} steps, "
        f"device {device}, {sum(p.numel() for p in agent.parameters()):,} parameters"
    )

    next_obs, _ = envs.reset(seed=cfg.seed)
    next_obs = torch.as_tensor(next_obs, device=device)
    next_done = torch.zeros(cfg.num_envs, device=device)
    global_step = 0
    start = time.time()

    play_time = learn_time = 0.0
    for update in range(1, num_updates + 1):
        t_play = time.time()
        if cfg.anneal_lr:
            optimizer.param_groups[0]["lr"] = cfg.learning_rate * (1 - (update - 1) / num_updates)

        # ------------------------------------------------------------- 1. PLAY
        for t in range(cfg.num_steps):
            global_step += cfg.num_envs
            obs_buf[t] = next_obs
            done_buf[t] = next_done
            with torch.no_grad():
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
                    best_reached = _log_episode(
                        writer, global_step, info, i, milestones, recent_milestones, best_reached
                    )

        play_time += time.time() - t_play
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

        learn_time += time.time() - t_learn

        # ------------------------------------------------------------- logging
        sps = int(global_step / (time.time() - start))
        explained_var = 1 - torch.var(b_ret - b_val) / (torch.var(b_ret) + 1e-8)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("losses/policy", pg_loss.item(), global_step)
        writer.add_scalar("losses/value", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy", ent_loss.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("losses/explained_variance", explained_var.item(), global_step)

        if update % 10 == 0 or update == num_updates:
            share = play_time / (play_time + learn_time)
            print(
                f"update {update}/{num_updates}  steps {global_step:,}  SPS {sps}  "
                f"(playing {share:.0%} of the time)  best milestones {best_reached}/{len(milestones)}"
            )
        if update % cfg.checkpoint_every == 0 or update == num_updates:
            _save(
                agent, cfg, run_dir / "checkpoints" / f"step_{global_step}.pt", obs_shape, n_actions
            )
            _save(agent, cfg, run_dir / "checkpoints" / "latest.pt", obs_shape, n_actions)

    envs.close()
    writer.close()
    print(f"Done. Checkpoints in {run_dir / 'checkpoints'}")


def _log_episode(writer, step, info, i, milestones, recent, best_reached) -> int:
    """Log one finished episode: return, exploration and milestones."""
    final = info["final_info"]
    writer.add_scalar("episode/return", info["episode"]["r"][i], step)
    writer.add_scalar("episode/length", info["episode"]["l"][i], step)
    writer.add_scalar("episode/tiles_visited", final["tiles_visited"][i], step)
    writer.add_scalar("episode/maps_visited", final["maps_visited"][i], step)

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


def _save(agent, cfg, path, obs_shape, n_actions) -> None:
    torch.save(
        {
            "model": agent.state_dict(),
            "obs_shape": tuple(obs_shape),
            "n_actions": int(n_actions),
            "config": dataclasses.asdict(cfg),
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
