"""Find out where training time goes: emulators, choosing actions, or learning.

Runs four short measurements (~1 minute in total) and prints a table.

Usage:
    uv run python scripts/diagnose_speed.py
    uv run python scripts/diagnose_speed.py --envs 1 2 4 6 --torch-threads 2
"""

import argparse
import time
from functools import partial
from pathlib import Path

import gymnasium as gym
import torch
from gymnasium.vector import AutoresetMode
from missingno_agents import CnnActorCritic
from missingno_envs import EnvConfig, PokemonEnv
from missingno_games import ADAPTERS


def make_env(rom: str, state: str | None, game: str) -> gym.Env:
    return PokemonEnv(EnvConfig(rom_path=rom, start_state_path=state), ADAPTERS[game]())


def vector_sps(rom, state, game, n_envs: int, seconds: float) -> float:
    """Raw speed of N games in parallel processes, random actions, no network."""
    envs = gym.vector.AsyncVectorEnv(
        [partial(make_env, rom, state, game) for _ in range(n_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
        context="spawn",
    )
    envs.reset(seed=0)
    for _ in range(20):  # warm-up
        envs.step(envs.action_space.sample())
    steps, start = 0, time.perf_counter()
    while time.perf_counter() - start < seconds:
        envs.step(envs.action_space.sample())
        steps += n_envs
    elapsed = time.perf_counter() - start
    envs.close()
    return steps / elapsed


def main(args) -> None:
    state = args.state if Path(args.state).exists() else None
    torch.set_num_threads(args.torch_threads)
    print(f"torch threads: {args.torch_threads}\n")

    # 1. One game, no multiprocessing: the reference.
    env = make_env(args.rom, state, args.game)
    env.reset(seed=0)
    n, start = 0, time.perf_counter()
    while time.perf_counter() - start < args.seconds:
        env.step(env.action_space.sample())
        n += 1
    single = n / (time.perf_counter() - start)
    obs_shape, n_actions = env.observation_space.shape, env.action_space.n
    env.close()
    print(f"[1] one game, same process:        {single:8,.0f} steps/s")

    # 2. N games in parallel processes.
    print("[2] games in parallel processes (no network):")
    for k in args.envs:
        sps = vector_sps(args.rom, state, args.game, k, args.seconds)
        print(
            f"      {k} games: {sps:8,.0f} steps/s   ({sps / k:,.0f} per game, ideal {single:,.0f})"
        )

    # 3. Choosing actions: one forward pass of the network per step.
    agent = CnnActorCritic(obs_shape, n_actions)
    k = max(args.envs)
    obs = torch.randint(0, 255, (k, *obs_shape), dtype=torch.uint8)
    with torch.inference_mode():
        for _ in range(10):
            agent.act(obs)
        reps, start = 200, time.perf_counter()
        for _ in range(reps):
            agent.act(obs)
    ms = (time.perf_counter() - start) / reps * 1000
    print(f"[3] choosing actions for {k} games:   {ms:8.2f} ms per step")

    # 4. One PPO learning phase (same sizes as training/ppo.py defaults).
    batch = k * args.num_steps
    optimizer = torch.optim.Adam(agent.parameters(), lr=2.5e-4)
    b_obs = torch.randint(0, 255, (batch, *obs_shape), dtype=torch.uint8)
    b_act = torch.randint(0, n_actions, (batch,))
    start = time.perf_counter()
    for _ in range(args.update_epochs):
        for idx in torch.randperm(batch).split(batch // 4):
            _, logp, ent, val = agent.act(b_obs[idx], b_act[idx])
            loss = -logp.mean() + val.pow(2).mean() - 0.01 * ent.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    learn = time.perf_counter() - start
    print(f"[4] one learning phase ({batch} steps x {args.update_epochs} epochs): {learn:6.2f} s")

    play = batch / sps + batch / k * ms / 1000
    print(
        f"\nEstimate with {k} games: playing {play:.1f} s + learning {learn:.1f} s per update "
        f"-> ~{batch / (play + learn):,.0f} steps/s overall"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--game", default="red", choices=ADAPTERS)
    parser.add_argument("--rom", default="roms/pokemon_red.gb")
    parser.add_argument("--state", default="states/red_start.state")
    parser.add_argument("--envs", type=int, nargs="+", default=[1, 2, 4, 6])
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=4)
    main(parser.parse_args())
