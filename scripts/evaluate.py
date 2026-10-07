"""Evaluate a trained model on any game: how far does it get, milestone by milestone?

Plays N episodes without a window and prints, for each milestone, the fraction of
episodes that reached it and the median step at which it was reached. This is
how we test transfer: a model trained on Red, evaluated on Blue or Yellow
without any training on them (zero-shot).

Usage:
    uv run python scripts/evaluate.py runs/badge_v22_s1/checkpoints/winner.pt --game yellow
    uv run python scripts/evaluate.py <checkpoint> --game red --episodes 20 --steps 30000
    uv run python scripts/evaluate.py <checkpoint> --game yellow --greedy
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from make_start_state import default_rom
from missingno_agents import CnnActorCritic
from missingno_envs import EnvConfig, PokemonEnv, RewardConfig
from missingno_games import ADAPTERS


def main(args) -> None:
    data = torch.load(args.checkpoint, map_location="cpu")
    agent = CnnActorCritic(tuple(data["obs_shape"]), data["n_actions"])
    agent.load_state_dict(data["model"])
    agent.eval()
    torch.set_num_threads(args.torch_threads)
    trained_on = data.get("config", {}).get("game", "?")
    reward_version = data.get("config", {}).get("reward_version", "v2")

    rom = args.rom or default_rom(args.game)
    state = args.state or f"states/{args.game}_start.state"
    if not Path(state).exists():
        raise SystemExit(
            f"No start state at {state}: create it with "
            f"uv run python scripts/make_start_state.py --game {args.game}"
        )
    env = PokemonEnv(
        EnvConfig(
            rom_path=rom,
            start_state_path=state,
            max_steps=args.steps,
            rewards=RewardConfig.preset(reward_version),
        ),
        ADAPTERS[args.game](),
    )
    milestones = env.adapter.milestones
    print(
        f"Model trained on {trained_on}, evaluated on {args.game}: "
        f"{args.episodes} episodes of {args.steps:,} steps ({'greedy' if args.greedy else 'sampled'})"
    )

    reached = np.full((args.episodes, len(milestones)), -1)
    maps = []
    for ep in range(args.episodes):
        start = time.time()
        obs, _ = env.reset(seed=args.seed + ep)
        for _ in range(args.steps):
            with torch.inference_mode():
                action, *_ = agent.act(torch.as_tensor(obs).unsqueeze(0), greedy=args.greedy)
            obs, _, terminated, truncated, info = env.step(int(action.item()))
            if terminated or truncated:
                break
        reached[ep] = info["milestone_step"]
        maps.append(info["maps_visited"])
        done = [m.id for m, s in zip(milestones, reached[ep], strict=True) if s >= 0]
        last = done[-1] if done else "-"  # milestones are in game order
        print(
            f"  episode {ep + 1:2d}: furthest {last:>4}, {info['maps_visited']} maps, "
            f"{info['tiles_visited']} tiles  ({time.time() - start:.0f}s)"
        )
    env.close()

    print(f"\n{'milestone':<32} {'reached':>8} {'median step':>12}")
    for i, m in enumerate(milestones):
        hits = reached[:, i][reached[:, i] >= 0]
        median = f"{int(np.median(hits)):,}" if len(hits) else "-"
        print(f"{m.id + ' ' + m.name:<32} {len(hits) / args.episodes:>7.0%} {median:>12}")
    print(f"\nMaps visited per episode: mean {np.mean(maps):.1f}, max {max(maps)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--game", default="red", choices=ADAPTERS)
    parser.add_argument("--rom", help="default: roms/pokemon_<game>.gb (or .gbc)")
    parser.add_argument("--state", help="default: states/<game>_start.state")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20_000, help="max steps per episode")
    parser.add_argument("--greedy", action="store_true", help="always pick the most likely button")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=2)
    main(parser.parse_args())
