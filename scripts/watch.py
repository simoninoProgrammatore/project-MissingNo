"""Watch an agent play in a window: a random agent, or a trained one.

It is also the first sanity check of the environment: you should see the player
move, and the reward should grow when it reaches new tiles or maps.

Usage:
    uv run python scripts/watch.py --steps 2000                                  # random agent
    uv run python scripts/watch.py --checkpoint runs/ppo_s1/checkpoints/latest.pt
    uv run python scripts/watch.py --checkpoint ... --greedy --speed 0           # best action, max speed
    uv run python scripts/watch.py --checkpoint ... --no-window --gif best.gif   # save the episode as a GIF
"""

import argparse
from pathlib import Path

from missingno_envs import ACTIONS, COMPONENTS, EnvConfig, PokemonEnv, RewardConfig
from missingno_games import ADAPTERS, default_rom, default_state


def load_policy(checkpoint: str, greedy: bool):
    from missingno_agents import Policy

    return Policy(checkpoint, greedy)  # with or without memory, as it was trained


def main(args) -> None:
    config = EnvConfig(
        rom_path=args.rom,
        start_state_path=args.state,
        max_steps=args.steps,
        # Never end the episode early: we want to see everything the agent does,
        # loops included (training stops them, see docs/rewards.md).
        rewards=RewardConfig.preset(args.reward_version).with_weights(
            stagnation_steps=0, battle_stagnation_steps=0
        ),
    )
    env = PokemonEnv(
        config,
        ADAPTERS[args.game](),
        render_mode=None if args.no_window else "human",
        emulation_speed=args.speed,
    )
    policy = load_policy(args.checkpoint, args.greedy) if args.checkpoint else None
    milestones = env.adapter.milestones

    obs, info = env.reset(seed=0)
    frames = []  # for --gif: one frame every --gif-every steps
    total = 0.0
    reached = set()
    for step in range(1, args.steps + 1):
        action = policy(obs) if policy else env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if args.gif and step % args.gif_every == 0:
            frames.append(env.pyboy.screen.ndarray[:, :, :3].copy())
        total += reward
        if args.verbose and info["reward_parts"]:
            print(f"step {step:5d}  {ACTIONS[action]:>5}  +{reward:.2f}  {info['reward_parts']}")
        for m, at in zip(milestones, info["milestone_step"], strict=True):
            if at >= 0 and m.id not in reached:
                reached.add(m.id)
                print(f"*** step {step:5d}: {m.id} {m.name}")
        if step % 500 == 0:
            print(
                f"--- step {step}: return {total:.2f}, tiles {info['tiles_visited']}, "
                f"maps {info['maps_visited']}, map {info['map_id']}, pos {info['position']}"
            )
        if terminated or truncated:
            break
    print(
        f"Episode return: {total:.2f}  tiles: {info['tiles_visited']}  maps: {info['maps_visited']}"
    )
    print(f"Milestones reached: {len(reached)}/{len(milestones)}")
    totals = ", ".join(f"{k} {v:.2f}" for k, v in info["reward_totals"].items() if v)
    print(f"Reward by component: {totals or 'none'}")
    env.close()
    if args.gif:
        save_gif(frames, args.gif, args.gif_every)


def save_gif(frames, path: str, every: int) -> None:
    from PIL import Image

    if not frames:
        print("No frames to save.")
        return
    images = [Image.fromarray(f).resize((320, 288), Image.NEAREST) for f in frames]
    # 24 frames per step at 60 fps -> one agent step = 0.4 s of game time; play it 4x faster.
    duration_ms = max(20, int(every * 400 / 4))
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
    print(f"Saved {len(images)} frames to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--game", default="red", choices=ADAPTERS)
    parser.add_argument("--rom", help="default: roms/pokemon_<game>.gb (or .gbc)")
    parser.add_argument("--state", help="default: states/<game>_start.state")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--speed", type=int, default=4, help="1 = real time, 0 = unlimited")
    parser.add_argument("--checkpoint", help="trained model (.pt); omit for a random agent")
    parser.add_argument("--greedy", action="store_true", help="always pick the most likely button")
    parser.add_argument("--verbose", action="store_true", help="print every reward")
    parser.add_argument("--no-window", action="store_true", help="run without a window")
    parser.add_argument(
        "--reward-version",
        choices=sorted(COMPONENTS),
        help="default: the checkpoint's own version (v2.2 for a random agent)",
    )
    parser.add_argument("--gif", help="save the episode as a GIF at this path")
    parser.add_argument("--gif-every", type=int, default=2, help="keep one frame every N steps")
    args = parser.parse_args()

    args.rom = args.rom or default_rom(args.game)
    args.state = args.state or default_state(args.game)
    if args.checkpoint and not Path(args.checkpoint).exists():
        available = sorted(str(p) for p in Path("runs").glob("*/checkpoints/*.pt"))
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Available checkpoints:" if available else "No checkpoints in runs/ yet.")
        for path in available:
            print(f"  {path}")
        raise SystemExit(1)
    if not args.reward_version:
        args.reward_version = "v2.2"
        if args.checkpoint:
            import torch

            saved = torch.load(args.checkpoint, map_location="cpu")
            args.reward_version = saved.get("config", {}).get("reward_version", "v1")
    if not Path(args.state).exists():
        print(f"No start state at {args.state}: booting from power-on.")
        args.state = None
    main(args)
