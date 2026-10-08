"""Watch an agent play in a window: a random agent, or a trained one.

It is also the first sanity check of the environment: you should see the player
move, and the reward should grow when it reaches new tiles or maps.

While it plays, the terminal shows a live panel: the reward of each component
(total, how many times it paid, share of the return), the last rewards received,
the milestones reached and the next one, the team, badges and Pokédex. It is how
you see *why* the agent does what it does. `--no-live` goes back to plain lines.

Usage:
    uv run python scripts/watch.py --steps 2000                                  # random agent
    uv run python scripts/watch.py --checkpoint runs/ppo_s1/checkpoints/latest.pt
    uv run python scripts/watch.py --checkpoint ... --greedy --speed 0           # best action, max speed
    uv run python scripts/watch.py --checkpoint ... --no-window --gif best.gif   # save the episode as a GIF
"""

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

from missingno_envs import ACTIONS, COMPONENTS, EnvConfig, PokemonEnv, RewardConfig
from missingno_games import ADAPTERS, default_rom, default_state

FRAMES_PER_STEP = 24
GAME_FPS = 59.7


def load_policy(checkpoint: str, greedy: bool):
    from missingno_agents import Policy

    return Policy(checkpoint, greedy)  # with or without memory, as it was trained


class LivePanel:
    """A panel redrawn in place in the terminal, a few times per second."""

    BAR = 20

    def __init__(self, milestones, title: str, total_steps: int, refresh: float = 0.25) -> None:
        self.milestones = milestones
        self.title = title
        self.total_steps = total_steps
        self.refresh = refresh
        self.counts: dict[str, int] = {}  # how many steps each component paid
        self.recent: deque = deque(maxlen=8)  # (step, component, value)
        self.reached: dict[str, int] = {}  # milestone id -> step
        self.last_draw = 0.0
        self.started = False
        if os.name == "nt":
            os.system("")  # turns on ANSI escape codes in the Windows console

    def update(self, step: int, action: int, info: dict, total: float, force=False) -> None:
        for name, value in info["reward_parts"].items():
            if value and not name.startswith("_"):
                self.counts[name] = self.counts.get(name, 0) + 1
                self.recent.append((step, name, value))
        for m, at in zip(self.milestones, info["milestone_step"], strict=True):
            if at >= 0 and m.id not in self.reached:
                self.reached[m.id] = int(at)  # the step the environment saw it
        now = time.monotonic()
        if force or now - self.last_draw >= self.refresh:
            self.last_draw = now
            self.draw(step, action, info, total)

    def draw(self, step: int, action: int, info: dict, total: float) -> None:
        lines = self.render(step, action, info, total)
        if not self.started:
            sys.stdout.write("\x1b[2J")  # clear the screen once
            self.started = True
        # Back to the top-left corner, then every line overwrites the old one.
        sys.stdout.write("\x1b[H" + "".join(line + "\x1b[K\n" for line in lines) + "\x1b[J")
        sys.stdout.flush()

    def render(self, step: int, action: int, info: dict, total: float) -> list[str]:
        seconds = step * FRAMES_PER_STEP / GAME_FPS
        game_time = f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"
        x, y = info["position"]
        team = " ".join(f"L{lv}" for lv in info["party_levels"]) or "none"
        lines = [
            self.title,
            "",
            (
                f"step {step:,}/{self.total_steps:,}   game time {game_time}   "
                f"button {ACTIONS[action]:<5}   {'IN BATTLE' if info['in_battle'] else ''}"
            ),
            (
                f"map {info['map_id']}  pos ({x}, {y})   maps {info['maps_visited']}   "
                f"tiles {info['tiles_visited']}   items {info['items_seen']}"
            ),
            (
                f"team {team}   badges {info['badges']}   "
                f"Pokédex {info['pokedex_owned']} caught / {info['pokedex_seen']} seen"
            ),
            "",
            f"{'component':<14}{'':<{self.BAR}}  {'total':>8}  {'times':>6}  {'share':>6}",
        ]
        totals = {k: v for k, v in info["reward_totals"].items() if not k.startswith("_")}
        biggest = max((abs(v) for v in totals.values()), default=0.0) or 1.0
        positive = sum(v for v in totals.values() if v > 0) or 1.0
        for name, value in sorted(totals.items(), key=lambda kv: -abs(kv[1])):
            bar = "#" * round(self.BAR * abs(value) / biggest)
            share = f"{100 * value / positive:5.1f}%" if value > 0 else ""
            lines.append(
                f"{name:<14}{bar:<{self.BAR}}  {value:8.2f}  {self.counts.get(name, 0):6d}  "
                f"{share:>6}"
            )
        lines.append(f"{'return':<14}{'':<{self.BAR}}  {total:8.2f}")
        lines += ["", "last rewards:"]
        for at, name, value in reversed(self.recent):
            lines.append(f"  step {at:>7,}  {name:<14} {value:+.3f}")
        lines += ["", f"milestones {len(self.reached)}/{len(self.milestones)}:"]
        done = [m for m in self.milestones if m.id in self.reached]
        for m in done[-5:]:
            lines.append(f"  {m.id:<4} {m.name:<34} step {self.reached[m.id]:,}")
        upcoming = next((m for m in self.milestones if m.id not in self.reached), None)
        if upcoming is not None:
            lines.append(f"  next: {upcoming.id} {upcoming.name}")
        return lines


def main(args) -> None:
    policy = load_policy(args.checkpoint, args.greedy) if args.checkpoint else None
    config = EnvConfig(
        rom_path=args.rom,
        start_state_path=args.state,
        max_steps=args.steps,
        # The screen the model was trained on (full or half resolution).
        downscale=policy.downscale if policy else 2,
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
    milestones = env.adapter.milestones
    who = Path(args.checkpoint).name if args.checkpoint else "random agent"
    mode = "greedy" if args.greedy else "sampling"
    live = (
        LivePanel(milestones, f"MissingNo - {args.game} - {who} ({mode})", args.steps)
        if args.live and sys.stdout.isatty()
        else None
    )

    obs, info = env.reset(seed=0)
    frames = []  # for --gif: one frame every --gif-every steps
    total = 0.0
    reached = set()
    for step in range(1, args.steps + 1):
        action = policy(obs) if policy else env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if args.gif and step % args.gif_every == 0:
            frames.append(env.emulator.screen().copy())
        total += reward
        done = terminated or truncated
        if live is not None:
            live.update(step, action, info, total, force=done)
        else:
            if args.verbose and info["reward_parts"]:
                print(
                    f"step {step:5d}  {ACTIONS[action]:>5}  +{reward:.2f}  {info['reward_parts']}"
                )
            for m, at in zip(milestones, info["milestone_step"], strict=True):
                if at >= 0 and m.id not in reached:
                    reached.add(m.id)
                    print(f"*** step {step:5d}: {m.id} {m.name}")
            if step % 500 == 0:
                print(
                    f"--- step {step}: return {total:.2f}, tiles {info['tiles_visited']}, "
                    f"maps {info['maps_visited']}, map {info['map_id']}, pos {info['position']}"
                )
        if done:
            break
    reached_count = len(live.reached) if live is not None else len(reached)
    print()
    print(
        f"Episode return: {total:.2f}  tiles: {info['tiles_visited']}  maps: {info['maps_visited']}"
    )
    print(f"Milestones reached: {reached_count}/{len(milestones)}")
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
    h, w = frames[0].shape[:2]
    images = [Image.fromarray(f).resize((w * 2, h * 2), Image.NEAREST) for f in frames]
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
    parser.add_argument(
        "--live",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="live reward panel in the terminal (--no-live: plain lines)",
    )
    parser.add_argument("--verbose", action="store_true", help="with --no-live: print every reward")
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
