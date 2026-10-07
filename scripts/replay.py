"""Replay a recorded game: the exact same game, frame by frame.

When training with --stop-at-goal, the episode that reaches the final goal
(e.g. the first badge) is saved as a replay: the start state plus every button
pressed. The emulator is deterministic, so replaying the buttons reproduces the
game exactly. This script plays it in a window at normal speed, or exports it as
a video.

Usage:
    uv run python scripts/replay.py runs/<run>/replays/goal_env0_ep12.npz              # window, normal speed
    uv run python scripts/replay.py <replay> --speed 4                                  # window, 4x
    uv run python scripts/replay.py <replay> --video badge.mp4                          # video, normal speed
    uv run python scripts/replay.py <replay> --video badge_4x.mp4 --video-speed 4       # shorter video
"""

import argparse
import io

import numpy as np
from missingno_envs import ACTIONS
from missingno_games import ADAPTERS
from pyboy import PyBoy


def main(args) -> None:
    data = np.load(args.replay)
    actions = data["actions"]
    frames_per_action = int(data["frames_per_action"])
    press_frames = int(data["press_frames"])
    game = str(data["game"])
    names = list(data["milestone_names"])
    adapter = ADAPTERS[game]() if game in ADAPTERS else None

    window = "null" if args.video else "SDL2"
    pyboy = PyBoy(args.rom, window=window, sound_emulated=False)
    pyboy.set_emulation_speed(0 if args.video else args.speed)
    pyboy.load_state(io.BytesIO(data["start_state"].tobytes()))
    pyboy.tick(1, True)

    writer = None
    if args.video:
        import imageio.v2 as imageio

        writer = imageio.get_writer(args.video, fps=60, macro_block_size=1)

    minutes = len(actions) * frames_per_action / 60 / 60
    print(f"{game}: {len(actions):,} actions, about {minutes:.0f} minutes of game time")
    reached = set()
    frame_count = 0
    for step, action in enumerate(actions, start=1):
        pyboy.button(ACTIONS[int(action)], press_frames)
        for _ in range(frames_per_action):
            if not pyboy.tick(1, True):
                print("Window closed.")
                return
            frame_count += 1
            if writer is not None and frame_count % args.video_speed == 0:
                writer.append_data(pyboy.screen.ndarray[:, :, :3])
        if adapter is None:
            continue
        signals = adapter.read(pyboy.memory)
        for m in adapter.milestones:
            if m.id not in reached and m.reached(signals):
                reached.add(m.id)
                print(
                    f"  step {step:6,d}  ({step * frames_per_action / 3600:5.1f} min)  {m.id} {m.name}"
                )

    if writer is not None:
        writer.close()
        print(f"Saved video: {args.video}")
    if adapter is not None:
        final = names[-1] if names else "goal"
        ok = adapter.milestones[-1].id in reached
        print(f"Replay {'reproduced' if ok else 'did NOT reproduce'} the final goal: {final}")
    if not args.video:
        print("Close the window to exit.")
        while pyboy.tick():
            pass
    pyboy.stop(save=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("replay", help="replay file (.npz) saved by training")
    parser.add_argument("--rom", default="roms/pokemon_red.gb")
    parser.add_argument(
        "--speed", type=int, default=1, help="window speed: 1 = normal, 0 = unlimited"
    )
    parser.add_argument("--video", help="export a video (.mp4) instead of opening a window")
    parser.add_argument(
        "--video-speed", type=int, default=1, help="keep one frame every N (N x faster)"
    )
    main(parser.parse_args())
