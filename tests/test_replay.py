"""The replay reproduces the game exactly: same buttons, same emulator state."""

import io
import os
from pathlib import Path

import numpy as np
import pyboy
from missingno_envs import ACTIONS, EnvConfig, PokemonEnv
from pyboy import PyBoy

TEST_ROM = os.path.join(os.path.dirname(pyboy.__file__), "default_rom.gb")


class GoalAfterSteps:
    """A fake game whose final milestone is reached after a fixed number of reads."""

    name = "test"

    def __init__(self, goal_at):
        import hashlib

        from missingno_core import ProgressSignals
        from missingno_games import Milestone

        self.rom_sha1 = hashlib.sha1(Path(TEST_ROM).read_bytes()).hexdigest()
        self.reads, self.goal_at, self.S = 0, goal_at, ProgressSignals
        self.milestones = (Milestone("G", "Goal", lambda s: s.badges >= 1),)

    def read(self, memory):
        self.reads += 1
        return self.S(map_id=memory[0xC000], x=0, y=0, badges=int(self.reads > self.goal_at))


def ram(pb):
    return bytes(pb.memory[a] for a in range(0xC000, 0xE000))


def test_goal_saves_a_replay_that_reproduces_the_game(tmp_path):
    env = PokemonEnv(
        EnvConfig(rom_path=TEST_ROM, stop_at_goal=True, replay_dir=str(tmp_path), max_steps=1000),
        GoalAfterSteps(goal_at=30),
    )
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(100):
        *_, truncated, info = env.step(int(rng.integers(len(ACTIONS))))
        if truncated:
            break
    assert info["goal_reached"] and info["replay"]
    trained_ram = ram(env.pyboy)
    env.close()

    # Replay frame by frame, rendering every frame (as scripts/replay.py does).
    data = np.load(info["replay"])
    pb = PyBoy(TEST_ROM, window="null", sound_emulated=False)
    pb.load_state(io.BytesIO(data["start_state"].tobytes()))
    pb.tick(1, True)
    for a in data["actions"]:
        pb.button(ACTIONS[int(a)], int(data["press_frames"]))
        for _ in range(int(data["frames_per_action"])):
            pb.tick(1, True)
    assert len(data["actions"]) == 30
    assert ram(pb) == trained_ram
    pb.stop(save=False)
