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


def replay(data):
    """Play a replay exactly as scripts/replay.py does, segment by segment."""
    pb = PyBoy(TEST_ROM, window="null", sound_emulated=False)
    pb.load_state(io.BytesIO(data["start_state"].tobytes()))
    pb.tick(1, True)
    starts = set(data["segment_starts"].tolist()) - {0}
    for i, a in enumerate(data["actions"]):
        if i in starts:
            pb.tick(1, True)
        pb.button(ACTIONS[int(a)], int(data["press_frames"]))
        for _ in range(int(data["frames_per_action"])):
            pb.tick(1, True)
    return pb


class GoalAfterReads(GoalAfterSteps):
    """Like GoalAfterSteps, and every step is a new cell, so the archive fills up."""

    def read(self, memory):
        self.reads += 1
        return self.S(
            map_id=memory[0xC000], x=self.reads % 200, y=0, badges=int(self.reads > self.goal_at)
        )


def test_a_goal_reached_from_an_archived_state_replays_from_the_start(tmp_path):
    """The lineage: archived state -> its parent episode -> the start state, in one replay."""
    adapter = GoalAfterReads(goal_at=10**9)  # no goal during the first episode
    env = PokemonEnv(
        EnvConfig(
            rom_path=TEST_ROM,
            stop_at_goal=True,
            replay_dir=str(tmp_path),
            max_steps=25,
            archive_prob=1.0,
        ),
        adapter,
    )
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    for _ in range(25):  # first episode, from the start: fills the archive
        env.step(int(rng.integers(len(ACTIONS))))
    adapter.goal_at = adapter.reads + 7  # the goal comes 7 steps into the next episode
    _, info = env.reset(seed=0)
    assert info["from_archive"]
    for _ in range(25):
        *_, truncated, info = env.step(int(rng.integers(len(ACTIONS))))
        if truncated:
            break
    assert info["goal_reached"] and info["replay"]
    trained_ram = ram(env.pyboy)
    env.close()

    data = np.load(info["replay"])
    assert len(data["segment_starts"]) == 2  # the archived state's history, then this episode
    pb = replay(data)
    assert ram(pb) == trained_ram
    pb.stop(save=False)


def test_shared_mode_two_games_one_hub_and_a_continuous_replay(tmp_path):
    """The training process's view: reports in, starting points out, replay from the hub."""
    from missingno_envs.exploration import ExplorationHub

    cfg = dict(
        rom_path=TEST_ROM,
        stop_at_goal=True,
        max_steps=30,
        archive_prob=0.5,
        curriculum_prob=0.5,
        curriculum_interval=4,
        shared_exploration=True,
    )
    a, b = GoalAfterReads(goal_at=10**9), GoalAfterReads(goal_at=10**9)
    envs = [PokemonEnv(EnvConfig(**cfg), adapter) for adapter in (a, b)]
    hub = ExplorationHub(archive_prob=0.5, curriculum_prob=0.5)
    rng = np.random.default_rng(0)
    for env in envs:
        env.reset(seed=0)
        for _ in range(30):
            env.step(int(rng.integers(len(ACTIONS))))
    for env in envs:
        hub.absorb(env.drain())
    assert len(hub.archive) >= 5  # cells are 4x4 tiles: 30 steps along a row
    assert all(env.archive is None for env in envs)  # no private archives in shared mode

    # Episodes now start from the hub's states; the goal comes a few steps in.
    env, adapter = envs[0], a
    env.pending_starts = [{"kind": "archive", **_entry(hub, rng)}]
    _, info = env.reset(seed=0)
    assert info["from_archive"]
    adapter.goal_at = adapter.reads + 5
    for _ in range(30):
        *_, truncated, info = env.step(int(rng.integers(len(ACTIONS))))
        if truncated:
            break
    assert info["goal_reached"] and not info["replay"]  # shared mode: the hub writes it
    report = info["goal_report"]
    path = hub.write_replay(
        tmp_path / "goal.npz", env.start_state(), report["node"], report["actions"],
        **env.replay_meta(),
    )  # fmt: skip
    trained_ram = ram(env.pyboy)
    data = np.load(path)
    pb = replay(data)
    assert ram(pb) == trained_ram
    pb.stop(save=False)
    for e in envs:
        e.close()


def _entry(hub, rng):
    key, entry = hub.archive.sample_entry(rng)
    return {"state": entry.state, "node": entry.node, "key": key}
