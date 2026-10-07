"""Gymnasium environment for Pokémon games running in PyBoy.

Design rules (see docs/research.md and docs/rewards.md):
- The agent observes ONLY the screen (grayscale, downscaled, last frames stacked).
- Rewards are generic and identical for every game, computed from the progress
  signals read by a game adapter. The adapter never feeds the agent.
- No game-specific rewards, no story flags, no walkthrough knowledge.
"""

from __future__ import annotations

import hashlib
import io
import warnings
import zlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from missingno_core import ProgressSignals
from missingno_games import GameAdapter
from pyboy import PyBoy

from missingno_envs.archive import StateArchive
from missingno_envs.curriculum import BackwardCurriculum, Demo
from missingno_envs.rewards import RewardConfig, RewardTracker

# The agent's buttons. SELECT is left out: it is almost never needed.
ACTIONS: tuple[str, ...] = ("down", "left", "right", "up", "a", "b", "start")


@dataclass
class EnvConfig:
    rom_path: str
    start_state_path: str | None = None  # None = boot from power-on
    frames_per_action: int = 24  # the agent acts once every N frames
    press_frames: int = 8  # how long a button is held
    downscale: int = 2  # 160x144 -> 80x72
    frame_stack: int = 3
    max_steps: int = 10_000  # episode length in agent steps
    rewards: RewardConfig | None = None  # None = the current default version (v2)
    # Archive of states reached by the agent (Go-Explore style, see archive.py).
    # Probability that an episode starts from an archived state instead of the
    # start state. 0 = off.
    archive_prob: float = 0.0
    archive_max_cells: int = 1000
    archive_progress_weight: float = 1.0  # 0 = choose archived states by counts only
    # Backward curriculum from the agent's own first-ever successes (see
    # curriculum.py). Probability that an episode starts from a demo. 0 = off.
    curriculum_prob: float = 0.0
    curriculum_interval: int = 64  # steps between saved states along each episode
    # Recording for GIFs: keep one full-resolution grayscale frame every N steps of
    # each episode started from the start state, and save it when the episode ends.
    # 0 = off. Recordings are compressed .npz files in `record_dir`.
    record_every: int = 0
    record_dir: str | None = None
    record_tag: str = "env"
    # Final goal: when an episode from the start state reaches the adapter's last
    # milestone (e.g. the first badge), the episode ends and its replay is saved:
    # start state + every button pressed. The emulator is deterministic, so the
    # replay reproduces the exact game (scripts/replay.py), at any speed.
    stop_at_goal: bool = False
    replay_dir: str | None = None


class PokemonEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        config: EnvConfig,
        adapter: GameAdapter,
        render_mode: str | None = None,
        emulation_speed: int = 1,
    ) -> None:
        super().__init__()
        self.config = config
        self.adapter = adapter
        self.rewards = config.rewards or RewardConfig()
        self._tracker = RewardTracker(self.rewards)
        self.archive = (
            StateArchive(max_cells=config.archive_max_cells) if config.archive_prob > 0 else None
        )
        if self.archive is not None:
            self.archive.progress_weight = config.archive_progress_weight
        self._from_archive = False
        self._archive_cell = None
        self.curriculum = (
            BackwardCurriculum(interval=config.curriculum_interval)
            if config.curriculum_prob > 0
            else None
        )
        self._demo: Demo | None = None
        self._snapshots: list[bytes] = []
        self._maps_ever: set[int] = set()  # maps reached in any episode so far
        self._recording: list[np.ndarray] | None = None
        self._episode_index = 0
        self._last_recording = ""
        self._actions: list[int] | None = None
        self._goal_reached = False
        self._last_replay = ""
        self.render_mode = render_mode

        _check_rom(config.rom_path, adapter)
        window = "SDL2" if render_mode == "human" else "null"
        self.pyboy = PyBoy(config.rom_path, window=window, sound_emulated=False)
        # 0 = as fast as possible; 1 = real time (only useful when watching)
        self.pyboy.set_emulation_speed(emulation_speed if render_mode == "human" else 0)

        if config.start_state_path:
            self._start_state = Path(config.start_state_path).read_bytes()
        else:
            # No start state: snapshot the power-on state, so every reset is identical.
            buffer = io.BytesIO()
            self.pyboy.save_state(buffer)
            self._start_state = buffer.getvalue()

        height = 144 // config.downscale
        width = 160 // config.downscale
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(config.frame_stack, height, width), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(len(ACTIONS))

        self._frames: deque[np.ndarray] = deque(maxlen=config.frame_stack)
        self._steps = 0
        # Step at which each milestone was first reached in this episode (-1 = not yet).
        self._milestone_step = np.full(len(adapter.milestones), -1, dtype=np.int64)

    # ------------------------------------------------------------------ API

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        cfg = self.config
        # Where does this episode start? A demo, an archived state, or the start state.
        r = self.np_random.random()
        self._demo, self._archive_cell, self._from_archive = None, None, False
        if self.curriculum is not None and len(self.curriculum) > 0 and r < cfg.curriculum_prob:
            self._demo, state = self.curriculum.sample(self.np_random)
        elif (
            self.archive is not None
            and len(self.archive) > 0
            and r < cfg.curriculum_prob + cfg.archive_prob
        ):
            self._archive_cell, state = self.archive.sample(self.np_random)
            self._from_archive = True
        else:
            state = self._start_state
        self.pyboy.load_state(io.BytesIO(state))
        self.pyboy.tick(1, True)
        self._snapshots = [zlib.compress(state, 1)] if self.curriculum is not None else []
        # Record only episodes from the start state: they show real progress.
        from_start = self._demo is None and not self._from_archive
        self._recording = [] if cfg.record_every > 0 and cfg.record_dir and from_start else None
        self._last_recording = ""
        self._episode_index += 1
        self._actions: list[int] | None = [] if cfg.stop_at_goal and from_start else None
        self._goal_reached = False
        self._last_replay = ""

        signals = self.adapter.read(self.pyboy.memory)
        self._maps_ever.add(signals.map_id)
        self._tracker.reset(signals)
        self._steps = 0
        self._milestone_step[:] = -1
        self._update_milestones(signals)

        frame = self._grab_frame()
        self._frames.clear()
        for _ in range(self.config.frame_stack):
            self._frames.append(frame)
        return self._observation(), self._info(signals, {})

    def step(self, action: int):
        button = ACTIONS[int(action)]
        cfg = self.config
        if self._actions is not None:
            self._actions.append(int(action))
        self.pyboy.button(button, cfg.press_frames)
        # Render only the last frame: much faster, and it's the one we observe.
        self.pyboy.tick(cfg.frames_per_action - 1, False)
        self.pyboy.tick(1, True)
        self._steps += 1

        signals = self.adapter.read(self.pyboy.memory)
        reward, parts = self._tracker.step(signals, self._steps)
        self._update_milestones(signals)
        if self.archive is not None:
            self.archive.observe(signals, self._save_state)
        demo_success = self._curriculum_step(signals)
        self._frames.append(self._grab_frame())
        if self._recording is not None and self._steps % cfg.record_every == 0:
            self._recording.append(self._grab_frame(downscale=1))

        terminated = False  # Pokémon has no "game over": only time limits
        stagnated = self._tracker.stagnant(self._steps)
        max_steps = self._demo.budget() if self._demo is not None else cfg.max_steps
        if self._actions is not None and self._milestone_step[-1] >= 0:
            self._goal_reached = True  # final milestone, in an episode from the start
            self._save_replay()
        truncated = self._steps >= max_steps or stagnated or demo_success or self._goal_reached
        if truncated:
            self._end_episode(demo_success)
        info = self._info(signals, parts)
        info["demo_success"] = demo_success
        info["stagnated"] = stagnated
        return self._observation(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.pyboy.screen.ndarray[:, :, :3].copy()
        return None  # "human": the PyBoy window is already showing the game

    def close(self) -> None:
        self.pyboy.stop(save=False)

    # ------------------------------------------------------------ internals

    def _curriculum_step(self, s: ProgressSignals) -> bool:
        """Save states along the way; turn a first-ever new map into a demo.

        Returns True if this episode started from a demo and just reached its goal.
        """
        if self.curriculum is None:
            self._maps_ever.add(s.map_id)
            return False
        cur = self.curriculum
        if self._steps % cur.interval == 0 and len(self._snapshots) < cur.max_snapshots:
            self._snapshots.append(zlib.compress(self._save_state(), 1))
        if s.map_id not in self._maps_ever:
            self._maps_ever.add(s.map_id)
            if self._demo is None:  # a demo episode just repeats a known success
                # A snapshot taken at this very step is already at the goal: skip it.
                taken_now = self._steps % cur.interval == 0 and len(self._snapshots) > 1
                cur.add_demo(self._snapshots[:-1] if taken_now else self._snapshots, s.map_id)
        return self._demo is not None and s.map_id == self._demo.target_map

    def _end_episode(self, demo_success: bool) -> None:
        """Report the outcome of the episode to the archive and the curriculum."""
        if self._demo is not None and self.curriculum is not None:
            self.curriculum.record(self._demo, demo_success)
        if self._archive_cell is not None and self.archive is not None:
            self.archive.record_outcome(self._archive_cell, sum(self._tracker.totals.values()))
        if self._recording:
            path = (
                Path(self.config.record_dir)
                / f"{self.config.record_tag}_ep{self._episode_index}.npz"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, frames=np.stack(self._recording))
            self._last_recording = str(path)
            self._recording = None

    def _save_replay(self) -> None:
        cfg = self.config
        folder = Path(cfg.replay_dir or ".")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"goal_{cfg.record_tag}_ep{self._episode_index}.npz"
        np.savez_compressed(
            path,
            actions=np.asarray(self._actions, dtype=np.uint8),
            start_state=np.frombuffer(self._start_state, dtype=np.uint8),
            frames_per_action=cfg.frames_per_action,
            press_frames=cfg.press_frames,
            game=self.adapter.name,
            milestone_step=self._milestone_step,
            milestone_names=np.array([m.name for m in self.adapter.milestones]),
        )
        self._last_replay = str(path)
        self._actions = None

    def _save_state(self) -> bytes:
        buffer = io.BytesIO()
        self.pyboy.save_state(buffer)
        return buffer.getvalue()

    def _update_milestones(self, s: ProgressSignals) -> None:
        for i, milestone in enumerate(self.adapter.milestones):
            if self._milestone_step[i] < 0 and milestone.reached(s):
                self._milestone_step[i] = self._steps

    def _grab_frame(self, downscale: int | None = None) -> np.ndarray:
        rgb = self.pyboy.screen.ndarray[:, :, :3].astype(np.uint16)
        gray = (77 * rgb[..., 0] + 150 * rgb[..., 1] + 29 * rgb[..., 2]) >> 8
        k = self.config.downscale if downscale is None else downscale
        return gray[::k, ::k].astype(np.uint8)

    def _observation(self) -> np.ndarray:
        return np.stack(self._frames, axis=0)

    def _info(self, s: ProgressSignals, parts: dict[str, float]) -> dict[str, Any]:
        return {
            "map_id": s.map_id,
            "position": (s.x, s.y),
            "badges": s.badges,
            "party_levels": s.party_levels,
            "in_battle": s.in_battle,
            "tiles_visited": len(self._tracker.visited_tiles),
            "maps_visited": len(self._tracker.visited_maps),
            "items_seen": len(self._tracker.items_seen),
            "pokedex_owned": s.pokedex_owned,
            "pokedex_seen": s.pokedex_seen,
            "reward_parts": parts,
            # Episode return of each reward component, to audit which one drives learning.
            "reward_totals": dict(self._tracker.totals),
            "steps": self._steps,
            "milestone_step": self._milestone_step.copy(),
            # Episodes started from the archive do not count for the milestone curves.
            "from_archive": self._from_archive,
            "archive_cells": len(self.archive) if self.archive is not None else 0,
            "from_demo": self._demo is not None,
            "demos_active": len(self.curriculum) if self.curriculum is not None else 0,
            "demos_completed": self.curriculum.completed if self.curriculum is not None else 0,
            "demo_progress": self._demo.progress if self._demo is not None else 0.0,
            "recording": self._last_recording,
            "goal_reached": self._goal_reached,
            "replay": self._last_replay,
        }


def _check_rom(rom_path: str, adapter: GameAdapter) -> None:
    path = Path(rom_path)
    if not path.exists():
        raise FileNotFoundError(
            f"ROM not found: {rom_path}. Put your legally obtained ROM in roms/ (see docs/SETUP.md)."
        )
    sha1 = hashlib.sha1(path.read_bytes()).hexdigest()
    if sha1 != adapter.rom_sha1:
        warnings.warn(
            f"ROM SHA-1 {sha1} does not match the expected {adapter.rom_sha1} for "
            f"'{adapter.name}'. Memory addresses may be wrong for this version.",
            stacklevel=2,
        )
