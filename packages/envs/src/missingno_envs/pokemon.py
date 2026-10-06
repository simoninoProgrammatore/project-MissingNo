"""Gymnasium environment for Pokémon games running in PyBoy.

Design rules (see docs/plan.md):
- The agent observes ONLY the screen (grayscale, downscaled, last frames stacked).
- Rewards are generic and identical for every game, computed from the progress
  signals read by a game adapter. The adapter never feeds the agent.
- No game-specific rewards, no story flags, no walkthrough knowledge.
"""

from __future__ import annotations

import hashlib
import io
import warnings
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

# The agent's buttons. SELECT is left out: it is almost never needed.
ACTIONS: tuple[str, ...] = ("down", "left", "right", "up", "a", "b", "start")


@dataclass
class RewardConfig:
    """Weights of the generic rewards. Same for every game."""

    new_tile: float = 0.02  # first visit to a tile
    new_map: float = 0.5  # first visit to a map (town, route, building, ...)
    badge: float = 5.0  # each new badge
    level: float = 0.2  # each new party level above the best seen so far


@dataclass
class EnvConfig:
    rom_path: str
    start_state_path: str | None = None  # None = boot from power-on
    frames_per_action: int = 24  # the agent acts once every N frames
    press_frames: int = 8  # how long a button is held
    downscale: int = 2  # 160x144 -> 80x72
    frame_stack: int = 3
    max_steps: int = 10_000  # episode length in agent steps
    rewards: RewardConfig | None = None


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
        self._visited_tiles: set[tuple[int, int, int]] = set()
        self._visited_maps: set[int] = set()
        self._best_badges = 0
        self._best_total_level = 0
        self._steps = 0
        # Step at which each milestone was first reached in this episode (-1 = not yet).
        self._milestone_step = np.full(len(adapter.milestones), -1, dtype=np.int64)

    # ------------------------------------------------------------------ API

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.pyboy.load_state(io.BytesIO(self._start_state))
        self.pyboy.tick(1, True)

        signals = self.adapter.read(self.pyboy.memory)
        self._visited_tiles = {signals.cell}
        self._visited_maps = {signals.map_id}
        self._best_badges = signals.badges
        self._best_total_level = signals.total_level
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
        self.pyboy.button(button, cfg.press_frames)
        # Render only the last frame: much faster, and it's the one we observe.
        self.pyboy.tick(cfg.frames_per_action - 1, False)
        self.pyboy.tick(1, True)
        self._steps += 1

        signals = self.adapter.read(self.pyboy.memory)
        reward, parts = self._reward(signals)
        self._update_milestones(signals)
        self._frames.append(self._grab_frame())

        terminated = False  # Pokémon has no "game over": only time limits
        truncated = self._steps >= cfg.max_steps
        return self._observation(), reward, terminated, truncated, self._info(signals, parts)

    def render(self):
        if self.render_mode == "rgb_array":
            return self.pyboy.screen.ndarray[:, :, :3].copy()
        return None  # "human": the PyBoy window is already showing the game

    def close(self) -> None:
        self.pyboy.stop(save=False)

    # ------------------------------------------------------------ internals

    def _reward(self, s: ProgressSignals) -> tuple[float, dict[str, float]]:
        r = self.rewards
        parts: dict[str, float] = {}
        if s.cell not in self._visited_tiles:
            self._visited_tiles.add(s.cell)
            parts["new_tile"] = r.new_tile
        if s.map_id not in self._visited_maps:
            self._visited_maps.add(s.map_id)
            parts["new_map"] = r.new_map
        if s.badges > self._best_badges:
            parts["badge"] = r.badge * (s.badges - self._best_badges)
            self._best_badges = s.badges
        # Reward only levels above the best ever seen: depositing and
        # re-withdrawing Pokémon must not be an exploitable loop.
        if s.total_level > self._best_total_level:
            parts["level"] = r.level * (s.total_level - self._best_total_level)
            self._best_total_level = s.total_level
        return sum(parts.values()), parts

    def _update_milestones(self, s: ProgressSignals) -> None:
        for i, milestone in enumerate(self.adapter.milestones):
            if self._milestone_step[i] < 0 and milestone.reached(s):
                self._milestone_step[i] = self._steps

    def _grab_frame(self) -> np.ndarray:
        rgb = self.pyboy.screen.ndarray[:, :, :3].astype(np.uint16)
        gray = (77 * rgb[..., 0] + 150 * rgb[..., 1] + 29 * rgb[..., 2]) >> 8
        k = self.config.downscale
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
            "tiles_visited": len(self._visited_tiles),
            "maps_visited": len(self._visited_maps),
            "reward_parts": parts,
            "steps": self._steps,
            "milestone_step": self._milestone_step.copy(),
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
