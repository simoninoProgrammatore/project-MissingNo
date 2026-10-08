"""Gymnasium environment for Pokémon games (PyBoy for Game Boy, mGBA for GBA).

Design rules (see docs/research.md and docs/rewards.md):
- The agent observes ONLY the screen (grayscale, downscaled, last frames stacked).
- Rewards are generic and identical for every game, computed from the progress
  signals read by a game adapter. The adapter never feeds the agent.
- No game-specific rewards, no story flags, no walkthrough knowledge.
"""

from __future__ import annotations

import hashlib
import warnings
import zlib
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from missingno_core import ProgressSignals
from missingno_games import GameAdapter, goal_index

from missingno_envs.archive import cell_key
from missingno_envs.emulator import make_emulator
from missingno_envs.exploration import START, ExplorationHub, compress, write_replay
from missingno_envs.rewards import RewardConfig, RewardTracker

# The agent's buttons. SELECT is left out: it is almost never needed. The Game Boy
# Advance's L and R are left out too, so the same agent plays every console.
ACTIONS: tuple[str, ...] = ("down", "left", "right", "up", "a", "b", "start")

# Every screen is brought to the Game Boy's size before the agent sees it, so the same
# network plays every console (the Game Boy Advance's 240x160 screen is resized).
SCREEN_HEIGHT, SCREEN_WIDTH = 144, 160


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
    curriculum_max_snapshots: int = 400  # per episode; then they are thinned out
    # Shared exploration (exploration.py): the archive and the curriculum live in the
    # training process, shared by all parallel games. The environment reports what it
    # finds (`drain`) and starts its episodes from `pending_starts`. False = this
    # environment keeps its own archive and curriculum (one game alone, tests, scripts).
    shared_exploration: bool = False
    # Recording for GIFs: keep one full-resolution grayscale frame every N steps of
    # each episode started from the start state, and save it when the episode ends.
    # 0 = off. Recordings are compressed .npz files in `record_dir`.
    record_every: int = 0
    record_dir: str | None = None
    record_tag: str = "env"
    # Final goal: when an episode reaches the goal milestone (`goal`, default the
    # adapter's own, e.g. the first badge), the episode ends and its replay is saved:
    # start state + every button pressed, from the bedroom, even if the episode started
    # from an archived state (its lineage is replayed first, see exploration.py). The
    # emulator is deterministic, so the replay reproduces the exact game.
    stop_at_goal: bool = False
    replay_dir: str | None = None
    goal: str | None = None  # a milestone id, e.g. "M12"; None = the adapter's goal


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
        self._use_archive = config.archive_prob > 0
        self._use_curriculum = config.curriculum_prob > 0
        # Local mode: a private hub. Shared mode: the training process has the hub.
        self.hub = (
            ExplorationHub(
                archive_prob=config.archive_prob,
                curriculum_prob=config.curriculum_prob,
                archive_max_cells=config.archive_max_cells,
                archive_progress_weight=config.archive_progress_weight,
                curriculum_interval=config.curriculum_interval,
            )
            if (self._use_archive or self._use_curriculum) and not config.shared_exploration
            else None
        )
        if self.hub is not None and self.hub.curriculum is not None:
            self.hub.curriculum.max_snapshots = config.curriculum_max_snapshots
        self.pending_starts: list[dict] = []  # shared mode: set by the training process
        self.known_maps: set[int] = set()  # shared mode: maps any game has reached
        self._outbox: dict = defaultdict(list)
        self._visits: Counter = Counter()
        self._known_cells: set = set()  # cells this environment already reported
        self._maps_ever: set[int] = set()  # maps reached in any episode so far
        self._spec: dict = dict(START)
        self._node = 0  # lineage node this episode started from
        self._actions = bytearray()  # buttons pressed in this episode
        self._snapshots: list[tuple[int, bytes]] = []  # (step, compressed state)
        self._snap_interval = config.curriculum_interval
        self._recording: list[np.ndarray] | None = None
        self._episode_index = 0
        self._last_recording = ""
        self._goal = goal_index(adapter, config.goal)
        self._goal_reached = False
        self._goal_report: dict | None = None
        self._last_replay = ""
        self.render_mode = render_mode

        _check_rom(config.rom_path, adapter)
        # PyBoy for Game Boy games, mGBA for Game Boy Advance ones (see emulator.py). Some
        # adapters force the classic Game Boy mode (e.g. Yellow), see games/red.py.
        self.emulator = make_emulator(config.rom_path, adapter, window=render_mode == "human")
        # 0 = as fast as possible; 1 = real time (only useful when watching)
        self.emulator.set_speed(emulation_speed if render_mode == "human" else 0)

        if config.start_state_path:
            self._start_state = Path(config.start_state_path).read_bytes()
            try:
                self.emulator.load_state(self._start_state)
            except Exception as error:
                raise RuntimeError(
                    f"Cannot load the start state {config.start_state_path} with "
                    f"{self.emulator.version()}: {error}\n"
                    "Save states are NOT compatible across emulator versions. Use the version "
                    "that created it (see uv.lock), or recreate it with "
                    "scripts/make_start_state.py."
                ) from error
        else:
            # No start state: snapshot the power-on state, so every reset is identical.
            self._start_state = self.emulator.save_state()

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
        self._spec = self._next_start()
        kind = self._spec["kind"]
        state = self._start_state if kind == "start" else zlib.decompress(self._spec["state"])
        self._node = self._spec.get("node", 0)
        self.emulator.load_state(state)
        self.emulator.tick(1, True)
        self._actions = bytearray()
        self._snap_interval = cfg.curriculum_interval
        self._snapshots = [(0, compress(state))] if self._use_curriculum else []
        # Record only episodes from the start state: they show real progress.
        self._recording = (
            [] if cfg.record_every > 0 and cfg.record_dir and kind == "start" else None
        )
        self._last_recording = ""
        self._episode_index += 1
        self._goal_reached = False
        self._goal_report = None
        self._last_replay = ""

        signals = self.adapter.read(self.emulator.memory)
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
        self._actions.append(int(action))
        self.emulator.press(button, cfg.press_frames)
        # Render only the last frame: much faster, and it's the one we observe.
        self.emulator.tick(cfg.frames_per_action - 1, False)
        self.emulator.tick(1, True)
        self._steps += 1

        signals = self.adapter.read(self.emulator.memory)
        reward, parts = self._tracker.step(signals, self._steps)
        self._update_milestones(signals)
        if self._use_archive:
            self._archive_step(signals)
        demo_success = self._curriculum_step(signals)
        self._frames.append(self._grab_frame())
        if self._recording is not None and self._steps % cfg.record_every == 0:
            self._recording.append(self._grab_frame(downscale=1))

        terminated = False  # Pokémon has no "game over": only time limits
        stop_reason = self._tracker.stop_reason(self._steps)
        stagnated = stop_reason != ""
        demo = self._spec["kind"] == "demo"
        max_steps = self._spec["budget"] if demo else cfg.max_steps
        # The goal counts in any episode, as long as it was reached during the episode
        # (an archived state may already have it at step 0).
        if not self._goal_reached and self._milestone_step[self._goal] > 0 and cfg.stop_at_goal:
            self._goal_reached = True
            self._on_goal()
        truncated = self._steps >= max_steps or stagnated or demo_success or self._goal_reached
        if truncated:
            self._end_episode(demo_success)
        info = self._info(signals, parts)
        info["demo_success"] = demo_success
        info["stagnated"] = stagnated
        # Ended stuck in a battle (e.g. choosing RUN in a trainer battle, forever).
        info["battle_loop"] = stop_reason == "battle"
        if self._goal_report is not None:
            info["goal_report"] = self._goal_report
        return self._observation(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.emulator.screen().copy()
        return None  # "human": the emulator's window is already showing the game

    def close(self) -> None:
        self.emulator.stop()

    @property
    def pyboy(self):
        """Old name of `emulator`, kept for scripts written before the Game Boy Advance."""
        return self.emulator

    @property
    def archive(self):
        """This environment's own archive (local mode), None otherwise."""
        return self.hub.archive if self.hub is not None else None

    @property
    def curriculum(self):
        return self.hub.curriculum if self.hub is not None else None

    # ------------------------------------------------- shared exploration protocol

    def drain(self) -> dict:
        """Shared mode: everything found since the last call (see ExplorationHub.absorb)."""
        report = dict(self._outbox)
        report["visits"] = dict(self._visits)
        report["maps"] = set(self._maps_ever)
        self._outbox = defaultdict(list)
        self._visits = Counter()
        return report

    def start_state(self) -> bytes:
        return self._start_state

    # ------------------------------------------------------------ internals

    def _next_start(self) -> dict:
        if self.config.shared_exploration:
            return self.pending_starts.pop(0) if self.pending_starts else dict(START)
        if self.hub is not None:
            return self.hub.next_start(self.np_random)
        return dict(START)

    def _emit(self, kind: str, item) -> None:
        """Report a discovery: to the shared hub (later) or to the private one (now)."""
        if self.hub is not None:
            self.hub.absorb({kind: [item]})
        else:
            self._outbox[kind].append(item)

    def _archive_step(self, s: ProgressSignals) -> None:
        key = cell_key(s)
        if key in self._known_cells:
            if self.hub is not None:
                self.hub.archive.add_visits(key, 1)
            else:
                self._visits[key] += 1
            return
        self._known_cells.add(key)
        self._emit("cells", (key, compress(self._save_state()), self._node, self._lineage()))

    def _lineage(self) -> bytes:
        return compress(self._actions)

    def _curriculum_step(self, s: ProgressSignals) -> bool:
        """Save states along the way; turn a first-ever new map into a demo.

        Returns True if this episode started from a demo and just reached its goal.
        """
        demo_episode = self._spec["kind"] == "demo"
        new_map = s.map_id not in self._maps_ever and s.map_id not in self.known_maps
        self._maps_ever.add(s.map_id)
        if not self._use_curriculum:
            return False
        if self._steps % self._snap_interval == 0:
            self._snapshots.append((self._steps, compress(self._save_state())))
            if len(self._snapshots) > self.config.curriculum_max_snapshots:
                # Long episode: keep every other snapshot, and save half as often from now.
                self._snapshots = self._snapshots[::2]
                self._snap_interval *= 2
        if new_map and not demo_episode:  # a demo episode just repeats a known success
            # A snapshot taken at this very step is already at the goal: leave it out.
            snaps = [(o, st) for o, st in self._snapshots if o < self._steps] or self._snapshots
            self._emit(
                "demos",
                (
                    s.map_id,
                    self._node,
                    self._lineage(),
                    [st for _, st in snaps],
                    [o for o, _ in snaps],
                    self._steps,
                ),
            )
        return demo_episode and s.map_id == self._spec["target_map"]

    def _end_episode(self, demo_success: bool) -> None:
        """Report the outcome of the episode to the archive and the curriculum."""
        kind = self._spec["kind"]
        if kind == "demo":
            self._emit("outcomes", ("demo", self._spec["demo"], demo_success))
        elif kind == "archive":
            self._emit(
                "outcomes", ("archive", self._spec["key"], sum(self._tracker.totals.values()))
            )
        if self._recording:
            path = (
                Path(self.config.record_dir)
                / f"{self.config.record_tag}_ep{self._episode_index}.npz"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, frames=np.stack(self._recording))
            self._last_recording = str(path)
            self._recording = None

    def replay_meta(self) -> dict:
        cfg = self.config
        return {
            "frames_per_action": cfg.frames_per_action,
            "press_frames": cfg.press_frames,
            "game": self.adapter.name,
            "goal": self.adapter.milestones[self._goal].id,
            "milestone_names": np.array([m.name for m in self.adapter.milestones]),
        }

    def _on_goal(self) -> None:
        """The goal: save the replay (here, or in the training process in shared mode)."""
        self._goal_report = {"node": self._node, "actions": bytes(self._actions)}
        if self.config.shared_exploration:
            return  # the training process has the lineage: it writes the replay
        cfg = self.config
        path = Path(cfg.replay_dir or ".") / f"goal_{cfg.record_tag}_ep{self._episode_index}.npz"
        segments = self.hub.chain(self._node) if self.hub is not None else []
        self._last_replay = write_replay(
            path, self._start_state, [*segments, bytes(self._actions)], **self.replay_meta()
        )

    def _save_state(self) -> bytes:
        return self.emulator.save_state()

    def _update_milestones(self, s: ProgressSignals) -> None:
        for i, milestone in enumerate(self.adapter.milestones):
            if self._milestone_step[i] < 0 and milestone.reached(s):
                self._milestone_step[i] = self._steps

    def _grab_frame(self, downscale: int | None = None) -> np.ndarray:
        rgb = self.emulator.screen()
        if rgb.shape[:2] != (SCREEN_HEIGHT, SCREEN_WIDTH):
            rgb = _resize(rgb, SCREEN_HEIGHT, SCREEN_WIDTH)
        rgb = rgb.astype(np.uint16)
        gray = (77 * rgb[..., 0] + 150 * rgb[..., 1] + 29 * rgb[..., 2]) >> 8
        k = self.config.downscale if downscale is None else downscale
        return gray[::k, ::k].astype(np.uint8)

    def _observation(self) -> np.ndarray:
        return np.stack(self._frames, axis=0)

    def _info(self, s: ProgressSignals, parts: dict[str, float]) -> dict[str, Any]:
        stats = self.hub.stats() if self.hub is not None else {}
        kind = self._spec["kind"]
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
            "from_archive": kind == "archive",
            "archive_cells": stats.get("archive_cells", 0),
            "from_demo": kind == "demo",
            "demos_active": stats.get("demos_active", 0),
            "demos_completed": stats.get("demos_completed", 0),
            "demo_progress": self._spec.get("progress", 0.0),
            "recording": self._last_recording,
            "goal_reached": self._goal_reached,
            "replay": self._last_replay,
        }


def _resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbour resize (fast, and enough to keep the screen readable)."""
    rows = np.arange(height) * image.shape[0] // height
    cols = np.arange(width) * image.shape[1] // width
    return image[rows][:, cols]


def _check_rom(rom_path: str, adapter: GameAdapter) -> None:
    path = Path(rom_path)
    if not path.exists():
        raise FileNotFoundError(
            f"ROM not found: {rom_path}. Put your legally obtained ROM in roms/ (see docs/SETUP.md)."
        )
    sha1 = hashlib.sha1(path.read_bytes()).hexdigest()
    if sha1 != adapter.rom_sha1 and sha1 not in getattr(adapter, "rom_sha1_alternatives", ()):
        warnings.warn(
            f"ROM SHA-1 {sha1} does not match the expected {adapter.rom_sha1} for "
            f"'{adapter.name}'. Memory addresses may be wrong for this version.",
            stacklevel=2,
        )
