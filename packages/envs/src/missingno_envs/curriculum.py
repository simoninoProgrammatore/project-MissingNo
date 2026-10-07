"""Backward curriculum from the agent's own successes.

Inspired by "Learning Montezuma's Revenge from a single demonstration"
(Salimans & Chen, 2018), with one key difference: the demonstration is not
human. It is an episode of the agent itself that, for the first time ever,
reached a map no episode had reached before (e.g. Route 1 after the lab).

How it works:
1. During every episode, the emulator state is saved every `interval` steps.
2. When an episode reaches a map never reached before, its saved states become
   a *demo* whose goal is "reach that map".
3. Some episodes start from a state of the demo close to the end (one snapshot
   before the success). If the agent reaches the goal often enough from there,
   the start moves one snapshot earlier, and so on, until the agent can do the
   whole path from the beginning of the demo. Then the demo is done.

A rare lucky success becomes a skill the agent can repeat. All states come from
the agent's own play: no human data, no walkthrough.
"""

from __future__ import annotations

import zlib
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Demo:
    snapshots: list[bytes]  # zlib-compressed states, one every `interval` steps
    target_map: int
    interval: int
    start: int = -1  # index of the current starting snapshot (set on creation)
    attempts: int = 0  # attempts at the current starting point
    recent: deque = field(default_factory=lambda: deque(maxlen=8))

    def __post_init__(self) -> None:
        if self.start < 0:
            self.start = len(self.snapshots) - 1

    @property
    def progress(self) -> float:
        """0 = only the last snapshot mastered, 1 = the whole path."""
        n = len(self.snapshots)
        return 1.0 - self.start / max(1, n - 1)

    def budget(self) -> int:
        """Steps allowed from the current start: twice the original path, plus a margin."""
        remaining = (len(self.snapshots) - self.start) * self.interval
        return 2 * remaining + 256


@dataclass
class BackwardCurriculum:
    interval: int = 64  # steps between snapshots
    max_demos: int = 8
    max_snapshots: int = 400  # per episode (400 x 64 = 25,600 steps)
    success_needed: int = 4  # successes out of the last 8 attempts to move back
    max_attempts: int = 200  # give up on a starting point after this many attempts
    demos: list[Demo] = field(default_factory=list)
    completed: int = 0
    abandoned: int = 0

    def __len__(self) -> int:
        return len(self.demos)

    def add_demo(self, snapshots: list[bytes], target_map: int) -> None:
        """A first-ever success: keep its path as a demo (the newest demos are kept)."""
        if len(snapshots) < 1:
            return
        self.demos.append(Demo(list(snapshots), target_map, self.interval))
        if len(self.demos) > self.max_demos:
            self.demos.pop(0)

    def sample(self, rng: np.random.Generator) -> tuple[Demo, bytes]:
        demo = self.demos[rng.integers(len(self.demos))]
        return demo, zlib.decompress(demo.snapshots[demo.start])

    def record(self, demo: Demo, success: bool) -> None:
        """Outcome of an episode started from `demo`: move the start back when it is mastered."""
        if demo not in self.demos:
            return  # dropped meanwhile
        demo.attempts += 1
        demo.recent.append(success)
        if sum(demo.recent) >= self.success_needed:
            if demo.start == 0:
                self.demos.remove(demo)  # the whole path is learned
                self.completed += 1
                return
            demo.start -= 1
            demo.attempts = 0
            demo.recent.clear()
        elif demo.attempts >= self.max_attempts:
            self.demos.remove(demo)  # too hard for now
            self.abandoned += 1
