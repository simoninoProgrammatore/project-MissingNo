"""Backward curriculum from the agent's own successes.

Inspired by "Learning Montezuma's Revenge from a single demonstration"
(Salimans & Chen, 2018), with one key difference: the demonstration is not
human. It is an episode of the agent itself that, for the first time ever,
reached a map no episode had reached before (e.g. Route 1 after the lab).

How it works:
1. During every episode, the emulator state is saved every few steps.
2. When an episode reaches a map never reached before, its saved states become
   a *demo* whose goal is "reach that map".
3. Some episodes start from a state of the demo close to the end (one snapshot
   before the success). If the agent reaches the goal often enough from there,
   the start moves earlier, and so on, until the agent can do the whole path
   from the beginning of the demo. Then the demo is done.

Two refinements, added after the first long run (a demo of ~17,000 steps had
~265 starting points and never moved back noticeably in 15M steps):
- Adaptive steps back: when the agent succeeds every time from a start, the
  start jumps back by a tenth of the path instead of one snapshot.
- Priority to the frontier: newer demos (the most recently discovered maps) are
  chosen more often than old, easy ones.

A rare lucky success becomes a skill the agent can repeat. All states come from
the agent's own play: no human data, no walkthrough.
"""

from __future__ import annotations

import zlib
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass(eq=False)
class Demo:
    snapshots: list[bytes]  # zlib-compressed states along the path
    target_map: int
    interval: int
    start: int = -1  # index of the current starting snapshot (set on creation)
    attempts: int = 0  # attempts at the current starting point
    recent: deque = field(default_factory=lambda: deque(maxlen=8))
    # Step of the episode at which each snapshot was taken, and at which the goal
    # was reached. Default: one snapshot every `interval` steps.
    offsets: list[int] | None = None
    goal_offset: int = -1
    # Lineage, for continuous replays (see exploration.py): the node the demo's
    # episode started from, and the buttons it pressed until the success.
    node: int = 0
    actions: bytes = b""  # zlib-compressed
    id: int = 0
    nodes: dict[int, int] = field(default_factory=dict)  # snapshot index -> lineage node

    def __post_init__(self) -> None:
        if self.offsets is None:
            self.offsets = [i * self.interval for i in range(len(self.snapshots))]
        if self.goal_offset < 0:
            self.goal_offset = len(self.snapshots) * self.interval
        if self.start < 0:
            self.start = len(self.snapshots) - 1

    @property
    def progress(self) -> float:
        """0 = only the last snapshot mastered, 1 = the whole path."""
        n = len(self.snapshots)
        return 1.0 - self.start / max(1, n - 1)

    def budget(self) -> int:
        """Steps allowed from the current start: twice the original path, plus a margin."""
        return 2 * (self.goal_offset - self.offsets[self.start]) + 256


@dataclass
class BackwardCurriculum:
    interval: int = 64  # steps between snapshots
    max_demos: int = 8
    max_snapshots: int = 400  # per episode; beyond that, snapshots are thinned out
    success_needed: int = 4  # successes out of the last 8 attempts to move back
    max_attempts: int = 200  # give up on a starting point after this many attempts
    jump: float = 0.1  # with no failure at all, move back by this fraction of the path
    demos: list[Demo] = field(default_factory=list)
    completed: int = 0
    abandoned: int = 0
    next_id: int = 1

    def __len__(self) -> int:
        return len(self.demos)

    def add_demo(
        self,
        snapshots: list[bytes],
        target_map: int,
        offsets: list[int] | None = None,
        goal_offset: int = -1,
        node: int = 0,
        actions: bytes = b"",
    ) -> Demo | None:
        """A first-ever success: keep its path as a demo (the newest demos are kept)."""
        if len(snapshots) < 1:
            return None
        demo = Demo(
            list(snapshots),
            target_map,
            self.interval,
            offsets=list(offsets) if offsets is not None else None,
            goal_offset=goal_offset,
            node=node,
            actions=actions,
            id=self.next_id,
        )
        self.next_id += 1
        self.demos.append(demo)
        if len(self.demos) > self.max_demos:
            self.demos.pop(0)
        return demo

    def get(self, demo_id: int) -> Demo | None:
        return next((d for d in self.demos if d.id == demo_id), None)

    def sample(self, rng: np.random.Generator) -> tuple[Demo, bytes]:
        demo = self.sample_demo(rng)
        return demo, zlib.decompress(demo.snapshots[demo.start])

    def sample_demo(self, rng: np.random.Generator) -> Demo:
        """Newer demos (the frontier) are chosen more often: weight 1, 2, ..., n."""
        weights = np.arange(1, len(self.demos) + 1, dtype=float)
        return self.demos[rng.choice(len(self.demos), p=weights / weights.sum())]

    def record(self, demo: Demo, success: bool) -> None:
        """Outcome of an episode started from `demo`: move the start back when it is mastered."""
        if all(d is not demo for d in self.demos):
            return  # dropped meanwhile
        demo.attempts += 1
        demo.recent.append(success)
        if sum(demo.recent) >= self.success_needed:
            if demo.start == 0:
                self.demos.remove(demo)  # the whole path is learned
                self.completed += 1
                return
            flawless = all(demo.recent)
            step = max(1, int(self.jump * len(demo.snapshots))) if flawless else 1
            demo.start = max(0, demo.start - step)
            demo.attempts = 0
            demo.recent.clear()
        elif demo.attempts >= self.max_attempts:
            self.demos.remove(demo)  # too hard for now
            self.abandoned += 1
