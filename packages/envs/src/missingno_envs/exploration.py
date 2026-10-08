"""Shared exploration: one archive and one curriculum for all the parallel games.

Until now every parallel game kept its own archive and its own curriculum: a
success found by one game (the first time on Route 2, say) helped only that game,
and everything was lost at every new Kaggle session. The ExplorationHub keeps
them in one place (the training process), is saved next to the checkpoints, and
hands out starting points to every game.

It also keeps the *lineage* of every saved state: the buttons the agent pressed
to get there, from the start of the game. A saved state is a node of a tree:

    node = (parent node, buttons pressed after loading the parent's state)

The root (node 0) is the start state (the bedroom). Every episode starts from a
node (the root, an archived state or a demo snapshot); every state it saves
becomes a child of that node. Following the parents back to the root gives a
list of segments; since the emulator is deterministic, replaying them in order
(see scripts/replay.py) reproduces the game exactly, from the bedroom. So even
a goal reached in an episode that started from an archived state comes with a
continuous replay of the whole game, played by the agent.

Each segment starts right after a reset, and every reset advances the emulator
by one frame before the agent acts (PokemonEnv.reset): replays do the same.
"""

from __future__ import annotations

import pickle
import zlib
from pathlib import Path

import numpy as np

from missingno_envs.archive import StateArchive
from missingno_envs.curriculum import BackwardCurriculum, Demo

START = {"kind": "start", "node": 0}


def compress(data: bytes) -> bytes:
    return zlib.compress(bytes(data), 1)


class ExplorationHub:
    """Archive + curriculum + lineage for one game, shared by all its parallel copies."""

    def __init__(
        self,
        archive_prob: float = 0.0,
        curriculum_prob: float = 0.0,
        archive_max_cells: int = 1000,
        archive_progress_weight: float = 1.0,
        curriculum_interval: int = 64,
    ) -> None:
        self.archive_prob = archive_prob
        self.curriculum_prob = curriculum_prob
        self.archive = (
            StateArchive(max_cells=archive_max_cells, progress_weight=archive_progress_weight)
            if archive_prob > 0
            else None
        )
        self.curriculum = (
            BackwardCurriculum(interval=curriculum_interval) if curriculum_prob > 0 else None
        )
        self.nodes: dict[int, tuple[int, bytes]] = {}  # id -> (parent id, compressed buttons)
        self.next_node = 1
        self.maps_ever: set[int] = set()

    # ---------------------------------------------------------------- lineage

    def add_node(self, parent: int, actions: bytes) -> int:
        """A new saved state: `actions` (compressed) pressed after loading `parent`."""
        node = self.next_node
        self.next_node += 1
        self.nodes[node] = (parent, actions)
        return node

    def chain(self, node: int) -> list[bytes]:
        """The button segments from the start state to `node` (root first, raw bytes)."""
        segments = []
        while node:
            parent, actions = self.nodes[node]
            segments.append(zlib.decompress(actions))
            node = parent
        return segments[::-1]

    def _demo_node(self, demo: Demo) -> int:
        """Lineage node of the demo's current starting snapshot (created on first use)."""
        offset = demo.offsets[demo.start]
        if offset == 0:
            return demo.node
        if demo.start not in demo.nodes:
            prefix = zlib.decompress(demo.actions)[:offset]
            demo.nodes[demo.start] = self.add_node(demo.node, compress(prefix))
        return demo.nodes[demo.start]

    # ---------------------------------------------------------------- reports

    def absorb(self, report: dict) -> None:
        """Take in what a game discovered since its last report (see PokemonEnv.drain)."""
        if self.curriculum is not None:
            # Before the maps of this report are counted: a demo is kept only if its map
            # is new for every game (another game may have got there first).
            for target, parent, actions, snapshots, offsets, goal in report.get("demos", ()):
                if target in self.maps_ever:
                    continue
                self.maps_ever.add(target)
                self.curriculum.add_demo(
                    snapshots,
                    target,
                    offsets=offsets,
                    goal_offset=goal,
                    node=parent,
                    actions=actions,
                )
        self.maps_ever |= set(report.get("maps", ()))
        if self.archive is not None:
            for key, state, parent, actions in report.get("cells", ()):
                if key not in self.archive.cells and len(self.archive) < self.archive.max_cells:
                    self.archive.add(key, state, self.add_node(parent, actions))
                else:
                    self.archive.add_visits(key, 1)
            for key, n in report.get("visits", {}).items():
                self.archive.add_visits(key, n)
        for kind, ref, value in report.get("outcomes", ()):
            if kind == "archive" and self.archive is not None:
                self.archive.record_outcome(ref, value)
                if ref in self.archive.cells:
                    self.archive.cells[ref].chosen += 1  # an episode really started there
            elif kind == "demo" and self.curriculum is not None:
                demo = self.curriculum.get(ref)
                if demo is not None:
                    self.curriculum.record(demo, bool(value))

    # ----------------------------------------------------------------- starts

    def next_start(self, rng: np.random.Generator) -> dict:
        """Where the next episode starts: a demo, an archived state, or the start state."""
        r = rng.random()
        cur, arc = self.curriculum, self.archive
        if cur is not None and len(cur) > 0 and r < self.curriculum_prob:
            demo = cur.sample_demo(rng)
            return {
                "kind": "demo",
                "state": demo.snapshots[demo.start],
                "node": self._demo_node(demo),
                "demo": demo.id,
                "target_map": demo.target_map,
                "budget": demo.budget(),
                "progress": demo.progress,
            }
        if arc is not None and len(arc) > 0 and r < self.curriculum_prob + self.archive_prob:
            key, entry = arc.sample_entry(rng, count=False)
            return {"kind": "archive", "state": entry.state, "node": entry.node, "key": key}
        return dict(START)

    def starts(self, n: int, rng: np.random.Generator) -> list[dict]:
        return [self.next_start(rng) for _ in range(n)]

    def stats(self) -> dict[str, float]:
        cur = self.curriculum
        return {
            "archive_cells": len(self.archive) if self.archive is not None else 0,
            "demos_active": len(cur) if cur is not None else 0,
            "demos_completed": cur.completed if cur is not None else 0,
            "demos_abandoned": cur.abandoned if cur is not None else 0,
            "demo_progress_max": max((d.progress for d in cur.demos), default=0.0)
            if cur is not None
            else 0.0,
            "lineage_nodes": len(self.nodes),
        }

    # ------------------------------------------------------------ persistence

    def save(self, path: Path) -> None:
        """Everything, so that a new session continues where the last one stopped."""
        tmp = Path(path).with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            saved = pickle.load(f)
        probs = (self.archive_prob, self.curriculum_prob)
        self.__dict__.update(saved)
        self.archive_prob, self.curriculum_prob = probs  # the new session's settings win
        if self.archive_prob > 0 and self.archive is None:
            self.archive = StateArchive()
        if self.curriculum_prob > 0 and self.curriculum is None:
            self.curriculum = BackwardCurriculum()

    # ---------------------------------------------------------------- replays

    def write_replay(self, path, start_state: bytes, node: int, actions: bytes, **meta) -> str:
        """A continuous replay from the start state: the lineage of `node`, then `actions`."""
        return write_replay(path, start_state, [*self.chain(node), bytes(actions)], **meta)


def write_replay(path, start_state: bytes, segments: list[bytes], **meta) -> str:
    """Save a replay: the start state and every button, segment by segment.

    Before each segment the emulator advances one frame, exactly as PokemonEnv.reset
    does after loading a state. A replay with a single segment is an ordinary episode
    from the start state.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lengths = [len(s) for s in segments]
    np.savez_compressed(
        path,
        actions=np.frombuffer(b"".join(segments), dtype=np.uint8),
        segment_starts=np.cumsum([0, *lengths[:-1]]).astype(np.int64),
        start_state=np.frombuffer(start_state, dtype=np.uint8),
        **meta,
    )
    return str(path)
