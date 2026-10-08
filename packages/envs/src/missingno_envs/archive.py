"""Archive of states reached by the agent itself (a simple form of Go-Explore).

Problem it solves: when every episode restarts from the bedroom, the agent
practices thousands of times the part it already masters and only rarely the
part where it gets stuck (e.g. leaving Oak's lab after the first battle).

How it works:
1. While playing, every time the agent reaches a new *cell* (a coarse region of
   a map, combined with a coarse summary of its progress), the emulator state is
   saved in the archive.
2. At the start of some episodes, instead of the bedroom, a state is picked from
   the archive, favouring cells that have been chosen and visited less often
   (the frontier of what the agent knows) and cells where the agent is
   currently *learning* the most.

Learning progress (inspired by Prioritized Level Replay and learning-progress
curricula): every cell keeps a fast and a slow moving average of the returns of
the episodes started from it. When the two differ, results from that cell are
changing, i.e. the agent is learning something there. Cells that are too easy
(always the same good result) or too hard (always nothing) have no progress and
are chosen less.

Every archived state comes from the agent's own play: nobody tells it where to
go. Episodes started from the archive are flagged, and are excluded from the
milestone curves, which always measure "from the bedroom".
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field

import numpy as np
from missingno_core import ProgressSignals

Cell = tuple[int, ...]


def cell_key(s: ProgressSignals, grid: int = 4) -> Cell:
    """Where the agent is (coarse) and how far it has got (coarse).

    The progress part makes "in the lab, before the starter" and "in the lab,
    after the first battle" different cells, so both get their own saved state.
    """
    seen = s.pokedex_seen if s.pokedex_seen < 10 else 10 + s.pokedex_seen // 10
    return (
        s.map_id,
        s.x // grid,
        s.y // grid,
        len(s.party_levels),
        s.badges,
        len(s.items) // 2,
        seen,
    )


@dataclass
class _Entry:
    state: bytes  # zlib-compressed emulator state
    chosen: int = 0  # times an episode started from it
    visits: int = 0  # times the agent stepped in this cell
    fast: float = 0.0  # moving averages of the returns of episodes started here
    slow: float = 0.0
    outcomes: int = 0
    node: int = 0  # lineage node: how the agent got here from the start (exploration.py)

    @property
    def progress(self) -> float:
        return abs(self.fast - self.slow) if self.outcomes >= 2 else 0.0


@dataclass
class StateArchive:
    max_cells: int = 1000
    grid: int = 4
    progress_weight: float = 1.0  # 0 = choose by counts only
    cells: dict[Cell, _Entry] = field(default_factory=dict)
    visits_unarchived: dict[Cell, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.cells)

    def observe(self, s: ProgressSignals, save_state) -> bool:
        """Count a visit; if the cell is new, store the current state.

        `save_state` is called only for new cells (saving costs a few ms).
        Returns True if a new cell was added.
        """
        key = cell_key(s, self.grid)
        entry = self.cells.get(key)
        if entry is not None:
            entry.visits += 1
            return False
        if len(self.cells) >= self.max_cells:
            self.visits_unarchived[key] = self.visits_unarchived.get(key, 0) + 1
            return False
        self.cells[key] = _Entry(state=zlib.compress(save_state(), 1), visits=1)
        return True

    def add(self, key: Cell, compressed_state: bytes, node: int = 0, visits: int = 1) -> bool:
        """Store an already-compressed state for a new cell (shared archive). False if known/full."""
        if key in self.cells:
            self.cells[key].visits += visits
            return False
        if len(self.cells) >= self.max_cells:
            self.visits_unarchived[key] = self.visits_unarchived.get(key, 0) + visits
            return False
        self.cells[key] = _Entry(state=compressed_state, visits=visits, node=node)
        return True

    def add_visits(self, key: Cell, n: int) -> None:
        entry = self.cells.get(key)
        if entry is not None:
            entry.visits += n

    def record_outcome(self, key: Cell, episode_return: float) -> None:
        """Tell the archive how an episode started from `key` went."""
        entry = self.cells.get(key)
        if entry is None:
            return
        if entry.outcomes == 0:
            entry.fast = entry.slow = episode_return
        else:
            entry.fast += 0.5 * (episode_return - entry.fast)
            entry.slow += 0.1 * (episode_return - entry.slow)
        entry.outcomes += 1

    def sample(self, rng: np.random.Generator) -> tuple[Cell, bytes]:
        """Pick a cell: frontier (chosen and visited less) plus learning progress."""
        key, entry = self.sample_entry(rng)
        return key, zlib.decompress(entry.state)

    def sample_entry(self, rng: np.random.Generator, count: bool = True) -> tuple[Cell, _Entry]:
        """Like `sample`, but returns the entry (compressed state, lineage node).

        count=False: do not count it as chosen yet (the shared hub hands out starting
        points in advance, and counts them when an episode reports back).
        """
        keys = list(self.cells)
        entries = [self.cells[k] for k in keys]
        weights = np.array(
            [1 / math.sqrt(1 + e.chosen) + 1 / math.sqrt(1 + e.visits) for e in entries]
        )
        progress = np.array([e.progress for e in entries])
        if self.progress_weight > 0 and progress.max() > 0:
            weights = weights + self.progress_weight * progress / progress.max()
        key = keys[rng.choice(len(keys), p=weights / weights.sum())]
        entry = self.cells[key]
        if count:
            entry.chosen += 1
        return key, entry
