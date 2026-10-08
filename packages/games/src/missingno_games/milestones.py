"""Milestones: game-specific checkpoints used ONLY to measure progress.

They never enter the reward. The agent is rewarded generically; milestones tell
us, the experimenters, how far it got (like grading a student, not teaching them).
"""

from collections.abc import Callable
from dataclasses import dataclass

from missingno_core import ProgressSignals


@dataclass(frozen=True)
class Milestone:
    id: str
    name: str
    reached: Callable[[ProgressSignals], bool]


def goal_index(adapter, goal: str | None = None) -> int:
    """Index of the goal milestone: `goal` (an id like "M12"), else the adapter's own
    `goal`, else its last milestone."""
    ids = [m.id for m in adapter.milestones]
    goal = goal or getattr(adapter, "goal", None) or ids[-1]
    if goal not in ids:
        raise ValueError(f"Unknown goal {goal!r} for {adapter.name}: choose from {ids}")
    return ids.index(goal)
