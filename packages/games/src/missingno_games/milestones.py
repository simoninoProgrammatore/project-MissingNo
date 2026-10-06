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
