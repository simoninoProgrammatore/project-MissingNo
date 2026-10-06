"""Game-agnostic progress signals.

Every game adapter turns its own memory layout into these same fields. They are
used ONLY to compute rewards and metrics during training: the agent itself never
sees them, it only sees the screen.

This is a plain dataclass (not Pydantic) on purpose: it is created at every
environment step, thousands of times per second, so it must be cheap.
"""

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class ProgressSignals:
    map_id: int
    x: int
    y: int
    badges: int = 0
    party_levels: tuple[int, ...] = field(default_factory=tuple)
    in_battle: bool = False

    @property
    def cell(self) -> tuple[int, int, int]:
        """The exact tile the player is standing on, across the whole game."""
        return (self.map_id, self.x, self.y)

    @property
    def total_level(self) -> int:
        return sum(self.party_levels)
