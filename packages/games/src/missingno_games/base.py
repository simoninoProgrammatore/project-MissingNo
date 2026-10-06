"""The contract every game adapter follows."""

from typing import Protocol, SupportsIndex

from missingno_core import ProgressSignals


class Memory(Protocol):
    """Anything indexable by address returning a byte, like `pyboy.memory`."""

    def __getitem__(self, address: SupportsIndex) -> int: ...


class GameAdapter(Protocol):
    """Translates one game's memory into game-agnostic progress signals.

    Adapters are the ONLY place in the project that knows a specific game.
    They are used for rewards and metrics during training, never as input to
    the agent.
    """

    name: str
    #: SHA-1 of the expected ROM, to warn if a different version is used.
    rom_sha1: str

    def read(self, memory: Memory) -> ProgressSignals: ...
