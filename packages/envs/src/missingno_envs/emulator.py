"""Emulators behind one small interface, so the environment does not care which console runs.

- Game Boy and Game Boy Color (Red, Blue, Yellow, Crystal): PyBoy.
- Game Boy Advance (FireRed, ...): mGBA, loaded as a libretro core (see libretro.py).

Every emulator offers the same few operations: press a button for some frames,
advance frames, read the screen and the memory, save and load states.
"""

from __future__ import annotations

import io
from typing import Protocol

import numpy as np


class Emulator(Protocol):
    #: Indexable by address, returning a byte (the game's memory, for adapters only).
    memory: object

    def press(self, button: str, frames: int) -> None:
        """Hold `button` ("a", "b", "start", "select", "up", ...) for `frames` frames."""

    def tick(self, frames: int = 1, render: bool = True) -> bool:
        """Advance `frames` frames. False if the window was closed."""

    def screen(self) -> np.ndarray:
        """The last rendered frame, (height, width, 3) uint8 RGB."""

    def save_state(self) -> bytes: ...

    def load_state(self, state: bytes) -> None: ...

    def set_speed(self, speed: int) -> None:
        """1 = real time, 0 = as fast as possible (only matters with a window)."""

    def stop(self) -> None: ...


class PyBoyEmulator:
    """PyBoy, for Game Boy and Game Boy Color games."""

    def __init__(self, rom_path: str, window: bool = False, cgb: bool | None = None) -> None:
        from pyboy import PyBoy

        self.pyboy = PyBoy(
            rom_path, window="SDL2" if window else "null", sound_emulated=False, cgb=cgb
        )
        self.memory = self.pyboy.memory

    def press(self, button: str, frames: int) -> None:
        self.pyboy.button(button, frames)

    def tick(self, frames: int = 1, render: bool = True) -> bool:
        return self.pyboy.tick(frames, render)

    def screen(self) -> np.ndarray:
        return self.pyboy.screen.ndarray[:, :, :3]

    def save_state(self) -> bytes:
        buffer = io.BytesIO()
        self.pyboy.save_state(buffer)
        return buffer.getvalue()

    def load_state(self, state: bytes) -> None:
        self.pyboy.load_state(io.BytesIO(state))

    def set_speed(self, speed: int) -> None:
        self.pyboy.set_emulation_speed(speed)

    def stop(self) -> None:
        self.pyboy.stop(save=False)

    @staticmethod
    def version() -> str:
        import pyboy

        return f"PyBoy {getattr(pyboy, '__version__', '?')}"


def make_emulator(rom_path: str, adapter, window: bool = False, interactive: bool = False):
    """The right emulator for the adapter's console (`adapter.platform`, default "gb")."""
    platform = getattr(adapter, "platform", "gb")
    if platform == "gba":
        from missingno_envs.libretro import LibretroEmulator

        return LibretroEmulator(rom_path, window=window, interactive=interactive)
    return PyBoyEmulator(rom_path, window=window, cgb=getattr(adapter, "cgb", None))
