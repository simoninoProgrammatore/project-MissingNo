"""The Game Boy Advance path: mGBA through libretro, behind the same environment.

Skipped when the mGBA core is not installed (scripts/get_mgba_core.py). No game is
needed: the test writes a tiny GBA program (one instruction, an infinite loop).
"""

import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest
from missingno_core import ProgressSignals
from missingno_envs import ACTIONS, EnvConfig, PokemonEnv
from missingno_envs.libretro import LibretroEmulator, find_core
from missingno_games import Milestone

try:
    find_core()
except FileNotFoundError:
    pytest.skip("mGBA libretro core not installed", allow_module_level=True)


@pytest.fixture
def rom(tmp_path):
    data = bytearray(0x200)
    data[0:4] = struct.pack("<I", 0xEAFFFFFE)  # ARM "b ." : loop forever
    data[0xA0:0xAC] = b"MISSINGNO   "
    path = tmp_path / "loop.gba"
    path.write_bytes(bytes(data))
    return str(path)


class TinyGba:
    name = "tiny"
    platform = "gba"
    milestones = (Milestone("G", "Goal", lambda s: s.badges >= 1),)

    def __init__(self, rom):
        self.rom_sha1 = hashlib.sha1(Path(rom).read_bytes()).hexdigest()

    def read(self, memory):
        return ProgressSignals(map_id=memory[0x02000000], x=memory[0x03000000], y=0)


def test_emulator_memory_state_and_screen(rom):
    emu = LibretroEmulator(rom)
    assert emu.memory.regions  # the core mapped its memory
    assert emu.memory[0x080000A0:0x080000A9] == b"MISSINGNO"
    assert emu.screen().shape == (160, 240, 3)
    state = emu.save_state()
    emu.tick(10)
    emu.load_state(state)
    emu.stop()


def test_environment_runs_a_gba_game_with_the_game_boy_observation(rom):
    env = PokemonEnv(EnvConfig(rom_path=rom, max_steps=20, record_every=1), TinyGba(rom))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3, 72, 80)  # the GBA screen is resized to the Game Boy's
    for _ in range(5):
        obs, *_ = env.step(int(np.random.randint(len(ACTIONS))))
    assert obs.shape == (3, 72, 80)
    env.close()


def test_two_gba_emulators_in_one_process_are_independent(rom):
    a, b = LibretroEmulator(rom), LibretroEmulator(rom)
    a.tick(5)
    b.tick(5)
    assert a.save_state() and b.save_state()
    a.stop()
    b.stop()
