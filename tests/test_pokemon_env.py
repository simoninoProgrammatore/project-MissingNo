"""Environment tests using the small test ROM bundled with PyBoy.

They check the mechanics (spaces, steps, observations, rewards), not Pokémon
itself: the real game is tested by running scripts/watch.py with your ROM.
"""

import hashlib
import os
import warnings

import numpy as np
import pyboy
import pytest
from gymnasium.utils.env_checker import check_env
from missingno_core import ProgressSignals
from missingno_envs import EnvConfig, PokemonEnv

TEST_ROM = os.path.join(os.path.dirname(pyboy.__file__), "default_rom.gb")


class ScriptedAdapter:
    """Returns a predefined sequence of progress signals, one per read."""

    name = "test"

    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.i = 0
        with open(TEST_ROM, "rb") as f:
            self.rom_sha1 = hashlib.sha1(f.read()).hexdigest()

    def read(self, memory):
        s = self.sequence[min(self.i, len(self.sequence) - 1)]
        self.i += 1
        return s


def make_env(sequence=None, **kw):
    sequence = sequence or [ProgressSignals(map_id=0, x=0, y=0)]
    return PokemonEnv(EnvConfig(rom_path=TEST_ROM, **kw), ScriptedAdapter(sequence))


def test_gymnasium_api_is_respected():
    env = make_env(max_steps=50)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(env, skip_render_check=True)
    env.close()


def test_observation_shape_and_type():
    env = make_env(frame_stack=4, downscale=2)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4, 72, 80)
    assert obs.dtype == np.uint8
    obs, *_ = env.step(0)
    assert env.observation_space.contains(obs)
    env.close()


def test_generic_rewards():
    seq = [
        ProgressSignals(map_id=0, x=0, y=0, party_levels=(5,)),  # reset
        ProgressSignals(map_id=0, x=1, y=0, party_levels=(5,)),  # new tile
        ProgressSignals(map_id=0, x=0, y=0, party_levels=(5,)),  # already visited
        ProgressSignals(map_id=1, x=0, y=0, party_levels=(5,)),  # new map (+ new tile)
        ProgressSignals(map_id=1, x=0, y=0, party_levels=(7,)),  # +2 levels
        ProgressSignals(map_id=1, x=0, y=0, party_levels=(5,)),  # level drop: no penalty
        ProgressSignals(map_id=1, x=0, y=0, party_levels=(7,)),  # back to 7: no reward
        ProgressSignals(map_id=1, x=0, y=0, badges=1, party_levels=(7,)),  # badge
    ]
    env = make_env(seq)
    r = env.rewards
    env.reset(seed=0)
    rewards = [env.step(0)[1] for _ in range(len(seq) - 1)]
    assert rewards == pytest.approx(
        [r.new_tile, 0, r.new_tile + r.new_map, 2 * r.level, 0, 0, r.badge]
    )
    env.close()


def test_truncation_at_max_steps():
    env = make_env(max_steps=3)
    env.reset(seed=0)
    flags = [env.step(0)[3] for _ in range(3)]
    assert flags == [False, False, True]
    env.close()


def test_missing_rom_has_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="roms/"):
        PokemonEnv(EnvConfig(rom_path=str(tmp_path / "nope.gb")), ScriptedAdapter([]))
