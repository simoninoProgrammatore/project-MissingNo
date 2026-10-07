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
from missingno_envs import EnvConfig, PokemonEnv, RewardConfig
from missingno_games import Milestone

TEST_ROM = os.path.join(os.path.dirname(pyboy.__file__), "default_rom.gb")


class ScriptedAdapter:
    """Returns a predefined sequence of progress signals, one per read."""

    name = "test"
    milestones = (
        Milestone("T1", "Reach map 1", lambda s: s.map_id == 1),
        Milestone("T2", "Earn a badge", lambda s: s.badges > 0),
    )

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


def test_v1_rewards_are_unchanged():
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
    env = make_env(seq, rewards=RewardConfig.preset("v1"))
    r = env.rewards
    env.reset(seed=0)
    rewards = [env.step(0)[1] for _ in range(len(seq) - 1)]
    assert rewards == pytest.approx(
        [r.new_tile, 0, r.new_tile + r.new_map_flat, 2 * r.level, 0, 0, r.badge]
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


def test_milestones_record_first_step_reached():
    seq = [
        ProgressSignals(map_id=0, x=0, y=0),  # reset
        ProgressSignals(map_id=0, x=1, y=0),  # step 1
        ProgressSignals(map_id=1, x=0, y=0),  # step 2: T1
        ProgressSignals(map_id=0, x=0, y=0),  # step 3: left map 1, T1 stays reached
        ProgressSignals(map_id=0, x=0, y=0, badges=1),  # step 4: T2
    ]
    env = make_env(seq)
    env.reset(seed=0)
    for _ in range(4):
        *_, info = env.step(0)
    assert info["milestone_step"].tolist() == [2, 4]
    # New episode: milestones are reset and re-checked on the starting state
    # (the scripted adapter now returns a state that already has a badge).
    _, info = env.reset(seed=0)
    assert info["milestone_step"].tolist() == [-1, 0]
    env.close()


def test_stagnation_truncates_the_episode():
    still = ProgressSignals(map_id=0, x=0, y=0)
    env = make_env([still], rewards=RewardConfig.preset("v2").with_weights(stagnation_steps=3))
    env.reset(seed=0)
    flags = [env.step(0)[3] for _ in range(3)]
    assert flags == [False, False, True]
    env.close()


def test_info_reports_reward_totals():
    seq = [ProgressSignals(map_id=0, x=0, y=0), ProgressSignals(map_id=0, x=1, y=0)]
    env = make_env(seq)
    env.reset(seed=0)
    *_, info = env.step(0)
    assert info["reward_totals"]["new_tile"] == pytest.approx(env.rewards.new_tile)
    assert set(info["reward_totals"]) == set(env._tracker.components)
    env.close()
