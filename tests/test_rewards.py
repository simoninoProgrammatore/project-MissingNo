"""Reward v2: each component, and the exploits it must not allow (docs/rewards.md)."""

import math

import pytest
from missingno_core import ProgressSignals
from missingno_envs import RewardConfig, RewardTracker, team_strength

C = RewardConfig.preset("v2")


def S(**kw):
    base = {"map_id": 0, "x": 0, "y": 0}
    base.update(kw)
    return ProgressSignals(**base)


def run(states, config=C):
    """Reset on the first state, step through the others. Returns per-step parts."""
    tracker = RewardTracker(config)
    tracker.reset(states[0])
    return tracker, [tracker.step(s, i)[1] for i, s in enumerate(states[1:], start=1)]


# --- exploration ---------------------------------------------------------------


def test_new_map_is_weighted_by_area():
    _, parts = run(
        [S(), S(map_id=1, map_area=400), S(map_id=2, map_area=64), S(map_id=3, map_area=4000)]
    )
    assert parts[0]["new_map"] == pytest.approx(C.new_map)  # a town or route
    assert parts[1]["new_map"] == pytest.approx(C.new_map * 64 / 400)  # a small house
    assert parts[2]["new_map"] == pytest.approx(C.new_map)  # huge maps are capped


def test_door_ping_pong_pays_only_once():
    states = [S(map_id=0, map_area=400)] + [S(map_id=m % 2, map_area=400) for m in range(1, 10)]
    tracker, _ = run(states)
    assert tracker.totals["new_map"] == pytest.approx(C.new_map)


# --- team -----------------------------------------------------------------------


def test_team_strength_is_concave_and_top4():
    assert team_strength((5,), 4) == pytest.approx(math.log(5))
    # A weak 5th Pokémon does not count:
    assert team_strength((20, 20, 20, 20, 2), 4) == team_strength((20, 20, 20, 20), 4)
    # The 10th level is worth more than the 50th:
    assert math.log(10) - math.log(9) > math.log(50) - math.log(49)


def test_first_pokemon_is_rewarded():
    _, parts = run([S(), S(party_levels=(5,))])
    assert parts[0]["team"] == pytest.approx(C.team * math.log(5))


def test_deposit_and_withdraw_is_not_a_farm():
    team = (12, 10)
    states = [
        S(party_levels=team),
        S(party_levels=(12,)),
        S(party_levels=team),
        S(party_levels=(12,)),
        S(party_levels=team),
    ]
    tracker, _ = run(states)
    assert tracker.totals["team"] == 0


def test_grinding_one_pokemon_has_diminishing_returns():
    _, early = run([S(party_levels=(5,)), S(party_levels=(6,))])
    _, late = run([S(party_levels=(50,)), S(party_levels=(51,))])
    assert early[0]["team"] > 5 * late[0]["team"]


# --- items ----------------------------------------------------------------------


def test_new_items_and_key_items():
    parcel, potion = 0x46, 0x14
    _, parts = run(
        [
            S(),
            S(items=frozenset({potion})),
            S(items=frozenset({potion, parcel}), key_items=frozenset({parcel})),
        ]
    )
    assert parts[0]["new_item"] == pytest.approx(C.new_item)
    assert parts[1]["new_item"] == pytest.approx(C.new_key_item)


def test_selling_and_rebuying_pays_once():
    potion = 0x14
    states = [S(), S(items=frozenset({potion})), S(), S(items=frozenset({potion}))]
    tracker, _ = run(states)
    assert tracker.totals["new_item"] == pytest.approx(C.new_item)


def test_items_held_at_start_do_not_pay():
    _, parts = run([S(items=frozenset({0x14})), S(items=frozenset({0x14}))])
    assert "new_item" not in parts[0]


# --- Pokédex --------------------------------------------------------------------


def test_pokedex_owned_decays_and_seen_is_flat():
    _, parts = run([S(), S(pokedex_owned=1, pokedex_seen=2)])
    assert parts[0]["dex_owned"] == pytest.approx(C.dex_owned)
    assert parts[0]["dex_seen"] == pytest.approx(2 * C.dex_seen)
    _, later = run([S(pokedex_owned=60), S(pokedex_owned=61)])
    assert later[0]["dex_owned"] == pytest.approx(
        C.dex_owned / math.sqrt(1 + 60 / C.dex_owned_scale)
    )


# --- stagnation and bookkeeping -------------------------------------------------


def test_stagnation_counts_steps_without_progress():
    tracker = RewardTracker(C.with_weights(stagnation_steps=100))
    tracker.reset(S())
    tracker.step(S(x=1), 10)  # progress at step 10
    assert not tracker.stagnant(109)
    assert tracker.stagnant(110)
    assert not RewardTracker(C.with_weights(stagnation_steps=0)).stagnant(10**9)


def test_totals_match_the_sum_of_rewards():
    states = [
        S(),
        S(x=1, party_levels=(5,), pokedex_owned=1, pokedex_seen=1),
        S(map_id=1, map_area=200, badges=1),
    ]
    tracker = RewardTracker(C)
    tracker.reset(states[0])
    total = sum(tracker.step(s, i)[0] for i, s in enumerate(states[1:], start=1))
    assert sum(tracker.totals.values()) == pytest.approx(total)


def test_badge_is_the_biggest_single_reward():
    # Hierarchy check from docs/rewards.md: one badge >> a route's worth of tiles and a new map.
    assert C.badge > 200 * C.new_tile + C.new_map


def test_unknown_version_fails_clearly():
    with pytest.raises(ValueError, match="Unknown reward version"):
        RewardConfig.preset("v9")
