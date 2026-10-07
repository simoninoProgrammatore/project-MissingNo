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


# --- v2.1: exploration that wears out with use -----------------------------------

C21 = RewardConfig.preset("v2.1")


def episodes(tracker, n, states):
    """Play the same short episode n times; return the parts of the last one."""
    for _ in range(n):
        tracker.reset(states[0])
        parts = [tracker.step(s, i)[1] for i, s in enumerate(states[1:], start=1)]
    return parts


def test_tile_reward_decays_across_episodes():
    tracker = RewardTracker(C21)
    walk = [S(), S(x=1)]
    first = episodes(tracker, 1, walk)[0]["new_tile"]
    hundredth = episodes(tracker, 99, walk)[0]["new_tile"]
    assert first == pytest.approx(C21.rare_tile)
    assert hundredth == pytest.approx(C21.rare_tile / 10)


def test_rare_tiles_are_worth_more_than_common_ones():
    tracker = RewardTracker(C21)
    episodes(tracker, 50, [S(), S(x=1)])  # x=1 is visited in every episode
    parts = episodes(tracker, 1, [S(), S(x=1), S(x=2)])  # x=2 is brand new
    assert parts[1]["new_tile"] > 5 * parts[0]["new_tile"]


def test_tile_still_pays_once_per_episode():
    tracker = RewardTracker(C21)
    tracker.reset(S())
    tracker.step(S(x=1), 1)
    _, parts = tracker.step(S(x=1), 2)
    assert "new_tile" not in parts


def test_passages_are_directed_and_pay_once_per_episode():
    tracker = RewardTracker(C21)
    lab, town = 40, 0
    states = [S(map_id=town), S(map_id=lab), S(map_id=town), S(map_id=lab), S(map_id=town)]
    tracker.reset(states[0])
    for i, s in enumerate(states[1:], start=1):
        tracker.step(s, i)
    # In and out: two passages, each paid once, however many times the door is used.
    assert tracker.totals["passage"] == pytest.approx(2 * C21.passage)


def test_passage_reward_decays_across_episodes():
    tracker = RewardTracker(C21)
    door = [S(map_id=0), S(map_id=40)]
    assert episodes(tracker, 1, door)[0]["passage"] == pytest.approx(C21.passage)
    assert episodes(tracker, 3, door)[0]["passage"] == pytest.approx(C21.passage / 2)


def test_v2_is_unchanged_by_v21():
    tracker = RewardTracker(C)
    parts = episodes(tracker, 10, [S(map_id=0), S(map_id=1, map_area=400, x=1)])[0]
    assert parts["new_tile"] == pytest.approx(C.new_tile)
    assert "passage" not in parts


# --- v2.2: experience instead of levels --------------------------------------------

C22 = RewardConfig.preset("v2.2")


def test_every_battle_won_is_rewarded_even_without_a_level_up():
    # Starter at level 5 (~135 exp), one wild battle won: +40 exp, same level.
    _, parts = run(
        [S(party_levels=(5,), party_exp=(135,)), S(party_levels=(5,), party_exp=(175,))], C22
    )
    assert parts[0]["experience"] > 0
    assert "team" not in parts[0]


def test_an_early_battle_is_worth_about_ten_new_tiles():
    _, parts = run([S(party_exp=(135,)), S(party_exp=(175,))], C22)
    assert 5 * C22.new_tile < parts[0]["experience"] < 20 * C22.rare_tile


def test_first_pokemon_is_rewarded_like_in_v2():
    _, v22 = run([S(), S(party_levels=(5,), party_exp=(135,))], C22)
    _, v2 = run([S(), S(party_levels=(5,))], C)
    assert v22[0]["experience"] == pytest.approx(v2[0]["team"], rel=0.25)


def test_grinding_pays_less_and_less():
    _, early = run([S(party_exp=(135,)), S(party_exp=(175,))], C22)
    _, late = run([S(party_exp=(30_000,)), S(party_exp=(30_800,))], C22)  # a strong team
    assert early[0]["experience"] > 5 * late[0]["experience"]


def test_deposit_and_withdraw_does_not_pay_experience():
    team = (3000, 2000)
    states = [
        S(party_exp=team),
        S(party_exp=(3000,)),
        S(party_exp=team),
        S(party_exp=(3000,)),
        S(party_exp=team),
    ]
    tracker, _ = run(states, C22)
    assert tracker.totals["experience"] == 0


def test_v22_keeps_rarity_and_passages():
    assert "passage" in RewardTracker(C22).components
    tracker = RewardTracker(C22)
    tracker.reset(S())
    assert tracker.step(S(x=1), 1)[1]["new_tile"] == pytest.approx(C22.rare_tile)


# --- v2.2 loop breaker ------------------------------------------------------------


def test_flee_loop_in_a_trainer_battle_ends_the_episode():
    """A battle with no progress for too long ends the episode (the 'flee loop')."""
    config = RewardConfig.preset("v2.2").with_weights(battle_stagnation_steps=50)
    tracker = RewardTracker(config)
    tracker.reset(S())
    tracker.step(S(x=1), 1)  # progress: a new tile
    for t in range(2, 10):
        tracker.step(S(x=1), t)  # walking in place, no battle: only the long limit applies
    assert not tracker.stagnant(9)
    for t in range(10, 59):
        tracker.step(S(x=1, in_battle=True), t)  # battle starts at step 10
    assert not tracker.stagnant(58)
    tracker.step(S(x=1, in_battle=True), 60)
    assert tracker.stagnant(60)


def test_progress_during_a_battle_restarts_the_count():
    config = RewardConfig.preset("v2.2").with_weights(battle_stagnation_steps=50)
    tracker = RewardTracker(config)
    tracker.reset(S(party_levels=(5,), party_exp=(135,)))
    for t in range(1, 40):
        tracker.step(S(party_levels=(5,), party_exp=(135,), in_battle=True), t)
    tracker.step(S(party_levels=(5,), party_exp=(150,), in_battle=True), 40)  # a win
    for t in range(41, 80):
        tracker.step(S(party_levels=(5,), party_exp=(150,), in_battle=True), t)
    assert not tracker.stagnant(79)  # 39 steps since the win
    assert tracker.stagnant(90)


def test_leaving_the_battle_stops_the_battle_count():
    config = RewardConfig.preset("v2.2").with_weights(
        battle_stagnation_steps=50, stagnation_steps=0
    )
    tracker = RewardTracker(config)
    tracker.reset(S())
    for t in range(1, 40):
        tracker.step(S(in_battle=True), t)
    for t in range(40, 200):
        tracker.step(S(), t)
    assert not tracker.stagnant(199)


def test_v22_preset_breaks_loops_and_earlier_versions_do_not_change():
    v22 = RewardConfig.preset("v2.2")
    assert (v22.stagnation_steps, v22.battle_stagnation_steps) == (5000, 1000)
    for version in ("v2", "v2.1"):
        assert RewardConfig.preset(version).battle_stagnation_steps == 0
        assert RewardConfig.preset(version).stagnation_steps == 2000
