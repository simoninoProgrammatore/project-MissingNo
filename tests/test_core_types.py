import pytest
from missingno_core import GameMode, GameState, Goal, GoalKind, PokemonInfo
from pydantic import ValidationError


def test_game_state_valid():
    state = GameState(
        game="red",
        mode=GameMode.OVERWORLD,
        map_id="PALLET_TOWN",
        position=(5, 6),
        party=[PokemonInfo(species="BULBASAUR", level=5, hp=19, max_hp=21)],
    )
    assert state.party[0].hp_fraction == pytest.approx(19 / 21)
    assert not state.party[0].fainted


def test_impossible_values_are_rejected():
    # Valori impossibili letti dalla RAM devono fallire subito, non un'ora dopo.
    with pytest.raises(ValidationError):
        PokemonInfo(species="MISSINGNO", level=0, hp=10, max_hp=10)
    with pytest.raises(ValidationError):
        GameState(game="red", mode=GameMode.BATTLE, map_id="X", position=(0, 0), badges=99)


def test_goal():
    goal = Goal(
        kind=GoalKind.REACH, target={"map_id": "PEWTER_CITY"}, description="Vai a Plumbeopoli"
    )
    assert goal.kind == "reach"
