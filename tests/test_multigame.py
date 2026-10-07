"""One run, several games: how parallel games are shared out, files, held-out guard."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from ppo import Config, game_files, game_list, game_of, train  # noqa: E402


def test_single_game_by_default():
    cfg = Config()
    assert game_list(cfg) == ["red"]
    assert game_files(cfg, "red") == ("roms/pokemon_red.gb", "states/red_start.state")
    assert game_files(Config(rom="x.gb", state="y.state"), "red") == ("x.gb", "y.state")


def test_games_are_shared_out_in_turn():
    cfg = Config(games="red, blue,yellow", num_envs=6)
    assert game_list(cfg) == ["red", "blue", "yellow"]
    assert [game_of(cfg, i) for i in range(6)] == ["red", "blue", "yellow"] * 2
    # Several games: always the default paths, one per title.
    assert game_files(cfg, "yellow") == ("roms/pokemon_yellow.gb", "states/yellow_start.state")


def test_unknown_game_fails_clearly():
    with pytest.raises(SystemExit, match="Unknown game"):
        game_list(Config(games="red,emerald"))


def test_training_on_the_held_out_game_is_refused():
    with pytest.raises(SystemExit, match="held out"):
        train(Config(games="red,crystal", run_name="should_not_exist"))
    assert not (Path("runs") / "should_not_exist").exists()
