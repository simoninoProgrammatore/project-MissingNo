"""Compare two players over many battles and report the win rate with a confidence interval.

This is the main measuring tool of the project: every new heuristic or model
gets compared against the previous best with this script.

Usage:
    uv run python scripts/compare_players.py heuristic random --n-battles 200
    uv run python scripts/compare_players.py heuristic simple --n-battles 500

Available players: see PLAYERS below.
"""

import argparse
import asyncio
import math

from missingno_battle import HeuristicPlayer
from missingno_envs import DEFAULT_FORMAT, local_server_config
from poke_env.player import (
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)

# Name -> player class. Add your new heuristics and models here.
PLAYERS: dict[str, type[Player]] = {
    "random": RandomPlayer,
    "maxpower": MaxBasePowerPlayer,  # poke-env: always the highest base power move
    "simple": SimpleHeuristicsPlayer,  # poke-env: a stronger reference heuristic
    "heuristic": HeuristicPlayer,  # ours
}


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a win rate (Wilson score interval).

    With few battles the win rate is noisy: the interval tells you whether a
    difference between two versions is real or just luck.
    """
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (center - half, center + half)


async def main(name_a: str, name_b: str, n_battles: int, battle_format: str) -> None:
    server = local_server_config()
    player_a = PLAYERS[name_a](battle_format=battle_format, server_configuration=server)
    player_b = PLAYERS[name_b](battle_format=battle_format, server_configuration=server)

    await player_a.battle_against(player_b, n_battles=n_battles)

    wins = player_a.n_won_battles
    low, high = wilson_interval(wins, n_battles)
    print(f"Format: {battle_format}")
    print(f"{name_a} vs {name_b}: {wins}/{n_battles} wins ({wins / n_battles:.1%})")
    print(f"95% confidence interval: {low:.1%} - {high:.1%}")
    if low > 0.5:
        print(f"-> {name_a} is better than {name_b}")
    elif high < 0.5:
        print(f"-> {name_a} is worse than {name_b}")
    else:
        print("-> No clear difference: play more battles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("player_a", choices=PLAYERS)
    parser.add_argument("player_b", choices=PLAYERS)
    parser.add_argument("--n-battles", type=int, default=200)
    parser.add_argument("--format", default=DEFAULT_FORMAT)
    args = parser.parse_args()
    asyncio.run(main(args.player_a, args.player_b, args.n_battles, args.format))
