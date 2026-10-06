"""Fase 0: l'agente euristico sfida l'agente casuale sul server Showdown locale.

Uso:
    uv run python scripts/play_baseline.py --n-battles 100
"""

import argparse
import asyncio

from missingno_battle import HeuristicPlayer
from missingno_envs import DEFAULT_FORMAT, local_server_config
from poke_env.player import RandomPlayer


async def main(n_battles: int, battle_format: str) -> None:
    server = local_server_config()
    heuristic = HeuristicPlayer(battle_format=battle_format, server_configuration=server)
    random_player = RandomPlayer(battle_format=battle_format, server_configuration=server)

    await heuristic.battle_against(random_player, n_battles=n_battles)

    win_rate = heuristic.n_won_battles / n_battles
    print(f"Formato: {battle_format}")
    print(f"Euristico vs casuale: {heuristic.n_won_battles}/{n_battles} vittorie ({win_rate:.0%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-battles", type=int, default=100)
    parser.add_argument("--format", default=DEFAULT_FORMAT)
    args = parser.parse_args()
    asyncio.run(main(args.n_battles, args.format))
