"""Baseline euristica: sceglie la mossa con il danno atteso più alto.

È il riferimento che il primo modello addestrato dovrà battere (fase 1).
"""

from poke_env.battle import AbstractBattle
from poke_env.player import Player

STAB_BONUS = 1.5


def move_score(
    base_power: float, type_multiplier: float, stab: bool, accuracy: float = 1.0
) -> float:
    """Stima grezza del danno atteso di una mossa.

    Funzione pura, separata dal Player, così è facile da testare e migliorare.
    """
    score = base_power * type_multiplier * accuracy
    if stab:
        score *= STAB_BONUS
    return score


class HeuristicPlayer(Player):
    """Usa la mossa più efficace; se non ha mosse utili, cambia o va a caso."""

    def choose_move(self, battle: AbstractBattle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if battle.available_moves and active is not None and opponent is not None:

            def score(move) -> float:
                accuracy = move.accuracy if isinstance(move.accuracy, float) else 1.0
                return move_score(
                    base_power=move.base_power,
                    type_multiplier=opponent.damage_multiplier(move),
                    stab=move.type in active.types,
                    accuracy=accuracy,
                )

            best = max(battle.available_moves, key=score)
            if score(best) > 0:
                return self.create_order(best)

        return self.choose_random_move(battle)
