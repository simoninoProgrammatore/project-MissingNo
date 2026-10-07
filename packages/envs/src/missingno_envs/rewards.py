"""Generic rewards, identical for every game (design: docs/rewards.md).

The reward only ever says "something new happened" or "you made progress",
never "do this" or "go there". Every component pays once (first visit, first
item, new species) or only above the best value reached so far, so nothing can
be farmed by repeating it.

Versions:
- v1: the Phase 1 baseline (new tiles, new maps, badges, linear levels).
       Kept identical so earlier results stay reproducible.
- v2: maps weighted by size, a concave "team strength" instead of linear levels,
       new items (key items worth more), Pokédex owned/seen, and stagnation
       truncation (episodes with no progress for too long end early).
- v2.1: v2 plus exploration that wears out with use. A new tile is still paid
       once per episode, but its value decays with how many past episodes have
       already visited it (1/sqrt(n)), so places seen in every episode become
       nearly worthless and rarely visited places become attractive. Passages
       between maps (doors, stairs, exits) are rewarded the first time they are
       crossed in each direction, with the same decay.
       The counts are kept for the whole training and built only from the
       agent's own experience: no map, no knowledge of the game.
- v2.2: v2.1 with experience instead of levels. Levels go up only every few
       battles, so a single battle paid nothing until the level changed, and
       the agent learned to always flee (walking on new tiles paid more and
       sooner). Experience points grow after every battle won, so every win is
       rewarded right away. Still concave and max-so-far: grinding forever, or
       depositing and re-withdrawing Pokémon, does not pay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from missingno_core import ProgressSignals

COMPONENTS: dict[str, tuple[str, ...]] = {
    "v1": ("new_tile", "new_map", "badge", "level"),
    "v2": ("new_tile", "new_map", "badge", "team", "new_item", "dex_owned", "dex_seen"),
    "v2.1": (
        "new_tile",
        "new_map",
        "passage",
        "badge",
        "team",
        "new_item",
        "dex_owned",
        "dex_seen",
    ),
    "v2.2": (
        "new_tile",
        "new_map",
        "passage",
        "badge",
        "experience",
        "new_item",
        "dex_owned",
        "dex_seen",
    ),
}

# Versions that use the lifelong rarity of tiles and the passages (v2.1 and later).
_RARITY_VERSIONS = ("v2.1", "v2.2")


@dataclass(frozen=True)
class RewardConfig:
    """Reward version and weights. Weights are relative to one badge."""

    version: str = "v2"
    new_tile: float = 0.02
    badge: float = 10.0
    # v1 only
    new_map_flat: float = 0.5
    level: float = 0.2  # per party level above the best total seen
    # v2 only
    new_map: float = 0.2  # a full-size map; smaller maps pay proportionally less
    full_map_area: int = 400  # steps: a typical town or route counts as "full size"
    team: float = 1.0  # per unit of the team-strength potential
    team_size: int = 4  # only the strongest N Pokémon count
    new_item: float = 0.3
    new_key_item: float = 1.0
    dex_owned: float = 0.3
    dex_owned_scale: float = 20.0  # owned species reward decays as 1/sqrt(1 + owned/scale)
    dex_seen: float = 0.05
    stagnation_steps: int = 2000  # 0 = never truncate for lack of progress
    # v2.1 only: rewards that decay with how many past episodes already had them
    rare_tile: float = 0.1  # a never-visited tile; after n episodes it is worth rare_tile/sqrt(n)
    passage: float = 0.2  # a never-crossed passage between maps, in one direction
    # v2.2 only: experience. Potential = log(1 + total_exp / exp_scale) over the
    # `team_size` Pokémon with the most experience; paid when it beats its best.
    # Calibrated so that an early battle won is worth about ten new tiles, and the
    # first Pokémon about as much as the v2 "team" reward.
    experience: float = 2.0
    exp_scale: float = 100.0

    @classmethod
    def preset(cls, version: str) -> RewardConfig:
        if version == "v1":
            # Exactly the Phase 1 baseline.
            return cls(version="v1", badge=5.0, stagnation_steps=0)
        if version in ("v2", "v2.1", "v2.2"):
            return cls(version=version)
        raise ValueError(f"Unknown reward version {version!r}: choose from {sorted(COMPONENTS)}")

    def with_weights(self, **weights: float) -> RewardConfig:
        return replace(self, **weights)


def team_strength(levels: tuple[int, ...], size: int) -> float:
    """Sum of log(level) of the strongest `size` Pokémon.

    Concave: the 10th level of a Pokémon is worth more than its 50th, so
    grinding one Pokémon stops paying. Top-N: a 5th or 6th Pokémon only counts
    if it becomes stronger than the current core.
    """
    best = sorted(levels, reverse=True)[:size]
    return sum(math.log(level) for level in best if level > 0)


def experience_potential(party_exp: tuple[int, ...], size: int, scale: float) -> float:
    """log(1 + total experience of the `size` most experienced Pokémon / scale).

    Grows after every battle won, with diminishing returns: the same battle is
    worth a lot for a young team and little for a strong one. A weak new catch
    only counts if it becomes one of the most experienced.
    """
    best = sorted(party_exp, reverse=True)[:size]
    return math.log1p(sum(best) / scale)


class RewardTracker:
    """Keeps what has been achieved in the current episode and turns each new
    state into a reward. One tracker per environment, reset at every episode."""

    def __init__(self, config: RewardConfig) -> None:
        self.config = config
        self.components = COMPONENTS[config.version]
        self.totals: dict[str, float] = {}
        # v2.1, lifelong: in how many episodes each tile / passage has been reached.
        # Never reset: this is what makes exploration "wear out" with use.
        self.tile_episodes: dict[tuple[int, int, int], int] = {}
        self.passage_episodes: dict[tuple[int, int], int] = {}

    def reset(self, s: ProgressSignals) -> None:
        c = self.config
        self.visited_tiles = {s.cell}
        self.visited_maps = {s.map_id}
        self.crossed_passages: set[tuple[int, int]] = set()
        self.previous_map = s.map_id
        self._count(self.tile_episodes, s.cell)
        self.best_badges = s.badges
        self.best_total_level = s.total_level
        self.best_team = team_strength(s.party_levels, c.team_size)
        self.best_exp = experience_potential(s.party_exp, c.team_size, c.exp_scale)
        self.items_seen = set(s.items)  # items already held at the start do not pay
        self.best_owned = s.pokedex_owned
        self.best_seen = s.pokedex_seen
        self.last_progress_step = 0
        self.totals = dict.fromkeys(self.components, 0.0)

    def step(self, s: ProgressSignals, step: int) -> tuple[float, dict[str, float]]:
        c = self.config
        parts: dict[str, float] = {}

        if s.cell not in self.visited_tiles:
            self.visited_tiles.add(s.cell)
            if c.version in _RARITY_VERSIONS:
                n = self._count(self.tile_episodes, s.cell)
                parts["new_tile"] = c.rare_tile / math.sqrt(n)
            else:
                parts["new_tile"] = c.new_tile

        if c.version in _RARITY_VERSIONS and s.map_id != self.previous_map:
            passage = (self.previous_map, s.map_id)  # directed: in and out are different
            if passage not in self.crossed_passages:
                self.crossed_passages.add(passage)
                n = self._count(self.passage_episodes, passage)
                parts["passage"] = c.passage / math.sqrt(n)
        self.previous_map = s.map_id

        if s.map_id not in self.visited_maps:
            self.visited_maps.add(s.map_id)
            if c.version == "v1":
                parts["new_map"] = c.new_map_flat
            else:
                # Houses and shops are tiny maps: a flat bonus would turn the agent
                # into a tourist walking in and out of every door.
                parts["new_map"] = c.new_map * min(1.0, s.map_area / c.full_map_area)

        if s.badges > self.best_badges:
            parts["badge"] = c.badge * (s.badges - self.best_badges)
            self.best_badges = s.badges

        if c.version == "v1":
            if s.total_level > self.best_total_level:
                parts["level"] = c.level * (s.total_level - self.best_total_level)
                self.best_total_level = s.total_level
        elif c.version == "v2.2":
            exp = experience_potential(s.party_exp, c.team_size, c.exp_scale)
            if exp > self.best_exp + 1e-9:
                parts["experience"] = c.experience * (exp - self.best_exp)
                self.best_exp = exp
        else:
            team = team_strength(s.party_levels, c.team_size)
            if team > self.best_team + 1e-9:
                parts["team"] = c.team * (team - self.best_team)
                self.best_team = team

        if c.version != "v1":
            new_items = s.items - self.items_seen
            if new_items:
                self.items_seen |= new_items
                n_key = len(new_items & s.key_items)
                parts["new_item"] = c.new_key_item * n_key + c.new_item * (len(new_items) - n_key)

            if s.pokedex_owned > self.best_owned:
                parts["dex_owned"] = sum(
                    c.dex_owned / math.sqrt(1 + n / c.dex_owned_scale)
                    for n in range(self.best_owned, s.pokedex_owned)
                )
                self.best_owned = s.pokedex_owned

            if s.pokedex_seen > self.best_seen:
                parts["dex_seen"] = c.dex_seen * (s.pokedex_seen - self.best_seen)
                self.best_seen = s.pokedex_seen

        if parts:
            self.last_progress_step = step
            for name, value in parts.items():
                self.totals[name] += value
        return sum(parts.values()), parts

    @staticmethod
    def _count(counts: dict, key) -> int:
        """Record one more episode reaching `key`; return the updated count."""
        counts[key] = counts.get(key, 0) + 1
        return counts[key]

    def stagnant(self, step: int) -> bool:
        """True if nothing new has happened for too long: the episode should end.

        We end the episode instead of punishing the agent: penalties teach it to
        stand still, or to end episodes on purpose.
        """
        k = self.config.stagnation_steps
        return k > 0 and step - self.last_progress_step >= k
