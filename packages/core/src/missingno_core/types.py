"""Tipi condivisi tra adattatori, ambienti, abilità e orchestratore."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class GameMode(StrEnum):
    """Situazione di gioco: decide quale abilità attiva l'orchestratore."""

    OVERWORLD = "overworld"
    BATTLE = "battle"
    MENU = "menu"
    DIALOGUE = "dialogue"


class PokemonInfo(BaseModel):
    """Informazioni su un Pokémon della squadra, comuni a tutti i giochi."""

    species: str
    level: int = Field(ge=1, le=100)
    hp: int = Field(ge=0)
    max_hp: int = Field(ge=1)
    types: list[str] = Field(default_factory=list)
    moves: list[str] = Field(default_factory=list, max_length=4)

    @property
    def hp_fraction(self) -> float:
        return self.hp / self.max_hp

    @property
    def fainted(self) -> bool:
        return self.hp == 0


class GameState(BaseModel):
    """Stato di gioco comune, prodotto da un adattatore a partire dalla RAM."""

    game: str  # es. "red", "crystal", "emerald"
    mode: GameMode
    map_id: str
    position: tuple[int, int]
    party: list[PokemonInfo] = Field(default_factory=list, max_length=6)
    badges: int = Field(default=0, ge=0, le=16)
    money: int = Field(default=0, ge=0)
    step: int = Field(default=0, ge=0)


class GoalKind(StrEnum):
    """Tipi di obiettivo che il pianificatore può assegnare."""

    REACH = "reach"  # raggiungere un luogo
    WIN_BATTLE = "win_battle"
    CATCH = "catch"
    HEAL = "heal"
    TRAIN = "train"


class Goal(BaseModel):
    """Obiettivo dato a un'abilità (goal-conditioned)."""

    kind: GoalKind
    target: dict[str, Any] = Field(default_factory=dict)
    description: str = ""  # versione leggibile, utile per log e pianificatore


class Button(StrEnum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    A = "a"
    B = "b"
    START = "start"
    SELECT = "select"


class ButtonPress(BaseModel):
    """Azione nell'emulatore: premere un tasto."""

    type: Literal["button"] = "button"
    button: Button


class BattleChoice(BaseModel):
    """Azione in lotta, indipendente da come il gioco la realizza."""

    type: Literal["battle"] = "battle"
    kind: Literal["move", "switch"]
    index: int = Field(ge=0)


Action = ButtonPress | BattleChoice


class SkillStatus(StrEnum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@runtime_checkable
class Skill(Protocol):
    """Contratto di ogni abilità. L'orchestratore conosce solo questo."""

    name: str

    def act(self, state: GameState, goal: Goal) -> Action: ...

    def status(self, state: GameState, goal: Goal) -> SkillStatus: ...
