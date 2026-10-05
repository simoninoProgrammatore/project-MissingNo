# MissingNo

Agente **orchestrato** che gioca a Pokémon: un orchestratore riconosce la situazione di gioco e attiva l'abilità adatta (lotta, navigazione, menu), con modelli piccoli addestrati da zero.

![Architettura](docs/missingno-architettura.svg)

Piano completo: [`docs/piano.md`](docs/piano.md) · Diario: [`JOURNAL.md`](JOURNAL.md)

## Stato

**Fase 0** — ambiente pronto, baseline euristica di lotta funzionante.

## Struttura

```
missingno/
├── packages/
│   ├── core/            # contratti condivisi: GameState, Goal, Action, Skill
│   ├── envs/            # ambienti: Showdown (ora), PyBoy (fase 3)
│   └── skills/
│       └── battle/      # abilità di lotta: euristica, poi modelli addestrati
├── scripts/             # esperimenti e script eseguibili
├── tests/
├── benchmark/
│   └── checkpoints/     # i "punti difficili" (savestate + descrizioni)
├── configs/             # configurazioni degli esperimenti
├── traces/              # log delle decisioni (non versionati)
└── docs/
```

Le altre cartelle (`games/`, `skills/navigation/`, `skills/menus/`, `orchestrator/`, `planner/`, `training/`) si aggiungono quando servono.

**Regola architetturale:** solo gli adattatori in `games/` conoscono la RAM di un gioco specifico. Tutto il resto dipende solo dai tipi in `core`.

## Setup

Requisiti: [uv](https://docs.astral.sh/uv/), Node.js (per Showdown), Git.

```bash
# 1. Dipendenze Python (uv installa anche Python 3.12 se manca)
uv sync

# 2. Hook di lint/format automatici a ogni commit
uv run pre-commit install

# 3. Server Pokémon Showdown locale (in un terminale a parte)
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
```

## Primi comandi

```bash
uv run pytest                                        # test
uv run ruff check . && uv run ruff format .          # lint e formattazione
uv run python scripts/play_baseline.py --n-battles 100   # euristico vs casuale
```

Risultato di riferimento (gen1randombattle, 30 lotte): **euristico 30/30 contro casuale**.

## Aggiungere un pacchetto

```bash
uv init --lib packages/skills/navigation --name missingno-navigation
```
poi aggiungerlo a `members` e `[tool.uv.sources]` nel `pyproject.toml` di root.

## Note

- Le ROM non vanno mai committate (sono escluse dal `.gitignore`): usare una copia ottenuta da una propria cartuccia.
- I nomi Pokémon sono marchi registrati; questo è un progetto personale e di ricerca open.
