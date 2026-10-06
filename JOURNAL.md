# Project MissingNo — Journal

*One entry per session: what I tried, what happened, what I learned, what's next.*

---

## YYYY-MM-DD — Phase 0: setup

**Done:**
- Created the uv workspace with `core`, `envs` and `skills/battle`.
- Local Pokémon Showdown server running.
- Rule-based player (always picks the move with the highest expected damage) against a random player: 30/30 in gen1randombattle.

**Learned:**
- 

**Problems:**
- 

**Next:**
- Study the existing literature and projects.
- Improve the rule-based player with real game knowledge (status moves, switching, Gen 1 quirks).
- Look at the human battle datasets released by the Metamon project and understand their format.

---

## YYYY-MM-DD — New direction and the environment

**Decided:**
- End-to-end agent: sees only the screen, presses buttons.
- Tabula rasa: no human games, no walkthrough knowledge ("the manual yes, the guide no").
- Generic rewards computed by per-game adapters, never shown to the agent.
- Showdown battle exercise moved to `archive/`.

**Done:**
- `RedAdapter`: reads map, position, badges, party levels, battle flag.
- `PokemonEnv` (Gymnasium): screen observations, 7 buttons, generic rewards.
- Scripts: `make_start_state.py`, `watch.py`, `benchmark_env.py`.
- Tests with PyBoy's test ROM: ~550 steps/s per core.

**Next:**
- Run with my Red ROM: create the start state, watch the random agent, check rewards make sense.
- Benchmark on my PC.
- Define the milestones for Pallet Town -> Viridian City and the cost gate.
