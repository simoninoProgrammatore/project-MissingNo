# Save states

Emulator save states used as starting points. They contain game data, so they are ignored by Git.

Create the starting state for Pokémon Red with:

```bash
uv run python scripts/make_start_state.py --rom roms/pokemon_red.gb --out states/red_start.state
```
