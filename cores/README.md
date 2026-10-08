# Emulator cores

Game Boy Advance games (FireRed, ...) run on mGBA, loaded as a *libretro core*: one
file, `mgba_libretro.dll` (Windows), `.so` (Linux) or `.dylib` (macOS). Get it with:

```bash
uv run python scripts/get_mgba_core.py                # the core for this computer
uv run python scripts/get_mgba_core.py --also-linux   # plus the Linux one, for Kaggle
```

Cores are not committed. Save states are tied to the core version: create start
states and train with the same core file (for Kaggle, put `mgba_libretro.so` in
your private dataset next to the ROMs).

mGBA is free software (Mozilla Public License 2.0): https://mgba.io
