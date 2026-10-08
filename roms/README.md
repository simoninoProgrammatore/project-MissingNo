# ROMs

Put your **legally obtained** game ROMs here (dumped from your own cartridges). They are ignored by Git and must never be committed or shared.

Expected file names:

| Game | File | SHA-1 |
|---|---|---|
| Pokémon Red (international) | `pokemon_red.gb` | `ea9bcae617fdf159b045185467ae58b2e4a48b9a` |
| Pokémon Blue (international) | `pokemon_blue.gb` | `d7037c83e1ae5b39bde3c30787637ba1d4c48ce2` |
| Pokémon Yellow (international) | `pokemon_yellow.gbc` (or `.gb`) | `cc7d03262ebfaf2f06772c1a480c7d9d5f4a38e1` |
| Pokémon FireRed 1.0 (English) | `pokemon_firered.gba` | `41cb23d8dccc8ebd7c649cd8fbb58eeace6e2fdc` |
| Pokémon FireRed 1.1 (English) | `pokemon_firered.gba` | `dd5945db9b930750cb39d00c84da8571feebf417` |
| Pokémon Crystal 1.0 (international) — held-out test game | `pokemon_crystal.gbc` | `f4cd194bdee0d04ca4eac29e09b8e4e9d818c133` |
| Pokémon Crystal 1.1 (international) — held-out test game | `pokemon_crystal.gbc` | `f2f52230b536214ef7c9924f483392993e226cfb` |

The environment checks the SHA-1 and warns if it does not match: memory addresses differ between versions.

Other languages (e.g. the Italian versions) have different memory addresses and are not supported yet.
