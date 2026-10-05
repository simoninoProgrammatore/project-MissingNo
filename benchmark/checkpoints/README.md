# Hard spots

Save states placed at moments where an AI player (or a human) commonly gets stuck. For each one:

| Field | Example |
|---|---|
| `id` | `red_001_old_man` |
| `game` | red |
| `savestate` | `red_001_old_man.state` (not versioned if it contains game data) |
| `situation` | The old man blocks the path north of Viridian City until Oak's Parcel is delivered |
| `success` | The player reaches Route 2 |
| `max steps` | 5000 |

Each hard spot also becomes an automated test: load the state, let the player run for N steps, check the success criterion.
