# Project MissingNo — Journal

*One entry per session: what I tried, what happened, what I learned, what's next.*

---

## 2026-10-05 — Phase 0: setup

**Done:**
- Created the uv workspace with `core`, `envs` and `skills/battle`.
- Local Pokémon Showdown server running.
- Rule-based player (always picks the move with the highest expected damage) against a random player: 30/30 in gen1randombattle.

**Next:**
- Study the existing literature and projects.
- Improve the rule-based player with real game knowledge (status moves, switching, Gen 1 quirks).
- Look at the human battle datasets released by the Metamon project and understand their format.

---

## 2026-10-05 — Identity and documentation

**Done:**
- All documentation rewritten in English.
- New README for a non-technical reader: the idea, the name, what inspired it, where we are, the road ahead.
- Emblem and banner: black and white, strong glitch, green inside. No Pokémon drawn (they are not ours to use).

---

## 2026-10-07 — New direction and the environment

**Decided:**
- End-to-end agent: sees only the screen, presses buttons.
- Tabula rasa: no human games, no LLM, no walkthrough knowledge. The rule: **"the manual yes, the guide no"**. What the game's manual tells a child (badges, items, battles, the Pokédex) is allowed; what only a walkthrough knows (where to go, which event comes next) is not. No story event flags.
- Generic rewards, identical for every game, computed by per-game adapters from the game's memory. The adapters are never an input to the agent.
- Battles on Showdown were a side exercise: moved to `archive/`.
- The long-term research question: can one agent, trained on some generations, play a generation it has never seen? (`docs/research.md`)

**Done:**
- `RedAdapter`: reads map, position, badges, party levels, battle flag. All addresses checked against the pret/pokered disassembly.
- `PokemonEnv` (Gymnasium): screen observations, 7 buttons, generic rewards.
- Scripts: `make_start_state.py`, `watch.py`, `benchmark_env.py`.
- Tests with PyBoy's test ROM: ~550 steps/s per core.
- `docs/research.md`: questions, hypotheses, phases with gates, metrics. It replaces the old plan.

---

## 2026-10-07 — Phase 1: PPO baseline

**Done:**
- Milestones from the bedroom to Viridian City, used only to measure progress, never as rewards.
- CNN actor-critic (~1M parameters) and a readable, single-file PPO (`training/ppo.py`).
- Resume from checkpoints, a time limit to stop cleanly, and a Kaggle notebook for long runs.

**Learned:**
- My PC's cores scale badly with many emulators: 4 parallel games is the sweet spot. Kaggle (4 cores + T4 GPU) gives ~370–390 steps/s; it is the place for long runs, the PC for development and short tests.
- The agent learned to leave the bedroom, the house, and to enter Oak's lab, then got **stuck in the lab** after the rival battle.

---

## 2026-10-07 — Reward v2

**Decided:**
- New maps weighted by their size: with a flat bonus the agent became a tourist, walking in and out of every small house.
- Levels through a concave "team strength" (log of levels, top 4 Pokémon) instead of linear levels: grinding one Pokémon stops paying.
- New items (key items worth more, as the game itself defines them), and the Pokédex (owned and seen).
- **Stagnation is a truncation, not a penalty:** an episode with no progress for too long just ends. Penalties teach the agent to stand still or to end episodes on purpose.
- Rewarding interactions in general (a new item, a new species) is fine; rewarding a specific NPC or a specific HM is "the guide". Those stay out, or become separate, declared experiments.

**Learned:**
- 2,000 steps of stagnation was too short: it cut good episodes in the lab. Long runs went back to no limit.

---

## 2026-10-07 — Exploration: from the lab to Viridian City

**Done:**
- **v2.1:** exploration that wears out with use. A tile is worth less the more past episodes have already visited it; passages between maps (doors, exits) are rewarded per direction.
- **State archive** (a simple Go-Explore): states reached by the agent itself are saved, and some episodes restart from them, favouring the frontier and the states where the agent is currently learning (inspired by Prioritized Level Replay).
- **Backward curriculum** from the agent's own first successes (after Salimans & Chen, 2018): the first time an episode reaches a new map, its path becomes a demo; episodes start near its end, then further back.
- **Self-imitation learning** (Oh et al., 2018): rare successes are replayed instead of forgotten.
- **Highlight GIFs** saved while training runs, so I can see what the agent does.

**Results:**
- PC: first Route 1 at ~520,000 steps.
- Kaggle: first Viridian City at ~1.34M steps.

**Mistakes:**
- I resumed the wrong checkpoint into the same run name and overwrote `v2_long`. Lesson: a new experiment from an old model always gets a new `--run-name`.

---

## 2026-10-07 — The flee bug and reward v2.2

**What happened:** watching the GIFs and the reward components, we realized that with the old version the agent **always fled** from battles. It was rational from its point of view: a battle takes about a hundred steps and paid nothing until a level went up, which takes two or three battles; in the same hundred steps, walking on new tiles paid right away. Fleeing works until the first trainer, where it is impossible, and Brock.

**Decided: we changed the reward for battles.** v2.2 rewards **experience** instead of levels. Experience grows after every battle won, so every win pays immediately; it is concave (the same battle is worth less for a strong team) and max-so-far (grinding and deposit/withdraw tricks do not pay). Experience exists in every game, so the rule stays generic.

**Also done:**
- Milestones extended to the **first badge** (M1–M12, the last one is the Boulder Badge).
- `--stop-at-goal`: training stops at the first win from the bedroom and saves a **replay** (start state + every button). The emulator is deterministic, so `scripts/replay.py` shows the exact game at normal speed, or exports a video.

**Problems:**
- Kaggle installs a newer PyBoy, and save states are not compatible across PyBoy versions. The notebook now installs the exact versions of `uv.lock`.

---

## 2026-10-07 — First transfer test: Yellow

**Done:**
- Adapters for **Blue** (same memory as Red) and **Yellow** (memory shifted by one byte, run in classic Game Boy mode so it looks like Red).
- `scripts/evaluate.py`: N episodes on any game, milestone by milestone.

**Results:**
- A model trained on Red, playing Yellow without any training on it, gets into its first battle against the rival and **presses RUN forever**: you cannot flee a trainer battle. The old flee habit, applied where it cannot work.

**Learned:**
- The agent is mostly **learning Red**, with a few general skills (doors, dialogues, the battle menu) and one wrong general rule (flee).
- Yellow is weak evidence of generalization: it looks almost identical to Red. Crystal will be the real test.

---

## 2026-10-07 — Kaggle run `badge_v22_s1`

From scratch, reward v2.2, 30,000-step episodes, archive 30%, curriculum 30%, self-imitation on, stop at the first badge. ~380 steps/s.

**Results (first session):**
- Viridian City at ~240,000 steps.
- **Oak's Parcel at ~480,000 steps** (about 53 hours of game time), 7 milestones out of 12.
- Exploration records: 11 maps at ~720,000 steps, 12 maps and 1,536 tiles at ~840,000.

**Context:** Whidden's agent got stuck at this exact point (it could not bring the parcel back to Oak), and he had his games start after the delivery. If ours delivers it alone, from the bedroom, it passes a point the reference project had to skip. For now it got the parcel once: a solid result is when `milestones/M7_rate` stays high.

**Problem:** this session still has the flee loop in trainer battles.

---

## 2026-10-07 — The goal for the professor, and the generalization design

**Decided:**
- **Goal to present:** one model, trained on Red, Blue and Yellow together, that wins the first badge in all three, with a replay of each first win, plus its zero-shot score on Crystal. Third generation and full games come next, with the CINECA proposal.
- **Held out by generation:** train on Generation 1 (and later Generation 3, with FireRed as a bridge), test on **Generation 2 (Crystal)**, never trained on. The code refuses to train on Crystal without an explicit flag.

**Done:**
- **Loop breaker for the flee bug:** with v2.2, an episode ends after 1,000 steps of one battle without progress (a normal battle takes a few dozen steps), and after 5,000 steps without progress anywhere. A loop becomes worth nothing, while winning the battle pays experience. v2.2 is now the default.
- **Several games in one run** (`--games red,blue,yellow`): parallel games shared out among the titles, curves and GIFs per title, a `winner_<game>.pt` at the first win of each game, stop when all are won.
- **Crystal adapter:** banked memory, maps as (group, number), bag pockets, Johto and Kanto badges; milestones up to the Zephyr Badge.
- `watch.py` and `evaluate.py` never end episodes early, so loops stay visible.

**Learned:**
- Whidden's 50,000 hours of game time are about 450M of our steps: two weeks of Kaggle, an estimated day on a CINECA node (to be measured). The free resources are enough for the first badges; full games need a supercomputer.

**Next:**
- Resume `badge_v22_s1` with the new code (the loop breaker activates by itself).
- Create the start states of Blue, Yellow and Crystal; launch the Red + Blue + Yellow run on Kaggle.
- Evaluate on Crystal; write the summary for the professor: replays, milestone curves, steps per milestone, cost.

---

## 2026-10-07 — Stuck after the parcel, and a short-term memory

**What happened:** `badge_v22_s1` kept exploring (13 maps, 1,691 tiles by 1.3M steps) but stayed at 7 milestones out of 12 for over a million steps: it has Oak's Parcel and does not bring it back. An old Kaggle run (the 6-milestone code) was killed: it had nothing left to show and was eating the GPU quota.

**Learned:**
- In Generation 1 the Pokédex is not a bag item: delivering the parcel pays almost nothing directly, the parcel even disappears from the bag. The real rewards (Route 2, Viridian Forest) come later.
- The deeper problem: **the agent cannot know it has the parcel.** It sees only the last 3 frames, and the parcel is in the bag, not on the screen. "Viridian with the parcel" and "Viridian without it" are the same picture. No reward can fix that alone.

**Decided:**
- **Short-term memory** (`--memory gru`): a GRU carried from step to step, wiped at every new episode, that learns what to remember. Same seed, with and without memory, to see if it solves the parcel.
- Later, a **two-level memory**: the GRU plus a long-term store of a few dozen memories, written when the screen changes a lot (computed from pixels) and read by attention. Inspired by complementary learning systems, MERLIN and Neural Episodic Control.
- Kept as options for later: an archive that restarts more often from recent progress, novelty of dialogues read from the screen, places becoming new again after a key item (with its risks: farming, toggling, touring).
- Shorter Kaggle sessions (`TIME_LIMIT_HOURS = 4`) during development, to look at results more often.
