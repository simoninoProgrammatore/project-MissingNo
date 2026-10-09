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

---

## 2026-10-07 — Route 2, Viridian Forest, and a Game Boy Advance

**Results (`badge_v22_s1`, no memory, old code without the loop breaker):**
- **Route 2 at ~7.83M steps:** the agent brought Oak's Parcel back and went north, alone, from the bedroom. This is the step the reference project skipped. It did not "know" it had the parcel: it got there by exploring, helped by the archive (which keeps "with the parcel" and "without it" as different states).
- **Viridian Forest** shortly after: once a wall falls, the next stretch comes fast. The cost of the game is in its walls, not in its length.

**Decided:**
- Long-term goal: **beat Red with reinforcement learning only, without any help** (no scripted actions, no story flags, no human data, no language models). Written definition of "no help" to be added to `docs/research.md`.
- A plan for the walls: a toolbox of generic tools (shared archive with action sequences, screen-based archive cells, screen novelty, new-move reward, two-level memory, full resolution, learned skills) and a protocol: diagnose, hypothesis, one generic tool, same-seed comparison. Never code written for one specific wall.
- Compute: ISCRA-C gives up to 100,000 core-hours on Leonardo GP (an estimated tens of billions of steps, to be measured): enough to make the method, not the compute, the bottleneck. Application with the professor as PI.
- A PhD student at ETH (Michele Viscione) said winning Gold never seen is almost impossible, and suggested MoE or an orchestrator. Agreed on the first point for zero-shot; the fair version is "never trained on, but learning while playing it", measured as speed. Of his suggestions, a **learned hierarchy of skills** fits the project.

**Done:**
- **Game Boy Advance support:** an emulator layer (PyBoy for Game Boy, mGBA for GBA through a small libretro frontend in Python, no compilation needed), `scripts/get_mgba_core.py`, and a **FireRed adapter** (pointers to the moving save blocks, encrypted experience, pockets, badges), with Red's milestones on FireRed's maps. The GBA screen is resized to the Game Boy's, so the same model plays both.
- First question for FireRed: same story as Red, different look. If the Red model does better on FireRed than on Crystal, it learned the story; if badly on both, it learned the pixels.

---

## 2026-10-08 — Shared exploration, continuous replays, the v3 model

**Diagnosis (`badge_v22_s1` at 15.8M steps, stuck in Viridian Forest for ~5M steps):**
- From the bedroom it reaches Route 2 in 2–4% of episodes and the forest in ~2%, around step 18,000 of a 30,000-step episode: little time left for the forest, Pewter and the gym.
- It wins more and more battles (`reward/experience` from ~2 to ~6.5): the flee bug is fixed.
- The curriculum was clogged: 8 demos all the time, none ever completed, starts barely moving back. Three flaws: one snapshot back at a time (a 17,000-step demo has ~265 starting points), one curriculum per parallel game (the Route 2 demo lived in 1 game of 4, competing with "leave the house"), and everything lost at every Kaggle session.
- Resumed for the night with the new code (loop breaker) and 100,000-step episodes.

**Done (block A, infrastructure):**
- **Shared exploration:** one archive and one curriculum per title in the training process, fed by all parallel games and **saved next to the checkpoints** (`exploration_<game>.pkl`), so they survive a resume.
- **Lineage and continuous replays:** every saved state remembers the buttons that led to it from the bedroom. A goal reached in an episode that started from an archived state still produces one continuous replay of the whole game, checked byte for byte by a test.
- **Faster curriculum:** a tenth of the path back at once when the agent never fails, newer demos chosen more often, snapshots thinned out in long episodes so demos cover the whole path.
- **Milestones for the whole of Red** (M1–M40, to the Hall of Fame); the goal of the current phase stays M12, `--goal M40` aims at the whole game.
- **"Without any help" written down** in `docs/research.md` (what is allowed, what is not, how a win is shown).

**Done (block B, the v3 model):** `--downscale 1` (full screen, readable text), `--network impala`, `--gamma 0.999` with `--norm-rewards`, together with `--memory gru`. About 6.6M parameters: GPU only.

**Next:** resume `badge_v22_s1` with block A; start `red_v3_s1` next to it; compare. Then the ISCRA-C draft.

---

## 2026-10-08 — A lost night, Kaggle's queue, and a compute strategy

**What happened:** the resume ran overnight and failed at once: the notebook looked for `red_start.state`, while the dataset had `pokemon_red.state`. A night lost to a file name.

**Done:**
- The notebook accepts either name for the start state and copies it under the name the code expects.
- The resume cell takes the `latest.pt` with the most training steps when the inputs hold several (a dataset and an old output together), and prints all of them.

**Learned:**
- "Queued" on Kaggle means waiting for a free machine. GPU machines are scarce, especially in the evening; an open editor with the GPU on also takes a slot. CPU-only sessions usually start at once and use no weekly quota.
- CPU-only does not mean more CPUs: both kinds of session have 4 cores. Without a GPU the same 4 cores also do the learning, so it is somewhat slower. Fine for the small network; the v3 model needs the GPU.
- Plan: the small network (`badge_v22_s1`) runs on CPU every day; the GPU quota goes to `red_v3_s1`.
- Honest assessment: the small network can plausibly reach the first badge; the whole game needs memory and a larger network, whatever the number of steps.

**Decided:** brute force is not the plan. The whole of Red is estimated at 10–50B steps, years on Kaggle and up to a whole ISCRA-C allocation. Written down in `docs/research.md` (§6.2, "Making every step count"): three levers that raise what each step is worth, each with its test. They are: splitting the game through the agent's own states (Go-Explore, already built), a world model (DreamerV3, a research branch), and more steps per second. Memory is a prerequisite, not a lever. This is also the skeleton of the ISCRA-C application.

---

## 2026-10-09 — Watching the rewards, and the CPU speed

**Done:**
- `watch.py` shows a live panel in the terminal: the reward of each component (total, how many times it paid, share of the return), the last rewards, the milestones and the next one, team, badges and Pokédex.
- The Kaggle notebook runs on CPU-only sessions (it no longer needs `nvidia-smi`).
- On a CPU, the network now learns with every core (`--learn-threads`, default: all cores on CPU). The games wait while the network learns, so their cores were idle.

**Learned:**
- The 464 steps/s session of `badge_v22_s1` ran with a GPU, not on CPU as I thought. On a real CPU-only session the small network does ~150 steps/s and spends ~60% of the time learning on one thread. With all cores while learning, the estimate is ~1.5x faster (measured +15% on a 2-core machine).
- In `watch.py` every tile is worth its full value, because the panel starts with no visit counts. In training, tiles visited millions of times are worth almost nothing, so the panel overstates `new_tile`.
- Kaggle shows the logs of a running version in bursts: hours without lines do not mean the run is stuck.
- In the watched game the agent reached Viridian City in ~2,500 steps, then spent 5,000 steps battling in the grass instead of entering the Mart: battles pay a little, often; the Mart pays once.
