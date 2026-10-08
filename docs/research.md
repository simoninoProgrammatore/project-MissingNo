# Project MissingNo — Research design

> **Status:** living document, Phase 1 (prototype). Hypotheses, held-out games and cost gates are written here *before* the corresponding experiments, and are only changed with a dated note.

---

## Summary

We study whether a single agent that **only sees the screen**, **learns from scratch** and receives **only game-agnostic rewards** can learn to play the Pokémon games of the first three generations, and whether what it learns **transfers to games it has never seen**.

Pokémon games are long (tens of thousands of decisions), have sparse rewards and require exploration, memory and many different skills. They also form a rare *family* of related environments with controlled differences: the same world on different consoles, the same engine with different worlds, and entirely new games. This makes them a strong testbed for two open problems in reinforcement learning: **hard exploration over long horizons** and **generalization across environments**.

---

## 1. Motivation

**The problem is open.** Reinforcement learning agents for Pokémon Red are trained on a single game with rewards designed for that game, such as story event flags read from its memory. Agents built on large language models have finished several games, but they rely on knowledge of the games absorbed during pre-training, plus heavy custom tooling. The community lists exploration in long tasks and completing games with open systems among its open challenges.

**There is a question beyond Pokémon.** How much can an agent learn *from interaction alone*, without prior knowledge? Many recent agent successes mix what a model already knows with what it learns while playing. A tabula rasa agent separates the two and provides the missing reference point: *this is what learning from scratch achieves, and at what cost*.

**Pokémon is a rare testbed for generalization.** Generalization in reinforcement learning is usually studied on short, simple games. Pokémon offers long, complex games that are related to each other in measurable ways, so we can ask precisely what transfers and what breaks.

**It is feasible with academic resources.** Measured on a consumer PC, one CPU core simulates about 660 agent steps per second (Section 7). The planned phases fit standard national HPC allocations.

**What would make this work redundant.** If it reduced to "another agent that plays Red", it would add little. Its value rests on three choices: **no game-specific knowledge**, **multiple games**, and **a rigorous measurement protocol**. All three are design requirements, not options.

---

## 2. Research questions

**Main question.** *Can a single end-to-end agent, observing only pixels and trained only with game-agnostic rewards, learn to play multiple Pokémon games and transfer its skills to games it has never seen?*

**RQ1 — Exploration.** Which exploration methods make learning from scratch feasible in long, sparse-reward games, and at what cost per milestone?

**RQ2 — Hand-designed rewards.** How far does an agent with generic rewards get compared with agents trained with game-specific rewards, and how much do those hand-designed rewards actually buy?

**RQ3 — Transfer.** Trained on some games, how much faster does the agent reach the milestones of a held-out game than an agent trained from scratch on it?

**RQ4 — What transfers.** Which capabilities carry over between games (moving, menus, battles, exploration) and which break (story progression, puzzles)?

---

## 3. Hypotheses

Written before the experiments. A rejected hypothesis is a result, not a failure.

- **H1 (exploration).** Plain PPO with generic rewards stalls before the first story gate of Pokémon Red (obtaining the first Pokémon). Structured exploration (Go-Explore and/or intrinsic motivation) reaches it within the Phase 1 budget.
- **H2 (rewards).** On the same segment of Red, generic rewards require more steps than game-specific rewards, but by less than one order of magnitude once structured exploration is used.
- **H3 (transfer).** An agent trained on several games reaches the early milestones of a held-out game with at least 5× fewer steps than an agent trained from scratch on that game.
- **H4 (what transfers).** Low-level skills (movement, menus, battles) transfer much better than story progression, and transfer degrades along the held-out ladder (Section 5.3).

---

## 4. Design principles

1. **Pixels only.** The agent observes only the screen. Memory contents are never part of its input.
2. **Tabula rasa.** No human demonstrations, no pre-trained models, no language models. Everything is learned from interaction.
3. **"The manual yes, the guide no."** Allowed: anything a player would find in the game's instruction booklet (general mechanics). Not allowed: walkthrough knowledge specific to a game (which tree to cut, where to go next).
4. **Generic rewards.** The same reward function for every game, computed from game-agnostic progress signals: new tiles and maps (worth less the more they were seen), passages, badges, experience, new items and key items, the Pokédex. The full design, with edge cases and the planned versions, is in [`rewards.md`](rewards.md).
5. **Adapters for rewards and evaluation only.** One small adapter per game reads its memory to compute rewards and milestones. This privileged information is used *by the training and evaluation code*, never by the agent.
6. **Game-specific milestones are for measuring, not for rewarding.** Milestones (e.g. "reached Viridian City") are defined per game to evaluate progress; they never enter the reward.
7. **Pre-registration.** Hypotheses, held-out games, budgets and cost gates are written in this document before running the corresponding experiments.

### 4.1 What "without any help" means

The long-term goal is to **beat Pokémon Red with reinforcement learning only, without any help**. To make the claim checkable, the rules are fixed here:

| Allowed | Not allowed |
|---|---|
| A start state after the intro (names, Oak's speech): it is setup, not gameplay | Starting after the first Pokémon, after Oak's Parcel, or anywhere later |
| The 7 buttons, one every 24 frames | Scripted actions: automatic use of HMs or items, automatic puzzle solving, automatic Poké Flute, skipped areas |
| Generic rewards from manual-level concepts: new places, passages, badges, experience, items and key items as the game itself defines them, the Pokédex, new dialogues or screens read from the pixels | Story event flags, rewards for specific places, people, trees, boulders or items ("the guide") |
| Exploration helpers built from the agent's own play: archive, curriculum, self-imitation | Human games, walkthroughs, language models, pre-trained models |
| The game as it is | Cheats: infinite money or health, changed encounter rates, edited RAM |
| Milestones read from memory, only to measure | Milestones, or anything read from memory, as input to the agent |

Any exception (for example, an HM-specific reward to test a hypothesis) is a separate, declared experiment, and is never part of the main result. A win is shown by a **continuous replay from the bedroom**: the start state plus every button the agent pressed, which the deterministic emulator reproduces exactly (archived states included, through their lineage; see `packages/envs/src/missingno_envs/exploration.py`).

Reference point: Pokémon RL Edition (2025) finished Red with scripted HMs, automatic boulder puzzles, an automatic Poké Flute, infinite money and story-event rewards; it could finish the game with any one of its helps removed, but not with all of them removed at once.

**Plan B (declared in advance).** If generic rewards and structured exploration prove insufficient, human demonstrations are added as an *additional, separately reported* experiment. This would change the main claim, and would be stated as such.

---

## 5. Method

### 5.1 Environment (implemented)

- **Emulator:** PyBoy (Game Boy / Game Boy Color); mGBA through a small libretro frontend for Generation 3, with the screen resized to the Game Boy's.
- **Observation:** grayscale screen, last 3 frames stacked; downscaled 2× to 80×72 (`--downscale 2`, the first model) or at full resolution, 160×144 (`--downscale 1`, the v3 model).
- **Actions:** 7 buttons (down, left, right, up, A, B, Start), one decision every 24 frames.
- **Episodes:** start from a fixed state after the intro (the intro is setup, played once by hand), fixed maximum length.
- **Interface:** standard Gymnasium environment, deterministic resets, tested.

### 5.2 Learning methods (to be compared)

| Method | Role |
|---|---|
| Random agent | Lower bound |
| PPO | Standard baseline, used by previous Pokémon Red work |
| PPO + intrinsic motivation (RND) | Rewards novelty when the game's rewards are silent |
| PPO + Go-Explore | Archive of promising states, restart exploration from them |
| DreamerV3 (world model) | Sample-efficient learning in imagination; evaluated after Phase 1 |

Self-supervised pre-training of the visual encoder on frames collected by the agent itself is a candidate addition for the multi-game phases.

### 5.3 Games and held-out protocol

Most games have a near-identical twin (Red/Blue, Gold/Silver, Ruby/Sapphire, FireRed/LeafGreen), so the eleven games contain about seven distinct ones. The split, **held out by generation**:

- **Training:** Generation 1 (Red, Blue, Yellow) and, later, Generation 3 (FireRed, which bridges the two by retelling Red on a new console, and Emerald).
- **Held out:** Generation 2 (**Crystal**, and Gold/Silver as twins). Never trained on: the code refuses to train on it unless an explicit flag is given, so the test cannot be contaminated by mistake. It sits between the two training generations, so it asks a fair question: does what was learned on both sides transfer to the generation in between?

Same-generation tests (Yellow for a model trained on Red alone) are kept as sanity checks: they look almost identical, so success there is weak evidence of generalization.

---

## 6. Experimental plan

Each phase answers a question and produces a result that stands on its own.

| Phase | Question | Compute | Output | Gate |
|---|---|---|---|---|
| **1 — Prototype** | RQ1 on a short segment | Own PC + Kaggle | Learning curves, cost estimates | First Pokémon obtained within 100M steps by at least one method |
| **2 — One game** | RQ1, RQ2 on Red | ISCRA-C | Comparison with game-specific rewards | Cerulean City within 5B steps |
| **3 — Several games** | RQ3, RQ4 | ISCRA-B | Held-out ladder results | Held-out milestones faster than from scratch |
| **4 — Full games** | Main question | ISCRA-B / EuroHPC | Agent completing games | — |

If a gate is not met, we stop, analyse why, and revise the approach before spending more compute.

**Memory experiment.** The first long run stopped after Oak's Parcel: the next step is to bring it back south, and a memoryless agent cannot know it is carrying it (it is not on the screen). The planned comparison, same configuration and seed: no memory, short-term memory (GRU), and later a two-level memory (a GRU plus an episodic store of a few dozen "memories" written when the screen changes a lot and read by attention, in the spirit of complementary learning systems, MERLIN and Neural Episodic Control). Which memory does what is a result in itself.

**First review milestone** (between Phases 1 and 3, on own PC and Kaggle): **one model, trained on Red, Blue and Yellow together, that wins the first badge in all three**, with a replay of each first win, plus its zero-shot score on Crystal milestone by milestone. It answers RQ1 on a real goal (a badge, not just a map), gives a first measured cost per badge, and the first number on generalization to an unseen generation.

### 6.1 Phase 1 in detail

**Segment:** Pokémon Red, from the bedroom to Viridian City.

**Milestones** (measured with map IDs from the pret/pokered disassembly; 37 and 38 verified in our first run):

| # | Milestone | Signal |
|---|---|---|
| M1 | Leave the bedroom | map 38 → 37 |
| M2 | Leave the house | Pallet Town (map 0) |
| M3 | Enter Oak's lab | map 40 |
| M4 | Obtain the first Pokémon | party size > 0 |
| M5 | Reach Route 1 | map 12 |
| M6 | Reach Viridian City | map 1 |

**Experiments:** random, PPO, PPO + RND, PPO + Go-Explore. Same budget (target: 100M steps), 3 seeds each.

**Deliverables for review:** learning curves with milestone markers, a table of steps and core-hours per milestone, a video of the best agent, and an extrapolated compute estimate for Phase 2.

### 6.2 Making every step count

**The problem.** By brute force, the whole of Red is estimated at 10–50B steps (Section 7). On Kaggle (~400 steps/s) that is about 290 days for 10B and 4 years for 50B; on Leonardo, at an estimated ~0.5B steps per node-day (to be measured), about 20 node-days (~2,000 GPU-hours) for 10B and about 100 node-days (close to a whole ISCRA-C allocation) for 50B. The goal is therefore not "more steps" but **more learning per step**. Three levers, none of which breaks the rules of Section 4.1:

**Lever 1 — Split the game through the agent's own states (implemented, being measured).** If every episode starts from the bedroom, practising a late stretch means replaying everything before it, so the cost of the game grows with its length times the number of attempts. The archive and the backward curriculum restart episodes from states **the agent itself reached**, so it practises the new stretch directly: the cost adds up stretch by stretch instead. This is the idea behind Go-Explore, which solved Montezuma's Revenge. The archive and curriculum are shared by all parallel games and saved across sessions, and a win is still shown as one continuous replay from the bedroom (lineage, Section 4.1).
*Test:* in `badge_v22_s1`, does the shared curriculum make the trip back with Oak's Parcel consistent (`milestones/M8_rate`), where the per-game curriculum did not?

**Lever 2 — A world model (research branch, after the v3 baseline).** The agent also learns *how the game works* (what it will see after pressing a button) and trains in its own imagination, so every real step is reused many times. DreamerV3 learned to collect diamonds in Minecraft from scratch, without human data, with far fewer interactions than model-free methods. Costs: more GPU per step, and a new learner to build, not a flag. It is the most promising answer to the main question, and the natural object of a larger allocation.
*Test:* on the Phase 1 segment and up to the first badge, steps per milestone of a world-model agent against PPO, same rewards and observations.

**Lever 3 — More steps per second on the same hardware.** Profile one environment step (emulation, memory reads, image resizing, reward) and remove what is not needed; on HPC nodes, run actors asynchronously so all cores emulate while the GPUs learn. It does not change the order of magnitude, but a 2–3× gain is worth a month of free compute.
*Test:* measured steps/s of the v3 model on one GPU (first `red_v3_s1` run), then on a Leonardo node.

**Memory is a prerequisite, not a lever.** Many later states look identical on screen but need different actions (carrying the parcel or not, holding the Silph Scope or not). A memoryless policy cannot tell them apart with any number of steps; with too small a network, learning later areas erases earlier ones. More compute does not fix either: this is why the v3 model (IMPALA encoder, GRU, γ = 0.999) is the base for the whole game, and the current small network the base for the first badge only.

**For the ISCRA-C application.** Not "a lot of compute for brute force" but: a method that cuts the compute needed (levers 1 and 2), a measured cost per milestone and a measured throughput, and a game nobody has beaten without help.

---

## 7. Metrics and rigor

**Metrics**

- Fraction of seeds reaching each milestone.
- Steps and core-hours to reach each milestone (median and spread across seeds).
- **Transfer ratio:** steps to a milestone on a held-out game when starting from scratch, divided by steps when starting from the multi-game agent. Values above 1 mean positive transfer.
- Exploration statistics: tiles and maps visited over time.

**Rigor**

- At least 3 seeds per configuration (5 for headline results): reinforcement learning is highly variable.
- Equal step budgets across methods.
- Hypotheses, held-out games and gates fixed in advance (Sections 3, 5.3, 6).
- Every experiment logged (configuration, curves, results) and reproducible from the repository.

**Measured throughput (Phase 0).** 661 agent steps/s per core on a consumer PC (24 frames per step), i.e. about **420 core-hours per billion steps**. On an 8-thread PC, a realistic 2,500–4,000 steps/s during training: roughly 100M steps per night.

**Compute estimates** (order of magnitude, to be refined with Phase 1 measurements; "with attempts" multiplies by 3–10 for failed runs, tuning and ablations):

| Scope | Final run | With attempts | Allocation |
|---|---|---|---|
| Phase 1 segment | ~0.1B steps | fits a PC / Kaggle | none |
| Red to Cerulean | 1–5B steps (~0.4–2k core-h) | ~4–20k core-h | ISCRA-C |
| Red complete | 10–50B steps | ~40–200k core-h | ISCRA-C/B |
| Several games | 50–300B steps | 0.2–1M+ core-h | ISCRA-B |

---

## 8. Contributions

- **C1 — Environment and protocol.** Gymnasium environments for Generation 1–3 Pokémon games with identical observations and rewards, per-game milestones, and a held-out protocol. Useful to other researchers regardless of our agent's results.
- **C2 — Exploration study.** Cost per milestone of PPO, intrinsic motivation, Go-Explore and world models under generic rewards.
- **C3 — Transfer study.** What transfers along the held-out ladder, and what breaks.
- **C4 — Ambitious goal.** An agent that completes games, including games it has never seen.

C1–C3 do not depend on reaching C4.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| The agent does not pass the first milestones | Phase 1 detects this early and cheaply; C1–C2 remain valid; a well-analysed negative result is publishable |
| Transfer does not happen | A measured negative result, with the analysis of what breaks (RQ4), is itself a contribution |
| Generic rewards are not enough | Plan B: demonstrations as a separate experiment (Section 4) |
| Reward hacking (agent exploits a reward) | Rewards only for new progress (e.g. levels above the best seen); watch agents regularly |
| Compute exceeds estimates | Cost gates per phase; staged allocations (try → ISCRA-C → ISCRA-B) |
| Emulator or tooling issues (GBA, Windows) | Linux/WSL for heavy runs; GBA support only from Phase 3 |
| "Why not language models?" | The question is what can be learned *without* prior knowledge; language-model agents answer a different question, and this work is their reference point |

---

## 10. Dissemination

arXiv in any case; IEEE Conference on Games (CoG) or AIIDE; workshops at NeurIPS/ICLR/ICML on agents and exploration; the NeurIPS Datasets & Benchmarks track for C1; future editions of the PokéAgent Challenge. Code and models open on GitHub and Hugging Face. Outside academia: videos of the agent playing and the development journal.

---

## 11. Legal notes

No ROMs or save states are distributed. Anyone running the code must use legally obtained copies of their own games. Pokémon names are trademarks of their respective owners; this is a non-commercial research project.

---

## References

- Pleines, Addis, Rubinstein, Zimmer, Preuss, Whidden. *Pokémon Red via Reinforcement Learning.* IEEE CoG 2025. arXiv:2502.19920
- Whidden et al. PokemonRedExperiments and pokemonred_puffer (open-source RL for Pokémon Red; full game completed in 2025)
- Mudireddy, Patibandla. *PokeRL: Reinforcement Learning for Pokémon Red.* 2026. arXiv:2604.10812
- Karten, Grigsby et al. *The PokéAgent Challenge: Competitive and Long-Context Learning at Scale.* 2026
- Grigsby et al. *Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers* (Metamon). RLC 2025. arXiv:2504.04395
- Suarez et al. *PufferLib: Making Reinforcement Learning Libraries and Environments Play Nice.* arXiv:2406.12905
- Schulman et al. *Proximal Policy Optimization Algorithms.* 2017
- Burda et al. *Exploration by Random Network Distillation.* 2018
- Ecoffet et al. *First return, then explore* (Go-Explore). Nature 2021
- Hafner et al. *Mastering Diverse Domains through World Models* (DreamerV3). 2023; Nature 2025
- Salimans, Chen. *Learning Montezuma's Revenge from a Single Demonstration.* 2018
- Oh et al. *Self-Imitation Learning.* ICML 2018
- Espeholt et al. *IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures.* ICML 2018
- Cobbe et al. *Leveraging Procedural Generation to Benchmark Reinforcement Learning* (Procgen). 2020
- Baker et al. *Video PreTraining (VPT).* 2022
- Sutton. *The Bitter Lesson.* 2019
