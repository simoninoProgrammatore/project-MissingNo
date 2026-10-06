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
4. **Generic rewards.** The same reward function for every game, computed from game-agnostic progress signals: first visit to a tile, first visit to a map, new badges, new party levels above the best seen.
5. **Adapters for rewards and evaluation only.** One small adapter per game reads its memory to compute rewards and milestones. This privileged information is used *by the training and evaluation code*, never by the agent.
6. **Game-specific milestones are for measuring, not for rewarding.** Milestones (e.g. "reached Viridian City") are defined per game to evaluate progress; they never enter the reward.
7. **Pre-registration.** Hypotheses, held-out games, budgets and cost gates are written in this document before running the corresponding experiments.

**Plan B (declared in advance).** If generic rewards and structured exploration prove insufficient, human demonstrations are added as an *additional, separately reported* experiment. This would change the main claim, and would be stated as such.

---

## 5. Method

### 5.1 Environment (implemented)

- **Emulator:** PyBoy (Game Boy / Game Boy Color). A GBA emulator (mGBA) will be needed for Generation 3.
- **Observation:** grayscale screen, downscaled 2× to 80×72, last 3 frames stacked.
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

Most games have a near-identical twin (Red/Blue, Gold/Silver, Ruby/Sapphire, FireRed/LeafGreen), so the eleven games contain about seven distinct ones. The proposed split, to be fixed before Phase 3:

- **Training:** Red (Gen 1), Crystal (Gen 2), Ruby (Gen 3).
- **Held-out ladder**, from easiest to hardest:
  1. **Twins** (Blue, Gold/Silver, Sapphire, LeafGreen): sanity checks.
  2. **Yellow:** same region as Red, different story details.
  3. **FireRed:** same story as Red, different console, graphics and engine.
  4. **Emerald:** a Generation 3 game related to Ruby, used to test transfer within a generation.

A stricter variant, run if resources allow: train on Generations 1–2 only and test on Generation 3 (new console, new world, new mechanics).

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
- Hafner et al. *Mastering Diverse Domains through World Models* (DreamerV3). 2023
- Cobbe et al. *Leveraging Procedural Generation to Benchmark Reinforcement Learning* (Procgen). 2020
- Baker et al. *Video PreTraining (VPT).* 2022
- Sutton. *The Bitter Lesson.* 2019
