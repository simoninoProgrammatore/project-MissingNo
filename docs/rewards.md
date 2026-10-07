# Project MissingNo — Reward design (RL only)

> **Status:** v1 and v2 are implemented (`packages/envs/src/missingno_envs/rewards.py`, selected with `--reward-version`; v2 is the default). v3 and v4 are design proposals. Components are added one at a time and ablated (Section 9). Weights are tuned **only on training games** and frozen before any held-out evaluation.

The reward is the only thing that tells a tabula rasa agent what "playing well" means. It must push toward progress in *every* Generation 1–3 game without ever saying *what* to do or *where* to go.

---

## 1. Rules every component must follow

1. **Same rule in every game.** A component is a function of game-agnostic signals produced by the adapters. If writing it requires knowing a specific game, it is walkthrough knowledge and is not allowed.
2. **"Something new happened", never "do this".** Components reward novelty and irreversible progress, not specific actions or places.
3. **First time only, or bounded.** Anything that can be repeated must pay once (per episode or per lifetime) or have a hard cap. Repeatable rewards are farmed.
4. **Monotone progress uses max-so-far.** Progress signals pay only when they exceed the best value seen, so losing and regaining progress is never profitable.
5. **Recoverable quantities use potentials.** For things that go up and down (HP, PP), reward the *change* of a potential, `F = γ·Φ(s') − Φ(s)`. Over any loop that returns to the same state the total is ≈ 0, so cycles cannot be exploited (potential-based shaping, Ng et al. 1999).
6. **Stagnation ends episodes, it is not punished.** Long stretches without progress truncate the episode early (Section 6). Per-step penalties teach agents to stand still, or worse, to end episodes on purpose.
7. **The agent never sees the reward signals.** Adapters read memory for the teacher (rewards, milestones), never for the student (observations).

Note on rule 4: max-so-far and visited-set rewards are not Markov in the raw game state, so the classic invariance guarantee of potential-based shaping does not strictly apply to them. They are Markov in the augmented state (game state + what has been achieved this episode), which is what the agent effectively optimizes.

---

## 2. Signals the adapters must provide

All are generic and exist in every Generation 1–3 game. Gen 2 and Gen 3 identify maps by (group, number): adapters must flatten that into one unique `map_id`.

| Signal | Use |
|---|---|
| `map_id`, `x`, `y`, map width/height | exploration |
| `in_battle`, battle type (wild / trainer), trainer identity | battle rewards, filters |
| party: species, level, HP/max HP, PP/max PP, moves | levels, survival, moves |
| badges | progress |
| Pokédex seen / owned counts | progress |
| bag: set of item IDs ever held, per pocket where available | progress |
| "Hall of Fame entered" | the win condition |
| sanity flags (Section 7) | corruption guard |

Plus one signal read from the **screen** without any adapter: the dialogue box (Section 3.5).

---

## 3. Components

Weights below are **starting points**, expressed relative to one badge = 10. They are tuned in Phase 1 (Section 8).

### 3.1 Exploration

| Component | Rule | Start weight |
|---|---|---|
| New tile (episodic) | first visit to `(map, x, y)` in this episode | 0.02 |
| New map (episodic) | first visit to a map in this episode, **scaled by map area** | 0.2 × min(1, area / 400) |
| Lifelong novelty | RND on observations (Section 5) | normalized, own value head |

**Why scale new maps by area.** Every house, Poké Mart and Pokémon Center interior is a separate map. A flat "new map" bonus turns the agent into a tourist who walks in and out of every door in town. Scaling by area makes a new route or cave worth much more than a one-room house, which matches how much there is to discover.

**Edge cases**

- *Forced movement* (spin tiles in Rocket hideouts, ice floors, sea currents, the Cycling Road slope): tiles are still novel, no special handling, but expect bursts of reward the agent did not "choose".
- *Teleport pads and warp mazes* (Silph Co., Saffron and Mossdeep-style gyms): first-visit rewards make ping-ponging between pads worthless after the first time.
- *Huge open areas* (water routes, the Safari Zone): tile novelty can produce "lawnmower" behavior. If observed, coarsen tiles to 2×2 cells on maps above a size threshold.
- *Safari Zone step limit* resets the player to the entrance: harmless, episodic novelty already paid.
- *Dark caves*: coordinates still come from memory, novelty works without Flash.

### 3.2 Progress (max-so-far)

| Component | Rule | Start weight |
|---|---|---|
| Badge | each new badge (Gen 2 has 16, same weight each) | 10 |
| Trainer defeated | **first** win against each trainer identity | 0.5 |
| Pokédex owned | each new species owned, concave: `1/√(1 + owned/20)` | 0.3 × factor |
| Pokédex seen | each new species seen | 0.05 |
| New item | first time an item ID is ever held | 0.3, or 1.0 if it cannot be bought |
| New move | first time any party Pokémon knows a move ID | 0.05 |
| Team strength | max-so-far of `Σ_top-4 log(level)` | 1.0 per unit |
| **Hall of Fame** | entering it ends the episode | 50, terminal |

**Levels: what a strong player actually optimizes.** Raw level sums reward grinding one Pokémon to 70 in the first route. A strong player cares about a team of a few Pokémon at reasonable levels. The `Σ_top-4 log(level)` potential has three properties:

- *Concave*: the 10th level of a Pokémon is worth more than its 50th, so grinding past usefulness stops paying.
- *Top-4*: a fifth and sixth Pokémon only count if they become stronger than the current core, which rewards building a team without punishing catches.
- *Max-so-far*: depositing and re-withdrawing Pokémon, or swapping the party in a Bug-Catching Contest, can never generate reward.

**Items: "key item" is generic metadata.** Every Generation 1–3 game marks some items as key items (from Generation 3 they even have their own bag pocket); HMs can never be bought. For Red, the set is taken from the game's own `KeyItemFlags` table, not written by hand. Weighting them higher rewards the NPC who gives you an item without naming any NPC. Shop items stay bounded: there are only a few dozen types.

**Trainers: first win per identity.** Rematch systems (VS Seeker in FireRed/LeafGreen, Match Call in Emerald, phone rematches in Gen 2) would otherwise be farms.

**Pokédex seen is a free exploration signal.** New areas have new wild species, so "seen" rewards entering new grass, caves and water without any map knowledge. Evolution registers a new owned species, so it is rewarded naturally.

**Edge cases**

- *Rare Candy duplication.* Generation 1 contains item duplication glitches. An agent that finds one can max out levels. The concave team potential bounds the gain, and the corruption guard (Section 7) catches the impossible states that usually come with it.
- *Day Care levels* (deposit, walk, withdraw): legitimate, bounded by money, allowed.
- *Game Corner prizes*: new items and species from the prize counter are legitimate and bounded.
- *Money*: never rewarded. Rewarding money creates sell-everything loops and gambling at the slot machines.

### 3.3 Survival (potentials)

| Component | Rule | Start weight |
|---|---|---|
| Party HP | potential `Φ = mean(HP / max HP)` over the party | 0.5 |
| Party PP | potential `Φ = mean(PP / max PP)` over known moves | 0.2 |
| Blackout | all party fainted | −2 |

Potentials teach the agent that damage is bad and healing is good, and **cannot be exploited**: taking damage and healing at a Pokémon Center nets ≈ 0. They are what makes "go back and heal before the gym" learnable without any rule about Pokémon Centers.

**Edge cases**

- *Death warping*: blacking out teleports you to the last Pokémon Center, a real speedrun technique. The blackout cost plus the money lost makes it a deliberate trade-off rather than a free shortcut. We accept that the agent may learn it.
- *Party size changes* (catching, depositing, releasing): the HP and PP potentials are means, so they do not jump with party size.
- *Releasing Pokémon in the PC*: possible and catastrophic. Team strength is max-so-far, so releasing earns nothing; the loss of strength shows up in later battles. No special rule.
- *Struggle* (all PP at 0): the PP potential already pushes away from it.

### 3.4 Battles

Battles have no direct "win" reward beyond the first win per trainer. Wild battles pay through levels, Pokédex seen and owned. This avoids the classic farm of fighting weak wild Pokémon forever. Fleeing is neutral.

### 3.5 Dialogue novelty (read from the screen)

| Component | Rule | Start weight |
|---|---|---|
| New dialogue | first time a dialogue text is seen in this episode, max N per map | 0.1, N = 10 |

Dialogue is where the games tell you where to go. This rewards *reading new text*, which in practice means talking to new people and reading signs, without knowing who says what.

**How it is detected:** the dialogue box region of the screen (the bottom window), binarized and hashed. No adapter needed, so it works unchanged on any game.

**Edge cases, and these are the dangerous ones**

- *Battle text* ("Wild X appeared!", "It's super effective!"): excluded when `in_battle`.
- *Menus* (Start menu, bag, Pokédex, PC, summary screens) also show text: only the dialogue box region counts, and only outside menus.
- *Variable text* (numbers, money, coins, the player's and rival's names, Pokémon nicknames, time-dependent lines in Gen 2, weather in Gen 3) would make the same NPC "new" forever. Digits are masked before hashing, and the per-map cap bounds whatever slips through.
- *Phone calls in Gen 2*: random calls produce new text while walking. The per-map cap bounds them.
- *Yes/no prompts* (saving, trades, nurse): first time only, harmless. In-game trades can give away a strong Pokémon: max-so-far team strength means no reward is lost, only future battle strength.

### 3.6 Deliberately excluded

| Not used | Why |
|---|---|
| Story event flags (or their count) | The game's own map of the story: walkthrough knowledge in disguise. Tested only as a separate, declared experiment |
| HM-specific rewards | Genre knowledge that says "do this". Separate ablation |
| Money | Farmable (selling, gambling) |
| Per-step time penalty | Teaches standing still or ending episodes |
| Distance to any target | Requires knowing where to go |

---

## 4. Discounting: the hidden problem

With `γ = 0.998` the effective horizon is about 500 steps, roughly 3–4 minutes of game time. A badge that is two hours away is invisible to the agent, whatever its weight: **no weight can fix a reward the discount has already erased.**

Solution, following the RND paper: **two value heads**.

- *Extrinsic head* (progress, survival, dialogue): high `γ` (0.9995–0.9999), episodic.
- *Intrinsic head* (lifelong novelty): lower `γ` (0.99), non-episodic.

The advantages of the two heads are combined with a weight. This also lets intrinsic rewards stay large early and fade as the world becomes familiar, without drowning the progress signals.

---

## 5. Two kinds of novelty

Inspired by Never Give Up (Badia et al., 2020):

- **Episodic novelty** (tiles, maps, dialogue, all "first time in this episode"): pushes the agent to explore *within* each run, so every episode is a full exploration attempt.
- **Lifelong novelty** (RND on observations): decays as the agent sees the same places across training, and pushes it toward parts of the world no episode has reached yet.

Episodic alone makes the agent re-explore the same early towns every episode; lifelong alone makes it lose interest in towns it must still cross. Together they cover both.

---

## 6. Stagnation: truncation, not penalty

If an episode produces **no extrinsic progress and no new tile for K steps** (start: K = 2,000), it is truncated. Effects:

- no compute wasted on an agent stuck in a corner or looping in a menu;
- no penalty, so no incentive to self-destruct;
- with Go-Explore, the last good state is already in the archive and exploration restarts from there.

Loops this catches: door ping-pong, menu spam, walking against a wall, oscillating at a ledge, circling a Pokémon Center.

---

## 7. Corruption guard

Generation 1 is famously glitchy (the project is named after one of those glitches), and RL agents are very good at finding glitches. An adapter-level sanity check validates invariants every step:

- party size between 0 and 6; levels between 1 and 100; HP ≤ max HP; valid species and move IDs; coordinates inside the map.

If an invariant breaks, the episode ends **with no reward for that step**, and the event is logged with a save state for inspection. We do not forbid glitches the agent finds legitimately, but we do not let corrupted memory turn into reward.

---

## 8. Normalization and tuning

**Normalization.** Each component is normalized by a running estimate of the standard deviation of its per-episode return, so weights stay comparable across games with different scales. No reward clipping: clipping to ±1 would erase the hierarchy between a badge and a tile.

**Hierarchy, as an expert would set it.** Badges ≫ unbuyable items ≈ first trainer wins > large new maps > Pokédex owned > dialogue > tiles. A rough sanity check: one badge should be worth on the order of a few hundred new tiles, so that exploring an entire route never beats beating a gym.

**Tuning.** Weights are tuned in **groups**, not individually: exploration, progress, survival, dialogue. Four knobs instead of fifteen. Phase 1: a small grid on the Pallet Town → Viridian City segment. Later on CINECA: population-based training, which tunes weights automatically and removes our own intuitions about Red from the loop. Report the sensitivity: if small weight changes flip the results, the reward is fragile, and that is a finding.

**Audit.** Every run logs the per-episode return **of each component**. If one component dominates, inspect videos before trusting the curves: that is how reward hacking is caught.

---

## 9. Rollout and ablations

| Version | Adds | Where |
|---|---|---|
| v1 ✅ | tiles, maps, badges, levels (linear, max-so-far) | Phase 1 baseline |
| v2 ✅ (default) | area-scaled maps, concave top-4 levels, new items (key items worth more), Pokédex, stagnation truncation | Phase 1 |
| v3 | dialogue novelty, HP/PP potentials, blackout, first trainer wins, corruption guard | Phase 1–2 |
| v4 | two value heads, episodic + lifelong (RND) novelty | Phase 2 |
| Declared extras | event-flag count; HM-specific reward | separate experiments, reported apart |

Each version is compared with the previous one on the same segment, budget and seeds. A component stays only if it helps or is needed to block an observed exploit.

---

## References

- Ng, Harada, Russell. *Policy invariance under reward transformations.* ICML 1999
- Burda et al. *Exploration by Random Network Distillation.* 2018
- Badia et al. *Never Give Up: Learning Directed Exploration Strategies.* ICLR 2020
- Ecoffet et al. *First return, then explore* (Go-Explore). Nature 2021
- Pleines et al. *Pokémon Red via Reinforcement Learning.* IEEE CoG 2025 (reward exploitation in ablations)
- Jaderberg et al. *Population Based Training of Neural Networks.* 2017
