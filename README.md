<p align="center">
  <img src="docs/assets/banner.svg" alt="Project MissingNo banner: a glitching pixel emblem over a duel between two original creatures" width="100%">
</p>

<h1 align="center">Project MissingNo</h1>

<p align="center"><em>Teaching a small AI to play Pokémon, and learning how AI is made along the way.</em></p>

---

## The idea

Project MissingNo is an attempt to build an artificial intelligence that can play the Pokémon games, and play them well: battling, exploring the world, catching Pokémon, beating gyms, and eventually finishing a whole game.

The twist is in *how*. Instead of relying on a huge, expensive AI model running somewhere in the cloud, the goal is to build something **small, open and trained from scratch**: models that anyone can download, inspect and run on their own computer. The aim is not just to finish a game once by luck, but to get an AI that plays sensibly in general, and maybe one day can pick up a Pokémon game it has never seen before.

It is also a personal journey. This project is how I am learning, hands-on, how AI models are actually trained. I know these games inside out; machine learning, not yet. Every step, including the failures, will be documented in the [journal](JOURNAL.md).

## Why "MissingNo"

Anyone who played the original Red and Blue remembers the rumors. There was a strange creature you could meet through a trick on Cinnabar Island's coast: a garbled block of pixels with no proper name, no Pokédex entry, and a reputation for messing with your save file. Its name, *MissingNo.*, is short for "missing number": it was never meant to exist. It appeared because the game was reading its own memory in a place where no real data lived.

It became a playground legend, and it felt like the right name for this project. This AI also lives inside the game's memory, reading it to understand what is happening. And like the original, it is something the game was never designed to contain: a player that is not a person.

## What inspired it

Pokémon has quietly become one of the most interesting playgrounds for artificial intelligence.

In 2014, **Twitch Plays Pokémon** let tens of thousands of people play Pokémon Red together by typing commands into a chat. It was chaotic, hilarious, and somehow it worked.

In 2023, a video by **Peter Whidden** showed an AI learning to play Pokémon Red completely on its own, starting from random button presses. That project grew into an open community effort that, in 2025, managed to beat the whole game.

Since then, large AI systems from several major labs have played and finished the original games, though with a lot of help from custom-built tools. And in 2025, Pokémon got its own research competition at NeurIPS, one of the biggest AI conferences in the world, with over a hundred teams building AI players.

What is still missing is a small, open AI that plays these games well in a general way, built by someone who knows the games deeply. That is the gap this project wants to explore.

## Where we are

**Right now, we are studying.** There is no final design yet, and that is on purpose. Before building the full system, the most important thing is to understand what has been tried, what worked, and why.

Some of the big questions we are exploring:

- **One big brain, or a team of specialists?** Should a single AI learn everything, or should there be separate "experts" for battling and for exploring, coordinated by a kind of coach?
- **How should it learn?** By watching millions of battles played by humans, by playing against itself, or both?
- **What should it see?** Should it read the game's internal memory, or look only at the screen, like a human does?
- **How do we know it is getting better?** One idea is a collection of "hard spots": moments in the games where players and AIs commonly get stuck, carefully chosen by someone who knows the games well, used as a test.

The first experiments will focus on **battles**, the most self-contained part of the game and a good place to learn the basics. A simple rule-based battler already exists as a starting point to beat.

## The road ahead

1. **Study** what others have done and learn the fundamentals.
2. **Battles:** train a first AI that battles better than simple rules.
3. **Exploring the world:** teach an AI to move around and reach places on the map.
4. **Putting it together:** combine everything into a player that makes progress through the story.
5. **Beyond:** try it on a different Pokémon game and see what carries over.

There are no fixed deadlines. This is a learning project, and the order matters more than the speed.

## Follow along

- 📓 [Journal](JOURNAL.md): what was tried, what broke, what was learned
- 🗺️ [Working plan](docs/plan.md): the current, evolving plan (a draft, not a final design)
- 🛠️ [Setup](docs/SETUP.md): for anyone who wants to run the code

## Disclaimer

This is an unofficial fan and research project. It is not affiliated with, endorsed by, or connected to Nintendo, Game Freak, Creatures Inc. or The Pokémon Company. Pokémon and all related names are trademarks of their respective owners. No game files (ROMs) are included or distributed: anyone running the code must use a legally obtained copy of their own game.
