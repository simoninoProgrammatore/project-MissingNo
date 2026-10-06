# Archive: Showdown battles

Early exercise from Phase 0: a rule-based battle player on a local Pokémon Showdown server, with a script to compare players and measure win rates with confidence intervals.

The project has since moved to an **end-to-end agent that plays from the screen**, so this code is no longer part of the main workspace. It is kept here for reference and as a learning exercise. It is not maintained and may need small fixes to run (it used `poke-env` and the `missingno-envs` Showdown helper, both now removed from the workspace).

Last measured results (gen1randombattle, 150 battles):

- rule-based vs `MaxBasePowerPlayer`: 70% wins
- rule-based vs `SimpleHeuristicsPlayer`: 33% wins
