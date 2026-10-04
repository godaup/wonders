# Changelog

## Unreleased

* Complete base game: full deck and wonders, with the special wonder powers mechanized
  (Babylon B, Olympia A/B, Halicarnassus).
* Optional, combinable expansions: Leaders (official deck, draft and recruitment) and
  Cities (official deck), with every leader and black-card power mechanized and
  checked against the official rulebooks; the Leaders guilds and the Rome, Petra and
  Byzantium boards; the training env supports the Leaders draft and recruitment.
* Score breakdown gains `leaders` and `cities` columns.
* Wonder-board-aware agents (`BoardAwareAgent`, `BoardFocusAgent`); observations expose the
  whole wonder board and own production.
* Agents: baselines, strategy agents and a tuned `HeuristicAgent`; `wonders.agents.bench`
  for win rates with confidence intervals.
* Training environments: fixed `ActionSpace` with masks, `WondersEnv` (multi-agent),
  `SingleAgentEnv` (Gym-style), `SyncVectorEnv` / `SubprocVectorEnv`, `CardEncoder`.
* Replayable `GameRecord` notation (version 2 adds power-granted decisions).
* About 2.8x faster engine and observations, behaviour verified unchanged by a golden test.
* CLI: `wonders play | replay | bench | agents | gallery | table`.
* Optional learned agent (`pip install wonders[ml]`): policy/value network, imitation
  learning, PPO, a bundled pretrained model and `wonders train imitate | ppo`.
* Requires Python 3.10+.

## 0.1.0 (2022-01-21)

* First release on PyPI.
