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
  learning, PPO and `wonders train imitate | ppo`.
* Bundled universal network (`ppo_universal_v2.pt`) for every expansion combination and 3–7
  players, replacing the base-game-only model; the `table-v2` observation encoder (every city
  seat-relative, rich hand / private / history blocks) is the default for new models. The bundled
  network is the learning-rate-decayed `s5-anneal` (53.2% head-to-head over its predecessor).
* Training: self-play league with a tiered opponent mix (scripted / recent self / anchors),
  long-lived self-anchors, large neural opponents on the GPU, linear learning-rate decay, a
  rank-shaped reward option, resumable JSON sweeps (`wonders train sweep`).
* Evaluation: `wonders train eval | h2h | exploit | leaders` — scripted lineups, models
  head-to-head with random seating, exploitability, Leaders diagnostics.
* Agents: data-driven leader play for the heuristic family (`leader_values`), the adaptive
  scripted agents `adaptive` and `rival`, leader-decision ablations; `NeuralAgent` plays the
  Leaders draft and recruitment with its own policy.
* `wonders gui`: play against agents in the browser (local only).
* `Effect.counts()` hook describing what counting effects pay per unit.
* Requires Python 3.10+.

## 0.1.0 (2022-01-21)

* First release on PyPI.
