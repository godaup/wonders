<p align="center">
  <img src="docs/images/logo.svg" width="112" alt="wonders logo">
</p>

<h1 align="center">wonders</h1>

<p align="center">
  <b>Seven wonders, three Ages, one engine — built for people and for neural networks.</b><br>
  A complete, rules-checked engine for a 7&nbsp;Wonders-style drafting game, with agents, training
  environments, a trained network and a browser table to play against them.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%20%E2%80%93%203.13-3776ab?logo=python&logoColor=white" alt="Python 3.10–3.13">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license">
  <img src="https://img.shields.io/badge/code%20style-black-000000" alt="code style: black">
  <img src="https://img.shields.io/badge/types-mypy-2a6db2" alt="type-checked with mypy">
  <img src="https://img.shields.io/badge/expansions-base%20%C2%B7%20leaders%20%C2%B7%20cities-c8951a" alt="expansions: base, leaders, cities">
  <img src="https://img.shields.io/badge/PyTorch-optional-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch optional">
</p>

<p align="center">
  <img src="docs/images/game.png" width="900" alt="Playing a five-player game with Leaders and Cities in the browser">
</p>

> **Disclaimer.** This is an independent, non-commercial fan project. It is not affiliated
> with, endorsed or sponsored by Repos Production or any publisher of *7 Wonders*.
> *7 Wonders* is a trademark of its respective owners. No publisher artwork is included: all
> card and board renderings are generated from game data.

### The zen of wonders

* **Cards are data, mechanics are effects.** Every card lives in JSON; new rules are new
  composable effects, never special cases.
* **A seat sees only what it may see.** Agents, encoders and the UI read one seat's
  observation — hidden hands and buried cards stay hidden.
* **Every game replays exactly.** A record is like a chess PGN: the same moves, the same game.
* **Rules first.** Expansions and powers are checked against the official rulebooks.
* **Fast and dependency-light.** Pure Python with `click`; PyTorch only for the learned agent.

## Contents

* [Features](#features)
* [Play in your browser](#play-in-your-browser)
* [Installation](#installation)
* [Command line](#command-line)
* [Training](#training-needs-pytorch)
* [Python API](#python-api)
* [The bundled network](#the-bundled-network)
* [Roadmap and further reading](#roadmap-and-further-reading)
* [License](#license)

## Features

* **Rules engine** — the base game for 3–7 players (68 cards, 10 guilds, 7 wonders with A and
  B sides, all wonder powers) plus the optional, combinable **Leaders** (draft and recruitment)
  and **Cities** expansions with every power mechanized. Cards are data
  (`wonders/cards/data/*.json`); mechanics are composable effects.
* **Agents** — from simple baselines (`random`, `greedy`, colour archetypes) to the
  board-aware `board` and the situational `adaptive` agent; `wonders agents` lists all of them.
  Every agent reads only what its seat may legally see.
* **Training environments** — a fixed discrete action space with legal-action masks, a
  PettingZoo-style multi-agent `WondersEnv`, a Gym-style `SingleAgentEnv` and multi-process
  vector envs. No dependency on Gym or PettingZoo.
* **Learned agent** (optional, PyTorch) — imitation learning, PPO with a self-play league and a
  tiered opponent mix, sweeps, and a bundled network for every expansion combination and
  player count.
* **Evaluation** — benchmarks with 95% confidence intervals, head-to-head comparison of models
  at the same tables, exploitability measurement.
* **Replayable game records** (like PGN for chess) and a **browser UI** for human play.
* The core is pure Python with `click` as its only runtime dependency.

## Play in your browser

```
wonders gui
```

A local page (http://127.0.0.1:8765; nothing leaves your machine): pick expansions, an opponent
for every seat and your wonder, then play. Cards and boards are rendered from the game data, build
costs include trade (and you choose which neighbour to buy from), every city shows its stats,
resources and wonder stages, and hovering anything explains it. The final score sheet is
colour-coded and sortable, and the game record can be downloaded at any time.

<p align="center">
  <img src="docs/images/cards.png" width="900" alt="Card renderings: raw material, commercial, military, civilian, guild, leader and city cards">
</p>
<p align="center">
  <img src="docs/images/scores.png" width="760" alt="The final score sheet">
</p>

## Installation

```
pip install "git+https://github.com/godaup/wonders"            # core
pip install "wonders[ml] @ git+https://github.com/godaup/wonders"  # + the learned agent (PyTorch)
```

For development (Python 3.10+):

```
git clone https://github.com/godaup/wonders && cd wonders
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test,dev]"      # add ml for the learned agent
make test                         # format, lint (flake8 + mypy) and the test suite
```

For training on an NVIDIA GPU install the matching PyTorch wheel first, e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu128`.

## Command line

```
wonders play -a heuristic,greedy,random --record game.json   # one game: transcript and scores
wonders play -e base -e leaders -e cities -a neural,adaptive,board,heuristic
wonders replay game.json                                      # re-simulate a record
wonders bench -a adaptive,board,heuristic -n 500 -p 4         # win rates with 95% CIs
wonders agents                                                # the built-in agents
wonders gui                                                   # play against agents in your browser
wonders gallery -o cards.html                                 # every card and wonder board as HTML
wonders table -o table.html                                   # render the final table of a demo game
```

`-e` lists the expansions in play (include `base`); `-a` takes one agent per seat — a built-in
name, `neural` for the bundled network, or `neural:<checkpoint.pt>`.

`wonders gui --checkpoint model.pt` adds your own network as an opponent in the browser table.

### Training (needs PyTorch)

```
wonders train imitate -n 3000 -p 6 -o imitation.pt         # behaviour cloning of a scripted agent
wonders train ppo --init imitation.pt -n 150 -o ppo.pt     # PPO against built-in opponents
wonders train sweep sweeps/stage3.json -p 6                # a full recipe from a JSON spec (resumable)
wonders train eval ppo.pt                                  # vs scripted lineups, per expansion and player count
wonders train h2h a=one.pt b=other.pt -p 6                 # models at the same tables (random seating)
wonders train exploit e=exploiter.pt t=target.pt           # one model at tables full of copies of another
wonders train leaders ppo.pt                               # Leaders diagnostics (teacher agreement, ablations)
```

## Python API

```python
import random
from wonders.agents import GreedyAgent, HeuristicAgent
from wonders.env import SingleAgentEnv

env = SingleAgentEnv([HeuristicAgent(), GreedyAgent()], learner=0, reward='margin')
obs, info = env.reset(seed=0)
done = False
while not done:
    legal = [i for i, ok in enumerate(obs['action_mask']) if ok]
    obs, reward, done, truncated, info = env.step(random.choice(legal))  # your policy here
print(info['scores'])
```

For throughput, batch games across processes with `wonders.env.SubprocVectorEnv` and a
`SingleAgentEnvFactory`. Agents implement `decide(observation) -> action` (see
`wonders/agents/base.py`); the observation and action space are documented in
`wonders/engine/observation.py` and `wonders/env/`.

## The bundled network

`wonders/ml/pretrained/ppo_universal_v2.pt` plays **every expansion combination and 3–7
players**: a 256×2 policy/value network over the `table-v2` observation encoder, cloned from the
`adaptive` agent and trained for 40M PPO steps against a mix of scripted agents, its own recent
snapshots and the best earlier models, then 10M more with a linearly decaying learning rate
(`sweeps/OBJECTIVES.md`, run `s5-anneal`). Head-to-head (random seating, heuristic agents in the
other seats; all expansions, 3–7 players, 4,000 games per pair) it ranks above:

| opponent at the same table | share of games ranked higher |
|---|---|
| the previously bundled network (the same model before the decay phase) | 53.2% |
| `adaptive`, the strongest scripted agent | 87.8% |
| `board` | 88.4% |

## Roadmap and further reading

* `docs/encoder-v2.md` — design of the `table-v2` observation encoder.
* `docs/adaptive-agents.md` — the adaptive agents, their rules and benchmarks, with the strategy
  research behind them.
* `sweeps/*.md` — training reports (`STAGE0`–`STAGE3`, `OBJECTIVES`): what was tried, the
  results and how they were measured. Each report's runs are reproducible from the JSON spec
  next to it (`wonders train sweep <spec>`); the raw run outputs are not part of the repository.
* `ROADMAP.md` — planned work: retraining after the rules fixes, search, post-game analysis and
  the remaining expansions.
* `CONTRIBUTING.md` — how to contribute; `CLAUDE.md` — an architecture overview written for
  AI coding assistants, useful for human contributors too.

## License

MIT — see `LICENSE`. The disclaimer above applies.
