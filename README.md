# wonders

[![Build Status](https://github.com/godaup/wonders/actions/workflows/dev.yml/badge.svg)](https://github.com/godaup/wonders/actions/workflows/dev.yml)

A Python engine for a **7 Wonders-style card drafting game**, built for agents and deep
learning: a clean, partial-observability-correct state/action interface, scripted agents,
training environments, a replayable game notation, benchmarks, a trained network that plays
every expansion combination, and a local browser UI to play against them.

> **Disclaimer.** This is an independent, non-commercial fan project. It is not affiliated
> with, endorsed or sponsored by Repos Production or any publisher of *7 Wonders*.
> *7 Wonders* is a trademark of its respective owners. No publisher artwork is included: all
> card and board renderings are generated from game data.

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

`wonders gui` serves a local page (http://127.0.0.1:8765; nothing leaves your machine): pick
expansions, opponents per seat and your wonder, then play with card renderings, build costs
including trade, the neighbours' cities and a move log; the game record can be downloaded at
any time. `--checkpoint model.pt` adds your own network as an opponent.

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

## Further reading

* `docs/encoder-v2.md` — design of the `table-v2` observation encoder.
* `docs/adaptive-agents.md` — the adaptive agents, their rules and benchmarks, with the strategy
  research behind them.
* `sweeps/*.md` — training reports (`STAGE0`–`STAGE3`, `OBJECTIVES`): what was tried, the
  results and how they were measured. Each report's runs are reproducible from the JSON spec
  next to it (`wonders train sweep <spec>`); the raw run outputs are not part of the repository.
* `CONTRIBUTING.md` — how to contribute; `CLAUDE.md` — an architecture overview written for
  AI coding assistants, useful for human contributors too.

## License

MIT — see `LICENSE`. The disclaimer above applies.
