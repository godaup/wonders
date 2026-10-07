# wonders


[![pypi](https://img.shields.io/pypi/v/wonders.svg)](https://pypi.org/project/wonders/)
[![python](https://img.shields.io/pypi/pyversions/wonders.svg)](https://pypi.org/project/wonders/)
[![Build Status](https://github.com/Stabmixxer/wonders/actions/workflows/dev.yml/badge.svg)](https://github.com/Stabmixxer/wonders/actions/workflows/dev.yml)



An implementation of a nice card game optimized for Deep Learning integration


* Documentation: <https://Stabmixxer.github.io/wonders>
* GitHub: <https://github.com/Stabmixxer/wonders>
* PyPI: <https://pypi.org/project/wonders/>
* Free software: MIT


## Features

* **Complete 7 Wonders base game** for 3–7 players: all 68 cards, 10 guilds and the
  7 wonders (A and B sides), including the special wonder powers (Babylon's 7th card,
  Olympia's free build and guild copy, Halicarnassus' discard build). Cards are data
  (`wonders/cards/data/*.json`); mechanics are composable effects.
* **Optional, combinable expansions**: Leaders (draft + recruitment) and Cities.
* **Agents**: random, greedy, colour archetypes, strategy agents and a tuned
  `HeuristicAgent`, all reading a partial-observability-correct `Observation`.
* **Training environments**: a fixed discrete action space with legal-action masks, a
  PettingZoo-style multi-agent `WondersEnv`, a Gym-style `SingleAgentEnv`, and
  multi-process vector envs (`SubprocVectorEnv`).
* **Replayable game records** (like PGN for chess) and an **agent benchmark** with
  confidence intervals.
* Pure Python, no runtime dependencies besides `click`.

## Quickstart

```
pip install -e ".[test]"

wonders play -a heuristic,greedy,random --record game.json   # one game, transcript + scores
wonders replay game.json                                      # re-simulate a record
wonders bench -a heuristic,greedy,builder -n 500 -p 4         # win rates with 95% CIs
wonders agents                                                # list built-in agents
wonders gui                                                   # play against agents in your browser
```

`wonders gui` serves a local page (http://127.0.0.1:8765, nothing leaves your machine): pick
expansions, opponents per seat and your wonder in the menu, then play with card renderings,
build costs incl. trade, neighbours' cities and a move log; the record can be downloaded at
any time. Add `--checkpoint model.pt` to offer your own trained network as an opponent.

Training against baseline opponents:

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
`SingleAgentEnvFactory`.

## Learned agent (optional, PyTorch)

```
pip install -e ".[ml]"          # CPU wheels: pip install torch --index-url https://download.pytorch.org/whl/cpu
                                # NVIDIA GPU: pip install torch --index-url https://download.pytorch.org/whl/cu128
                                # (training uses CUDA automatically when available)

wonders play -a neural,heuristic,greedy                 # the bundled trained network
wonders play -e base -e leaders -e cities -a neural,adaptive,board,heuristic
wonders bench -a neural,heuristic,greedy -n 600 -p 6

wonders train imitate -n 3000 -p 6 -o imitation.pt      # clone HeuristicAgent
wonders train ppo --init imitation.pt -n 150 -o ppo.pt  # fine-tune with PPO
wonders bench -a neural:ppo.pt,heuristic,greedy
wonders train sweep sweeps/stage3.json -p 6             # the full recipe of the bundled model
wonders train h2h a=one.pt b=other.pt -p 6              # models head-to-head (random seating)
```

The bundled network (`wonders/ml/pretrained/ppo_universal_v2.pt`) plays **every expansion
combination and 3–7 players**: a 256×2 network over the `table-v2` encoder, cloned from the
`adaptive` agent and trained for 40M PPO steps against a tiered mix of scripted agents, its own
recent snapshots and the best earlier models (`sweeps/STAGE3.md`). Head-to-head (random
seating, heuristic agents filling the other seats), it ranks above:

| opponent at the same table | share of games ranked higher |
|---|---|
| the previously bundled base-game model (base, 3–7 players) | 88.0% |
| `adaptive`, the strongest scripted agent (all expansions, 3–7 players) | 86.2% |
| `board` (all expansions, 3–7 players) | 87.6% |

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [waynerv/cookiecutter-pypackage](https://github.com/waynerv/cookiecutter-pypackage) project template.
