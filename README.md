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
```

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

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [waynerv/cookiecutter-pypackage](https://github.com/waynerv/cookiecutter-pypackage) project template.
