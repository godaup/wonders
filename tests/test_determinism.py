"""Golden-master test: a fixed set of games must play out exactly the same.

Performance work (caching, shared observation pieces) must never change behaviour.
This plays a varied, seeded batch — 3 to 7 players, both wonder sides, the special
wonder powers, every baseline agent family and the training env — and compares a
digest of every record and score breakdown against a stored value.

If a deliberate *rules or agent* change alters games, update ``EXPECTED`` (print
``_digest()``) and say so in the commit; a pure refactor must leave it unchanged.
"""

import hashlib
import json
import random
from typing import Callable, List

from wonders.agents import Agent, BuilderAgent, GreedyAgent, HeuristicAgent, RandomAgent, TableAgent
from wonders.agents.match import play_match
from wonders.cards.content import load_content
from wonders.engine.record import PlayerSetup
from wonders.env import WondersEnv

EXPECTED = '878edc70fafb927988b4439105d1476138f5ed8c73c6dddc07c4f8cc9131b788'


def _digest() -> str:
    content = load_content('base')
    names = list(dict.fromkeys(w.name for w in content.wonders))
    factories: List[Callable[[], Agent]] = [
        HeuristicAgent,
        GreedyAgent,
        BuilderAgent,
        TableAgent,
        lambda: RandomAgent(random.Random(0)),
    ]
    h = hashlib.sha256()
    for seed in range(25):
        rng = random.Random(seed)
        n = 3 + seed % 5
        setups = [PlayerSetup(f'P{i}', w, 'AB'[(seed + i) % 2]) for i, w in enumerate(rng.sample(names, n))]
        agents = [factories[(seed + i) % len(factories)]() for i in range(n)]
        game, record = play_match(content, setups, agents, rng=rng)
        h.update(record.to_json().encode())
        h.update(json.dumps([game.score(i).as_dict() for i in range(n)]).encode())
    env = WondersEnv(n_players=4, content=content, reward='dense')
    pick = random.Random(1)
    for seed in range(5):
        observations, _ = env.reset(seed)
        while observations:
            actions = {
                s: pick.choice([i for i, ok in enumerate(o['action_mask']) if ok]) for s, o in observations.items()
            }
            observations, rewards, *_ = env.step(actions)
            h.update(json.dumps(sorted(rewards.items())).encode())
        h.update(env.record().to_json().encode())
    return h.hexdigest()


def test_games_are_unchanged():
    assert _digest() == EXPECTED
