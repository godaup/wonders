"""Tests for the data-driven leader values and the scripted agents' leader decisions."""

import random

import pytest

from wonders.agents import HeuristicAgent
from wonders.agents.bench import AGENTS, run_bench
from wonders.agents.board import BoardAwareAgent
from wonders.agents.leader_values import LEADER_VALUES, estimate_leader_values
from wonders.cards.content import load_content
from wonders.engine.actions import Action

LEADERS = ('base', 'leaders')


@pytest.fixture(scope='module')
def leaders():
    return {c.name: c for c in load_content(*LEADERS).leaders()}


def test_every_leader_has_a_value(leaders):
    assert set(leaders) == set(LEADER_VALUES)


@pytest.mark.parametrize('agent_cls', [HeuristicAgent, BoardAwareAgent])
def test_draft_keeps_the_most_valuable_leader(agent_cls, leaders):
    options = [leaders[n] for n in ('Imhotep', 'Midas', 'Sappho', 'Caesar')]
    best = max(options, key=lambda c: LEADER_VALUES[c.name]).name
    assert agent_cls().draft_leader(options).name == best


class _Obs:
    """The two Observation fields recruit() reads."""

    def __init__(self, legal, next_wonder_stage=None):
        self.legal_actions = legal
        self.next_wonder_stage = next_wonder_stage


def _recruit(agent, legal) -> Action:
    obs = _Obs(legal)
    action = agent.recruit([a.card for a in legal], obs)  # type: ignore[arg-type]
    assert action is not None
    return action


def test_recruit_plays_a_good_leader_and_gives_up_the_worst(leaders):
    agent = HeuristicAgent()
    good, bad = leaders['Caesar'], leaders['Imhotep']
    legal = [Action.build(good), Action.build(bad), Action.discard(good), Action.discard(bad)]
    assert _recruit(agent, legal) == Action.build(good)
    # Only bad leaders and no stage: discard the least valuable one for 3 coins.
    worse = leaders['Aristotle']
    legal = [Action.build(bad), Action.build(worse), Action.discard(bad), Action.discard(worse)]
    assert _recruit(agent, legal) == Action.discard(bad)  # Imhotep (-2.11) is worth less than Aristotle


def test_rome_and_courtesans_pick_the_best_leader(leaders):
    agent = HeuristicAgent()
    legal = [Action.build(leaders[n]) for n in ('Sappho', 'Amytis', 'Nero')]

    class Obs(_Obs):
        decision = 'copy_leader'
        hand: list = []

    obs = Obs(legal)
    assert agent.decide(obs).card.name == 'Amytis'  # type: ignore[arg-type]


def test_scripted_agents_play_full_leaders_games():
    result = run_bench(['heuristic', 'board', 'board-focus', 'greedy'], n_games=4, seed=1, expansions=LEADERS)
    assert sum(s.games for s in result.stats.values()) == 16


def test_estimate_leader_values_smoke():
    values = estimate_leader_values(games=2, tables=((LEADERS, 3),), seed=0)
    assert values and all(set(v) == {'value', 'se', 'recruited'} for v in values.values())
    assert all(v['recruited'] >= 1 for v in values.values())


def test_leader_ablated_opponents_by_name():
    pytest.importorskip('torch')
    from wonders.ml.ppo import opponent_spec

    spec = opponent_spec('board[cost]')
    assert spec.name == 'board[cost]'
    agent = spec.build(random.Random(0))
    assert agent.name == 'board[cost]'
    assert opponent_spec('board') is AGENTS['board']
    with pytest.raises(ValueError):
        opponent_spec('board[smart]').build(random.Random(0))
