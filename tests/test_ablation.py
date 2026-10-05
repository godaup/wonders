"""Tests for leader-decision ablations and per-game bench ranks."""

import random

import pytest

from wonders.agents.ablation import LEADER_MODES, LeaderAblation, leader_ablation
from wonders.agents.base import Agent
from wonders.agents.bench import AGENTS, run_bench
from wonders.agents.match import play_match
from wonders.cards.content import load_content
from wonders.engine.record import PlayerSetup

LEADERS = ('base', 'leaders')


@pytest.mark.parametrize('mode', LEADER_MODES)
def test_ablated_agents_play_leaders_games(mode):
    spec = leader_ablation(AGENTS['board'], mode)
    assert spec.name == f'board[{mode}]'
    result = run_bench([spec, 'board', 'heuristic'], n_games=3, seed=1, expansions=LEADERS)
    assert result.stats[spec.name].games == 3
    assert len(result.ranks) == 3
    for game in result.ranks:
        assert sorted(rank for _name, rank in game) == [1, 2, 3]


def test_ablation_specs_are_picklable():
    spec = leader_ablation(AGENTS['heuristic'], 'random')
    serial = run_bench([spec, 'board', 'greedy'], n_games=4, seed=2, expansions=LEADERS)
    parallel = run_bench([spec, 'board', 'greedy'], n_games=4, seed=2, expansions=LEADERS, processes=2)
    assert serial.ranks == parallel.ranks


class _Spy(Agent):
    """Records which decision kinds reached it."""

    def __init__(self, inner: Agent, seen: list):
        self.inner, self.seen = inner, seen

    def decide(self, obs):
        return self.inner.decide(obs)

    def act(self, obs, options):
        self.seen.append(obs.decision)
        return self.inner.act(obs, options)


@pytest.mark.parametrize(
    'mode, inner_sees_draft, inner_sees_recruit',
    [('random-draft', False, True), ('random', False, False), ('cost', False, False)],
)
def test_ablation_replaces_only_leader_decisions(mode, inner_sees_draft, inner_sees_recruit):
    ablated = LeaderAblation(AGENTS['board'], mode, rng=random.Random(0))
    seen: list = []
    ablated.inner = _Spy(ablated.inner, seen)
    content = load_content(*LEADERS)
    setups = [
        PlayerSetup('A', 'The Pyramids of Giza'),
        PlayerSetup('B', 'The Colossus of Rhodes'),
        PlayerSetup('C', 'The Temple of Artemis in Ephesus'),
    ]
    agents = [ablated, AGENTS['board'].build(random.Random(1)), AGENTS['heuristic'].build(random.Random(2))]
    play_match(content, setups, agents, rng=random.Random(3), expansions=LEADERS)
    assert ('draft' in seen) is inner_sees_draft
    assert ('recruit' in seen) is inner_sees_recruit
    assert seen.count('play') >= 18


def test_cost_mode_keeps_the_costliest_leader():
    content = load_content(*LEADERS)
    leaders = sorted(content.leaders(), key=lambda c: c.name)[:4]
    ablated = LeaderAblation(AGENTS['random'], 'cost', rng=random.Random(0))
    assert ablated.draft_leader(leaders).cost.coins == max(c.cost.coins for c in leaders)


def test_unknown_mode():
    with pytest.raises(ValueError):
        LeaderAblation(AGENTS['board'], 'smart')
