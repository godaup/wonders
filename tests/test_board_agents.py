"""Tests for the wonder-board view in observations and the board-aware agents."""

import random

import pytest

from wonders.agents import BoardAwareAgent, BoardFocusAgent, GreedyAgent, HeuristicAgent
from wonders.agents.match import play_match
from wonders.cards.content import load_content
from wonders.engine.game import Game
from wonders.engine.observation import build_observation
from wonders.engine.player import Player
from wonders.engine.record import PlayerSetup

CONTENT = load_content('base', 'leaders', 'cities')


def observe(wonder, side='A', hand=('Lumber Yard', 'Altar')):
    others = [CONTENT.wonder('The Colossus of Rhodes', 'A'), CONTENT.wonder('The Lighthouse of Alexandria', 'A')]
    g = Game([Player('P0', CONTENT.wonder(wonder, side))] + [Player(f'P{i + 1}', w) for i, w in enumerate(others)])
    g.age = 1
    return g, build_observation(g, 0, [CONTENT.card(n) for n in hand])


def test_observation_exposes_the_whole_board_and_production():
    g, obs = observe('The Hanging Gardens of Babylon', 'B')
    board = obs.wonder_board
    assert board is not None and board.side == 'B' and board.starting_resource == 'bricks'
    assert [s['index'] for s in board.stages] == [1, 2, 3] and not any(s['built'] for s in board.stages)
    assert any(e.get('plays_last_card') for e in board.stages[1]['effects'])  # power keys are summarized
    assert obs.production == [['bricks']]
    assert obs.to_dict()['wonder_board']['name'] == 'The Hanging Gardens of Babylon'
    g.players[0].add_stage(g.players[0].wonder.stages[0])
    assert build_observation(g, 0, []).wonder_board.stages[0]['built']  # type: ignore[union-attr]


def test_rome_board_shows_no_starting_resource_and_passive_power():
    _g, obs = observe('The Colosseum of Rome', 'A')
    assert obs.wonder_board.starting_resource is None  # type: ignore[union-attr]
    assert obs.production == []
    assert obs.wonder_board.effects  # type: ignore[union-attr]


def test_stage_needs_and_production_planning():
    _g, obs = observe('The Pyramids of Giza', 'A')  # stages: 2 stone, 3 wood, 4 stone; produces stone
    agent = BoardAwareAgent()
    needs = agent._stage_needs(obs)
    assert needs['wood'] == 3 and needs['stone'] == 3
    lumber = next(c for c in obs.hand if c.name == 'Lumber Yard')
    plain = BoardAwareAgent({"plan": 0.0})  # same agent without stage planning
    assert agent._card_value(obs, lumber) - plain._card_value(obs, lumber) == pytest.approx(agent.w['plan'])


def test_board_powers_are_valued():
    _g, obs = observe('The Hanging Gardens of Babylon', 'B')
    stage_two = obs.wonder_board.stages[1]['effects']  # type: ignore[union-attr]
    assert BoardAwareAgent()._power_value(obs, stage_two) > 0
    _g, olympia = observe('The Statue of Zeus in Olympia', 'A')
    early = BoardAwareAgent()._power_value(olympia, olympia.wonder_board.stages[1]['effects'])  # type: ignore
    olympia.age = 3
    late = BoardAwareAgent()._power_value(olympia, olympia.wonder_board.stages[1]['effects'])  # type: ignore
    assert early > late > 0  # a once-per-Age power is worth more with more Ages left


def test_board_colors_follow_the_board():
    _g, rhodes = observe('The Colossus of Rhodes', 'A')
    assert 'red' in BoardFocusAgent.board_colors(rhodes)
    _g, babylon = observe('The Hanging Gardens of Babylon', 'A')
    assert 'green' in BoardFocusAgent.board_colors(babylon)


@pytest.mark.parametrize('expansions', [('base',), ('base', 'leaders', 'cities')])
def test_board_agents_play_legal_games(expansions):
    content = load_content(*expansions)
    names = list(dict.fromkeys(w.name for w in content.wonders))
    for seed in range(4):
        rng = random.Random(seed)
        setups = [PlayerSetup(f'P{i}', w, rng.choice('AB')) for i, w in enumerate(rng.sample(names, 4))]
        agents = [BoardAwareAgent(), BoardFocusAgent(), HeuristicAgent(), GreedyAgent()]
        game, record = play_match(content, setups, agents, rng=rng, expansions=expansions)
        replayed = record.replay(content)
        assert [replayed.score(i).total for i in range(4)] == [game.score(i).total for i in range(4)]
