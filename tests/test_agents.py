"""Tests for the agent interface, observations, and encoders."""

import random

import pytest

from wonders.agents.baselines import (
    BuilderAgent,
    CivicAgent,
    DiscardAgent,
    EconomistAgent,
    GreedyAgent,
    MilitaristAgent,
    RandomAgent,
    ScientistAgent,
)
from wonders.agents.encoders import DictEncoder, FlatEncoder
from wonders.agents.heuristic import HeuristicAgent
from wonders.agents.match import play_match
from wonders.agents.strategies import BalancedAgent, BlockerAgent, OpportunistAgent, TableAgent
from wonders.cards.content import build_age_deck, deal_hands, load_content
from wonders.engine.actions import ActionKind
from wonders.engine.game import Game
from wonders.engine.observation import build_observation
from wonders.engine.player import Player
from wonders.engine.record import PlayerSetup

AGENT_FACTORIES = [
    GreedyAgent,
    MilitaristAgent,
    ScientistAgent,
    CivicAgent,
    EconomistAgent,
    BuilderAgent,
    HeuristicAgent,
    DiscardAgent,
    OpportunistAgent,
    BlockerAgent,
    TableAgent,
    BalancedAgent,
    lambda: RandomAgent(random.Random(3)),
]

SETUPS = [
    PlayerSetup('Alice', 'The Pyramids of Giza', 'A'),
    PlayerSetup('Bob', 'The Colossus of Rhodes', 'A'),
    PlayerSetup('Cara', 'The Temple of Artemis in Ephesus', 'A'),
]


def _fresh_observation(seed=0):
    content = load_content('base')
    game = Game([Player(s.name, content.wonder(s.wonder, s.side)) for s in SETUPS])
    game.age = 1
    game.phase = game.phase.AGE_I
    deck = build_age_deck(content, 1, 3, rng=random.Random(seed))
    hands = deal_hands(deck, 3)
    return game, build_observation(game, 0, hands[0])


def test_observation_has_full_state():
    game, obs = _fresh_observation()
    assert obs.seat == 0 and obs.n_players == 3
    assert len(obs.table) == 3
    assert obs.me.name == 'Alice'
    assert obs.left.name == 'Bob' and obs.right.name == 'Cara'
    # discard is always available, so there is at least one legal action per hand card
    assert len(obs.legal_actions) >= len(obs.hand)
    assert {a.kind for a in obs.legal_actions} <= {ActionKind.BUILD, ActionKind.WONDER, ActionKind.DISCARD}


def test_agents_return_legal_actions():
    _game, obs = _fresh_observation()
    for agent in (GreedyAgent(), RandomAgent(random.Random(1))):
        action = agent.decide(obs)
        assert action in obs.legal_actions


def test_dict_encoder_is_json_serializable():
    import json

    _game, obs = _fresh_observation()
    encoded = DictEncoder().encode(obs)
    json.dumps(encoded)  # must not raise
    assert encoded['seat'] == 0
    assert 'hand' in encoded and 'table' in encoded


def test_flat_encoder_is_fixed_length():
    _game, obs = _fresh_observation()
    plain = FlatEncoder(include_history=False)
    with_history = FlatEncoder(include_history=True)
    v1 = plain.encode(obs)
    v2 = plain.encode(obs)
    assert len(v1) == len(v2) and all(isinstance(x, float) for x in v1)
    # history adds 3 counts per player
    assert len(with_history.encode(obs)) == len(v1) + 3 * obs.n_players


def test_history_is_filtered_for_opponents():
    # Run a match, then inspect an observation's history filtering by replaying part.
    content = load_content('base')
    agents = [GreedyAgent(), GreedyAgent(), GreedyAgent()]
    captured = {}

    # Wrap one agent to capture the observation it sees mid-game.
    class Spy(GreedyAgent):
        def decide(self, obs):
            if obs.history and 'spied' not in captured:
                captured['spied'] = obs
            return super().decide(obs)

    agents[0] = Spy()
    play_match(content, SETUPS, agents, rng=random.Random(2))
    obs = captured.get('spied')
    assert obs is not None
    for move in obs.history:
        if move.seat != obs.seat and move.kind in ('wonder', 'discard'):
            assert move.card is None  # opponents' buried/discarded cards are hidden
        if move.kind == 'build':
            assert move.card is not None  # builds are public


def test_match_runs_and_is_reproducible():
    content = load_content('base')

    def run():
        agents = [GreedyAgent(), RandomAgent(random.Random(9)), GreedyAgent()]
        return play_match(content, SETUPS, agents, rng=random.Random(4))

    g1, r1 = run()
    g2, r2 = run()
    assert r1.to_json() == r2.to_json()
    assert [g1.score(i).total for i in range(3)] == [g2.score(i).total for i in range(3)]


@pytest.mark.parametrize('make', AGENT_FACTORIES)
def test_every_agent_returns_a_legal_action(make):
    _game, obs = _fresh_observation()
    assert make().decide(obs) in obs.legal_actions


def test_next_wonder_stage_exposed():
    _game, obs = _fresh_observation()
    assert obs.next_wonder_stage is not None
    assert obs.next_wonder_stage.index == 1
    assert obs.next_wonder_stage.effects  # Giza stage 1 grants 3 VP


def test_heuristic_prefers_a_high_value_build():
    content = load_content('base')
    game = Game([Player(s.name, content.wonder(s.wonder, s.side)) for s in SETUPS])
    game.age = 1
    game.phase = game.phase.AGE_I
    # Giza produces Stone, so Baths (1 stone -> 3 VP) is free to build here.
    hand = [content.card('Baths'), content.card('Lumber Yard')]
    obs = build_observation(game, 0, hand)
    action = HeuristicAgent().decide(obs)
    assert action.kind is ActionKind.BUILD and action.card.name == 'Baths'


def test_diverse_agents_play_a_full_match():
    content = load_content('base')
    agents = [HeuristicAgent(), MilitaristAgent(), ScientistAgent()]
    game, _record = play_match(content, SETUPS, agents, rng=random.Random(1))
    assert game.phase.value == 'finished'
    assert all(p.coins >= 0 for p in game.players)


def _obs_with(hand_names, own_cards=(), seat_wood=False):
    content = load_content('base')
    game = Game([Player(s.name, content.wonder(s.wonder, s.side)) for s in SETUPS])
    game.age = 1
    game.phase = game.phase.AGE_I
    if seat_wood:
        game.players[0].add_card(content.card('Lumber Yard'))  # so red/1-wood cards build
    for name in own_cards:
        game.players[0].add_card(content.card(name))
    hand = [content.card(n) for n in hand_names]
    return content, game, build_observation(game, 0, hand)


def test_balanced_agent_fills_the_scarcest_colour():
    # Own city skews blue; between a buildable red and blue card, pick the red.
    _c, _g, obs = _obs_with(['Stockade', 'Pawnshop'], own_cards=['Altar', 'Theater'], seat_wood=True)
    action = BalancedAgent().decide(obs)
    assert action.kind is ActionKind.BUILD and action.card.name == 'Stockade'


def test_table_agent_follows_its_value_table():
    _c, _g, obs = _obs_with(['Theater', 'Pawnshop'])  # both blue and free
    agent = TableAgent(values={'Pawnshop': 10.0, 'Theater': 1.0})
    assert agent.decide(obs).card.name == 'Pawnshop'


def test_blocker_targets_the_neighbours_most_wanted_card():
    # Left neighbour (target in Age I) is science-focused; blocker goes for the science card.
    _c, game, _obs = _obs_with(['Workshop', 'Altar'])
    game.players[1].add_card(load_content('base').card('Scriptorium'))  # left neighbour: science
    obs = build_observation(game, 0, [load_content('base').card(n) for n in ('Workshop', 'Altar')])
    action = BlockerAgent().decide(obs)
    assert action.card.name == 'Workshop'  # taken or trashed, but not left for them


def test_opportunist_switches_focus_over_turns():
    agent = OpportunistAgent(random.Random(0), switch_every=1)
    _game, obs = _fresh_observation()
    focuses = []
    for _ in range(4):
        agent.decide(obs)
        focuses.append(agent._focus)
    assert len(set(focuses)) > 1
