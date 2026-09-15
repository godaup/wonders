"""Tests for the agent interface, observations, and encoders."""

import random

from wonders.agents.baselines import GreedyAgent, RandomAgent
from wonders.agents.encoders import DictEncoder, FlatEncoder
from wonders.agents.match import play_match
from wonders.cards.content import build_age_deck, deal_hands, load_content
from wonders.engine.actions import ActionKind
from wonders.engine.game import Game
from wonders.engine.observation import build_observation
from wonders.engine.player import Player
from wonders.engine.record import PlayerSetup

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
