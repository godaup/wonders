"""Tests for the game notation: record, serialize, and deterministic replay."""

import random

from wonders.agents.baselines import GreedyAgent, RandomAgent
from wonders.agents.match import play_match
from wonders.cards.content import load_content
from wonders.engine.record import GameRecord, PlayerSetup

SETUPS = [
    PlayerSetup('Alice', 'The Pyramids of Giza', 'A'),
    PlayerSetup('Bob', 'The Colossus of Rhodes', 'A'),
    PlayerSetup('Cara', 'The Temple of Artemis in Ephesus', 'A'),
]


def _match(seed=0):
    content = load_content('base')
    agents = [GreedyAgent(), RandomAgent(random.Random(seed + 1)), GreedyAgent()]
    game, record = play_match(content, SETUPS, agents, rng=random.Random(seed))
    return content, game, record


def test_replay_reproduces_scores():
    content, game, record = _match()
    replayed = record.replay(content)
    assert [replayed.score(i).total for i in range(3)] == [game.score(i).total for i in range(3)]
    assert [p.coins for p in replayed.players] == [p.coins for p in game.players]
    assert [sum(p.tokens) for p in replayed.players] == [sum(p.tokens) for p in game.players]


def test_record_json_round_trip_then_replay():
    content, game, record = _match(seed=3)
    restored = GameRecord.from_json(record.to_json())
    replayed = restored.replay(content)
    assert [replayed.score(i).total for i in range(3)] == [game.score(i).total for i in range(3)]


def test_record_structure_and_text():
    _content, _game, record = _match()
    assert set(record.moves) == {1, 2, 3}
    # every recorded turn has one decision per seat
    for age in (1, 2, 3):
        for turn in record.moves[age]:
            assert len(turn) == 3
            assert all(code in ('B', 'W', 'D') for code, _name in turn)
    text = record.text()
    assert 'Age 1:' in text and 'Alice' in text


def test_replay_is_engine_independent_of_rng():
    # Replay must not consult any RNG: two replays are identical.
    content, _game, record = _match(seed=5)
    a = record.replay(content)
    b = record.replay(content)
    assert [a.score(i).total for i in range(3)] == [b.score(i).total for i in range(3)]
