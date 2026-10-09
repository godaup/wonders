"""Tests for the game notation: record, serialize, and deterministic replay."""

import random

import pytest

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


class _Logger:
    """Wraps an agent and logs the observation and move of every decision it makes."""

    def __init__(self, agent):
        self.agent = agent
        self.name = agent.name
        self.log = []

    def act(self, obs, options):
        action = self.agent.act(obs, options)
        self.log.append((obs.decision, obs.to_dict(), action))
        return action

    def decide(self, obs):
        return self.act(obs, [])


@pytest.mark.parametrize('seat', [0, 1, 2, 3])
def test_replay_observations_reproduce_what_the_seat_saw(seat):
    import random

    from wonders.agents import HeuristicAgent
    from wonders.agents.match import play_match
    from wonders.cards.content import load_content
    from wonders.engine.record import PlayerSetup

    expansions = ('base', 'leaders', 'cities')
    content = load_content(*expansions)
    setups = [
        PlayerSetup('A', 'The Pyramids of Giza', 'A'),
        PlayerSetup('B', 'The Colossus of Rhodes', 'B'),
        PlayerSetup('C', 'The Mausoleum of Halicarnassus', 'A'),
        PlayerSetup('D', 'The Hanging Gardens of Babylon', 'B'),
    ]
    agents = [_Logger(HeuristicAgent()) for _ in setups]
    _game, record = play_match(content, setups, agents, random.Random(seat), expansions)
    played = [(d, o) for d, o, _a in agents[seat].log if d != 'draft']  # the draft is not part of a record
    replayed = list(record.replay_observations(content, seat))
    assert len(replayed) == len(played)
    for (decision, logged), (_point, obs, action) in zip(played, replayed):
        assert obs.decision == decision
        assert obs.to_dict() == logged
    moves = [a for d, _o, a in agents[seat].log if d != 'draft']
    assert [(a.kind, a.card.name) for _p, _o, a in replayed] == [(a.kind, a.card.name) for a in moves]


def test_branch_with_the_recorded_move_reproduces_the_game():
    import random

    from wonders.agents import HeuristicAgent
    from wonders.engine.actions import ActionKind

    content = load_content('base')
    setups = [
        PlayerSetup(n, w, 'A')
        for n, w in zip(
            'ABCD',
            [
                'The Pyramids of Giza',
                'The Colossus of Rhodes',
                'The Temple of Artemis in Ephesus',
                'The Lighthouse of Alexandria',
            ],
        )
    ]
    agents = [HeuristicAgent() for _ in setups]
    game, record = play_match(content, setups, agents, random.Random(2))
    finals = [game.score(i).total for i in range(4)]
    decisions = list(record.replay_observations(content, 1))

    def policy(_game, i, obs, options):
        return agents[i].act(obs, options)

    same = record.branch(content, 1, 7, decisions[7][2], policy)
    assert [same.score(i).total for i in range(4)] == finals
    point, obs, played = decisions[7]
    other = next(a for a in obs.legal_actions if (a.kind, a.card.name) != (played.kind, played.card.name))
    branched = record.branch(content, 1, 7, other, policy)
    mine = [c.name for c in branched.players[1].cards] + ['#%d' % branched.players[1].wonder_stages_built]
    if other.kind is ActionKind.BUILD:
        assert other.card.name in mine
    reshuffled = record.branch(content, 1, 7, other, policy, rng=random.Random(9))
    assert reshuffled.phase.value == branched.phase.value  # later Ages dealt differently, still a full game
    with pytest.raises(ValueError):
        record.branch(content, 1, 99, other, policy)
