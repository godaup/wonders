"""Tests for the network-guided search agent (skipped unless PyTorch is installed)."""

import random

import pytest

torch = pytest.importorskip('torch')

from wonders.agents import GreedyAgent  # noqa: E402
from wonders.agents.match import play_match  # noqa: E402
from wonders.cards.content import load_content  # noqa: E402
from wonders.engine.record import PlayerSetup  # noqa: E402
from wonders.env.environment import WondersEnv  # noqa: E402
from wonders.ml import NeuralAgent, pretrained_path  # noqa: E402
from wonders.ml.search import SearchAgent  # noqa: E402

SETUPS = [
    PlayerSetup('S', 'The Pyramids of Giza', 'A'),
    PlayerSetup('N', 'The Colossus of Rhodes', 'B'),
    PlayerSetup('G', 'The Hanging Gardens of Babylon', 'B'),
    PlayerSetup('H', 'The Temple of Artemis in Ephesus', 'A'),
]


@pytest.mark.parametrize('depth', [1, None])
def test_search_agent_plays_full_games(depth):
    expansions = ('base', 'leaders', 'cities')
    content = load_content(*expansions)
    search = SearchAgent(pretrained_path(), worlds=2, candidates=2, depth=depth, rng=random.Random(0))
    agents = [search, NeuralAgent(pretrained_path()), GreedyAgent(), GreedyAgent()]
    game, record = play_match(content, SETUPS, agents, random.Random(1), expansions)
    assert 0 <= game.winner() < 4
    assert len(record.moves[3]) == 7  # Cities deals 8 cards: 7 plays per Age


def test_search_evaluates_candidates_on_shared_worlds():
    env = WondersEnv(n_players=3, expansions=('base',))
    env.reset(seed=3)
    obs = env.observation(0)
    search = SearchAgent(pretrained_path(), worlds=3, candidates=3, depth=2, rng=random.Random(0))
    actions = obs.legal_actions[:3]
    values = search.evaluate(obs, actions)
    assert len(values) == 3 and all(isinstance(v, float) for v in values)
    again = SearchAgent(pretrained_path(), worlds=3, candidates=3, depth=2, rng=random.Random(0))
    assert again.evaluate(obs, actions) == values  # seeded: reproducible
    assert search.decide(obs) in obs.legal_actions


def test_search_agent_rejects_bad_settings():
    with pytest.raises(ValueError):
        SearchAgent(pretrained_path(), worlds=0)
