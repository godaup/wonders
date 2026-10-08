"""Tests for rebuilding games from one seat's observation (wonders.agents.worlds)."""

import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pytest

from wonders.agents.heuristic import HeuristicAgent
from wonders.agents.worlds import sample_world
from wonders.engine.actions import Action, ActionKind, DecisionKind
from wonders.engine.observation import Observation, public_table
from wonders.env.environment import WondersEnv

PUBLIC = ('coins', 'cards', 'stages_built', 'tokens', 'debt', 'diplomacy', 'score_breakdown', 'production')
COMBOS = [('base',), ('base', 'leaders'), ('base', 'cities'), ('base', 'leaders', 'cities')]


def _play_to(env: WondersEnv, age: int, turn: int, seed: int) -> Tuple[Optional[Dict[str, Any]], list]:
    """Play heuristic agents; snapshot seat 0's view and the true hidden state at ``(age, turn)``.

    Returns the snapshot and every later decision as ``(kind, {seat: (ActionKind, card)})``.
    """
    env.reset(seed=seed)
    agents = [HeuristicAgent() for _ in range(env.n_players)]
    snap: Optional[Dict[str, Any]] = None
    later: List[Tuple[DecisionKind, Dict[int, Tuple[ActionKind, str]]]] = []
    while not env.done:
        point = env._point
        assert point is not None and env.game is not None
        if snap is None and point.kind is DecisionKind.PLAY and (point.age, point.turn) == (age, turn):
            game = env.game
            snap = {
                'obs': env.observation(0),
                'hands': [[c.name for c in point.options[i]] for i in range(env.n_players)],
                'pile': [c.name for c in game.discard_pile],
                'leaders': [[c.name for c in p.leaders_hand] for p in game.players],
                'box': [c.name for c in game.leader_box],
                'table': public_table(game),
            }
        indices, chosen = {}, {}
        for seat in env.acting:
            action = agents[seat].act(env.observation(seat), env.options(seat))
            indices[seat] = env.action_space.index(action)
            chosen[seat] = (action.kind, action.card.name)
        if snap is not None:
            later.append((point.kind, chosen))
        env.step(indices)
    return snap, later


@pytest.mark.parametrize('expansions', COMBOS, ids=['+'.join(c[1:]) or 'base' for c in COMBOS])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_rebuilt_world_matches_the_game(expansions, seed):
    """Public state is reproduced exactly; with the true hidden cards, the rest plays out identically."""
    n = 3 + 2 * seed
    env = WondersEnv(n_players=n, expansions=expansions)
    snap, later = _play_to(env, 3, 1 + seed, seed)
    assert snap is not None
    world = sample_world(snap['obs'], random.Random(seed))
    for true, rebuilt in zip(snap['table'], public_table(world.game)):
        for key in PUBLIC:
            assert getattr(true, key) == getattr(rebuilt, key), key
    assert [len(h) for h in world.hands] == [len(h) for h in snap['hands']]
    assert [c.name for c in world.hands[0]] == snap['hands'][0]
    assert len(world.game.discard_pile) == len(snap['pile'])

    index = world.index
    world.hands[:] = [[index.card(name, 3) for name in hand] for hand in snap['hands']]
    world.game.discard_pile = [index.card(name) for name in snap['pile']]
    for player, names in zip(world.game.players, snap['leaders']):
        player.leaders_hand = [index.card(name) for name in names]
    world.game.leader_box = [index.card(name) for name in snap['box']]
    steps = world.steps()
    point = next(steps)
    for kind, chosen in later:
        assert point.kind is kind and set(point.options) == set(chosen)
        actions = {}
        for seat, (action_kind, name) in chosen.items():
            actions[seat] = Action(action_kind, next(c for c in point.options[seat] if c.name == name))
        world.record(point, actions)
        try:
            point = steps.send(actions)
        except StopIteration:
            break
    assert [world.game.score(i).total for i in range(n)] == env.scores()


def _hidden_state(obs: Observation, seed: int) -> list:
    world = sample_world(obs, random.Random(seed))
    return [[c.name for c in hand] for hand in world.hands] + [[c.name for c in world.game.discard_pile]]


def test_sampled_worlds_only_depend_on_the_observation():
    """Worlds come from the observation alone: they differ by seed, never by the true hidden cards."""
    env = WondersEnv(n_players=4, expansions=('base', 'leaders', 'cities'))
    snap, _later = _play_to(env, 2, 3, seed=5)
    assert snap is not None
    obs = snap['obs']
    assert _hidden_state(obs, 1) == _hidden_state(obs, 1)
    assert _hidden_state(obs, 1) != _hidden_state(obs, 2)
    world = sample_world(obs, random.Random(1))
    opponents = [card for hand in world.hands[1:] for card in hand]
    assert all(c.age == 2 for c in opponents), [(c.name, c.age) for c in opponents]
    # Cards the seat saw earlier this Age and nobody built since sit in a hidden place:
    # an opponent's hand, or an opponent's hidden discard / wonder play (sampled into the history).
    built = Counter(e.card for e in obs.history if e.age == 2 and e.card and e.kind in ('build', 'free'))
    hidden_plays = Counter(
        move[1]
        for age, _turn, moves in world.observations.history
        if age == 2
        for seat, move in enumerate(moves)
        if move is not None and seat != 0 and move[0] in (ActionKind.DISCARD, ActionKind.WONDER)
    )
    assert Counter(obs.seen_cards) - built <= Counter(c.name for c in opponents) + hidden_plays


def test_worlds_need_a_regular_turn():
    env = WondersEnv(n_players=3, expansions=('base', 'leaders'))
    env.reset(seed=0)
    obs = env.observation(0)
    assert obs.decision == 'draft'
    with pytest.raises(ValueError):
        sample_world(obs, random.Random(0))
