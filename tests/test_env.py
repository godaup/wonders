"""Tests for the fixed action space and the step-based environments."""

import random

import pytest

from wonders.agents import BuilderAgent, GreedyAgent, HeuristicAgent, RandomAgent
from wonders.agents.encoders import CardEncoder
from wonders.cards.content import load_content
from wonders.engine.actions import Action, ActionKind
from wonders.engine.record import PlayerSetup
from wonders.env import KINDS, ActionSpace, SingleAgentEnv, WondersEnv


def random_index(mask, rng):
    return rng.choice([i for i, ok in enumerate(mask) if ok])


def play_random(env, seed):
    rng = random.Random(seed)
    observations, _infos = env.reset(seed)
    steps, total = 0, {seat: 0.0 for seat in env.possible_agents}
    while observations:
        actions = {seat: random_index(o['action_mask'], rng) for seat, o in observations.items()}
        observations, rewards, terms, truncs, _infos = env.step(actions)
        for seat, r in rewards.items():
            total[seat] += r
        steps += 1
    assert all(terms.values()) and not any(truncs.values())
    return steps, total


def test_action_space_round_trips_every_legal_action():
    content = load_content('base')
    space = ActionSpace.from_content(content)
    assert space.n == len(space.card_names) * len(KINDS)
    card = content.card('Lumber Yard')
    for kind in KINDS:
        action = Action(kind, card)
        idx = space.index(action)
        assert space.describe(idx) == ('Lumber Yard', kind)
        assert space.to_action(idx, [action]) is action
    mask = space.mask([Action.discard(card)])
    assert sum(mask) == 1 and mask[space.index(Action.discard(card))]
    with pytest.raises(ValueError):
        space.to_action(space.index(Action.build(card)), [Action.discard(card)])


def test_env_plays_a_full_game_with_masks():
    env = WondersEnv(n_players=4)
    observations, infos = env.reset(seed=1)
    assert sorted(observations) == [0, 1, 2, 3]
    first = observations[0]
    assert len(first['observation']) == env.encoder.size  # type: ignore[attr-defined]
    assert len(first['action_mask']) == env.action_space.n
    assert infos[0]['decision'] == 'play'
    steps, total = play_random(env, seed=1)
    assert steps >= 18  # 3 Ages x 6 regular turns, plus any extra decisions
    assert env.done and env.agents == []
    scores = env.scores()
    for seat in env.possible_agents:
        assert total[seat] == scores[seat] - max(s for j, s in enumerate(scores) if j != seat)


def test_env_record_replays_to_the_same_scores():
    env = WondersEnv(n_players=3, sides='B')
    play_random(env, seed=7)
    replayed = env.record().replay(env.content)
    assert [replayed.score(i).total for i in range(3)] == env.scores()


def test_env_is_deterministic_per_seed():
    a, b = WondersEnv(n_players=3), WondersEnv(n_players=3)
    play_random(a, seed=3)
    play_random(b, seed=3)
    assert a.record().to_json() == b.record().to_json()


def test_env_handles_power_decisions():
    setups = [
        PlayerSetup('Bab', 'The Hanging Gardens of Babylon', 'B'),
        PlayerSetup('Hal', 'The Mausoleum of Halicarnassus', 'B'),
        PlayerSetup('Oly', 'The Statue of Zeus in Olympia', 'A'),
    ]
    env = WondersEnv(n_players=3, setups=setups)
    agents = [BuilderAgent(), BuilderAgent(), HeuristicAgent()]
    decisions = set()
    for seed in range(6):
        observations, infos = env.reset(seed)
        while observations:
            decisions |= {info['decision'] for info in infos.values()}
            if len(observations) == 1:  # extra decisions are single-seat
                assert infos[next(iter(infos))]['decision'] != 'play'
            actions = {s: env.action_space.index(agents[s].decide(infos[s]['obs'])) for s in observations}
            observations, _r, _t, _tr, infos = env.step(actions)
        replayed = env.record().replay(env.content)
        assert [replayed.score(i).total for i in range(3)] == env.scores()
    assert decisions == {'play', 'last_card', 'discard_build'}


@pytest.mark.parametrize('mode', ['win', 'score', 'dense'])
def test_reward_modes(mode):
    env = WondersEnv(n_players=3, reward=mode)
    _steps, total = play_random(env, seed=2)
    scores = env.scores()
    if mode == 'score':
        assert [total[i] for i in range(3)] == scores
    elif mode == 'win':
        assert sum(total.values()) == pytest.approx(0.0) and total[env.game.winner()] == 1.0  # type: ignore[union-attr]
    else:
        for i in range(3):
            assert total[i] == pytest.approx(scores[i] - sum(s for j, s in enumerate(scores) if j != i) / 2)


def test_env_rejects_bad_input():
    with pytest.raises(NotImplementedError):
        WondersEnv(expansions=('leaders',))
    with pytest.raises(ValueError):
        WondersEnv(reward='elo')
    env = WondersEnv(n_players=3)
    with pytest.raises(RuntimeError):
        env.step({0: 0})
    observations, _ = env.reset(0)
    with pytest.raises(ValueError):
        env.step({0: 0})  # missing seats
    illegal = next(i for i, ok in enumerate(observations[0]['action_mask']) if not ok)
    with pytest.raises(ValueError):
        env.step({0: illegal, 1: 0, 2: 0})


def test_single_agent_env_runs_against_baselines():
    env = SingleAgentEnv([GreedyAgent(), RandomAgent(random.Random(0))], learner=1)
    rng = random.Random(0)
    obs, info = env.reset(seed=5)
    assert info['obs'].seat == 1
    total, done = 0.0, False
    while not done:
        obs, reward, done, _trunc, info = env.step(random_index(obs['action_mask'], rng))
        total += reward
    scores = info['scores']
    assert total == scores[1] - max(scores[0], scores[2])
    assert info['record'].replay(env.env.content).score(1).total == scores[1]


def test_card_encoder_marks_hand_and_tableau():
    env = WondersEnv(n_players=3)
    observations, infos = env.reset(seed=0)
    obs = infos[0]['obs']
    enc = CardEncoder(env.action_space.card_names)
    vec = enc.encode(obs)
    assert len(vec) == enc.size
    hand_offset = enc.size - 2 * len(enc.vocab) - 11  # tail: hand, hand costs, 11 context features
    hand = vec[hand_offset : hand_offset + len(enc.vocab)]
    assert {enc.vocab[i] for i, v in enumerate(hand) if v} == {c.name for c in obs.hand}
    assert ActionKind.BUILD in {
        env.action_space.describe(i)[1] for i, ok in enumerate(observations[0]['action_mask']) if ok
    }
