"""Tests for the rank-shaped reward and the PPO learning-rate schedule."""

import random

import pytest

from wonders.agents.bench import AGENTS
from wonders.env import WondersEnv


def _play(reward: str, seed: int = 0, n: int = 4):
    env = WondersEnv(n_players=n, expansions=('base', 'leaders'), reward=reward, rank_bonus=10.0)
    env.encode_seats = set()
    env.reset(seed)
    agents = [AGENTS['heuristic'].build(random.Random(i)) for i in range(n)]
    totals = [0.0] * n
    while not env.done:
        actions = {s: env.action_space.index(agents[s].act(env.observation(s), env.options(s))) for s in env.acting}
        _obs, rewards, _term, _trunc, _info = env.step(actions)
        for seat, r in rewards.items():
            totals[seat] += r
    return env, totals


def test_dense_rank_adds_a_placing_bonus_to_the_margin():
    env, totals = _play('dense_rank')
    n = env.n_players
    scores = env.scores()
    margins = [scores[i] - (sum(scores) - scores[i]) / (n - 1) for i in range(n)]
    bonus = [t - m for t, m in zip(totals, margins)]
    order = env._placing(scores)
    assert bonus[order[0]] == pytest.approx(10.0)
    assert bonus[order[-1]] == pytest.approx(-10.0)
    assert sum(bonus) == pytest.approx(0.0)  # symmetric: the table's bonuses cancel out
    _env, dense = _play('dense')
    assert dense == pytest.approx(margins)  # plain dense is unchanged


def test_learning_rate_schedule():
    torch = pytest.importorskip('torch')  # noqa: F841
    from wonders.ml.ppo import PPOConfig, learning_rate

    constant = PPOConfig(lr=1e-4, iterations=10)
    assert learning_rate(constant, 1) == learning_rate(constant, 10) == 1e-4
    linear = PPOConfig(lr=1e-4, iterations=11, lr_schedule='linear', lr_final=0.1)
    assert learning_rate(linear, 1) == pytest.approx(1e-4)
    assert learning_rate(linear, 6) == pytest.approx(0.55e-4)
    assert learning_rate(linear, 11) == pytest.approx(1e-5)
    with pytest.raises(ValueError):
        learning_rate(PPOConfig(lr_schedule='cosine'), 1)


def test_ppo_runs_with_rank_reward_and_linear_schedule():
    pytest.importorskip('torch')
    from wonders.env.vocab import universal_vocab
    from wonders.ml.imitation import new_model
    from wonders.ml.ppo import PPOConfig, train_ppo

    model, _ = new_model(hidden=16, vocab=universal_vocab())
    config = PPOConfig(
        tables=[(('base', 'leaders'), 3)],
        opponent_pool=('heuristic', 'greedy'),
        universal=True,
        iterations=3,
        lr_schedule='linear',
        reward='dense_rank',
        n_envs=2,
        envs_per_worker=0,
        rollout_steps=12,
        minibatch=16,
        epochs=1,
    )
    history = train_ppo(model, config, log=None)
    assert [h['lr'] for h in history] == sorted((h['lr'] for h in history), reverse=True)
    assert history[-1]['lr'] == pytest.approx(config.lr * config.lr_final)
