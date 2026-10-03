"""Tests for the vectorized environments."""

import functools
import random

import pytest

from wonders.agents import GreedyAgent, HeuristicAgent, RandomAgent
from wonders.cards.content import load_content
from wonders.env import SingleAgentEnvFactory, SubprocVectorEnv, SyncVectorEnv

FACTORY = SingleAgentEnvFactory(
    opponents=[GreedyAgent, functools.partial(RandomAgent, random.Random(0))],
    learner=0,
)


def first_legal(obs):
    return obs['action_mask'].index(True)


def run(vec, steps, seed=0):
    """Step with a deterministic policy; return everything observed."""
    observations, infos = vec.reset(seed)
    trace = [observations]
    finals = []
    for _ in range(steps):
        observations, rewards, terminated, truncated, infos = vec.step([first_legal(o) for o in observations])
        trace.append((observations, rewards, terminated, truncated))
        finals += [info['final_info'] for info, done in zip(infos, terminated) if done]
    return trace, finals


def test_sync_vector_env_autoresets_and_records():
    vec = SyncVectorEnv([FACTORY, FACTORY])
    assert vec.num_envs == 2
    _trace, finals = run(vec, steps=45)  # > 2 full games per env at ~19 learner steps each
    assert len(finals) >= 4
    content = load_content('base')
    for final in finals:
        replayed = final['record'].replay(content)
        assert [replayed.score(i).total for i in range(3)] == final['scores']


def test_infos_drop_raw_observation_unless_asked():
    _obs, infos = SyncVectorEnv([FACTORY]).reset(0)
    assert 'obs' not in infos[0] and infos[0]['decision'] == 'play'
    _obs, infos = SyncVectorEnv([FACTORY], keep_obs=True).reset(0)
    assert infos[0]['obs'].seat == 0


def test_subproc_matches_sync_exactly():
    sync_trace, sync_finals = run(SyncVectorEnv([FACTORY] * 3), steps=25, seed=11)
    # Platform-default start method (spawn on Windows/macOS); groups are uneven: 2 + 1.
    with SubprocVectorEnv([FACTORY] * 3, envs_per_worker=2) as vec:
        proc_trace, proc_finals = run(vec, steps=25, seed=11)
    assert proc_trace == sync_trace
    assert [f['scores'] for f in proc_finals] == [f['scores'] for f in sync_finals]


def test_subproc_works_with_spawn():
    factory = SingleAgentEnvFactory(opponents=[HeuristicAgent, GreedyAgent], learner=2, env_kwargs={'reward': 'win'})
    with SubprocVectorEnv([factory, factory], context='spawn') as vec:
        assert vec.action_space.n > 0
        observations, _infos = vec.reset(3)
        _o, rewards, terminated, _t, _i = vec.step([first_legal(o) for o in observations])
        assert len(rewards) == 2 and terminated == [False, False]


def test_vector_env_rejects_wrong_action_count():
    vec = SyncVectorEnv([FACTORY, FACTORY])
    vec.reset(0)
    with pytest.raises(ValueError):
        vec.step([0])
