"""Tests for the PPO opponent mix: tiers, external anchors and long-lived self-anchors."""

import os
import random
from collections import Counter

import pytest

torch = pytest.importorskip('torch')

from wonders.env.vocab import universal_vocab  # noqa: E402
from wonders.ml.imitation import new_model  # noqa: E402
from wonders.ml.ppo import PPOConfig, draw_opponents, opponent_device, opponent_spec, train_ppo  # noqa: E402
from wonders.ml.sweep import ppo_config  # noqa: E402

TINY = dict(n_envs=2, envs_per_worker=0, rollout_steps=12, minibatch=16, epochs=1)
POOL = ['heuristic', 'heuristic', 'board', 'neural:recent-1.pt', 'neural:recent-2.pt']
ANCHORS = ['neural:best.pt']


def _tiers(names):
    return Counter('anchor' if n in ANCHORS else ('recent' if n.startswith('neural:') else 'scripted') for n in names)


def test_legacy_draw_is_uniform_over_the_pool():
    names = draw_opponents(PPOConfig(), random.Random(0), POOL, ANCHORS, epoch=0, k=2000)
    assert set(names) <= set(POOL)  # anchors only exist with tier weights
    assert 0.35 < names.count('heuristic') / 2000 < 0.45  # listed twice of five


def test_tier_weights_set_the_seat_shares():
    config = PPOConfig(tier_weights=(0.3, 0.4, 0.3))
    shares = _tiers(draw_opponents(config, random.Random(1), POOL, ANCHORS, epoch=0, k=4000))
    assert abs(shares['scripted'] / 4000 - 0.3) < 0.03
    assert abs(shares['recent'] / 4000 - 0.4) < 0.03
    assert abs(shares['anchor'] / 4000 - 0.3) < 0.03
    scripted = [n for n in draw_opponents(config, random.Random(2), POOL, ANCHORS, 0, 4000) if not n.startswith('n')]
    assert 0.6 < scripted.count('heuristic') / len(scripted) < 0.73  # repeated names weigh more (2 of 3)


def test_anchor_tier_waits_for_anchors_from_and_empty_tiers_renormalise():
    config = PPOConfig(tier_weights=(0.5, 0.0, 0.5), anchors_from=100)
    early = _tiers(draw_opponents(config, random.Random(3), POOL, ANCHORS, epoch=50, k=500))
    late = _tiers(draw_opponents(config, random.Random(3), POOL, ANCHORS, epoch=100, k=500))
    assert early == Counter({'scripted': 500})
    assert late['anchor'] > 150 and late['recent'] == 0
    with pytest.raises(ValueError):
        draw_opponents(PPOConfig(tier_weights=(0, 1, 0)), random.Random(0), ['heuristic'], [], 0, 1)


def test_league_keeps_self_anchors_and_resumes(tmp_path):
    vocab = universal_vocab()
    model, _ = new_model(hidden=16, vocab=vocab)
    anchor_path = str(tmp_path / 'anchor.pt')
    from wonders.ml import save_checkpoint

    save_checkpoint(anchor_path, new_model(hidden=16, vocab=vocab)[0], vocab.card_names, **vocab.meta())
    config = PPOConfig(
        tables=[(('base',), 3), (('base', 'leaders'), 4)],
        opponent_pool=('heuristic', 'greedy'),
        universal=True,
        iterations=4,
        snapshot_every=1,
        league=True,
        league_size=1,
        league_dir=str(tmp_path / 'snaps'),
        tier_weights=(0.4, 0.3, 0.3),
        anchors=(f'neural:{anchor_path}',),
        anchors_from=2,
        league_anchor_every=2,
        league_anchor_max=1,
        **TINY,
    )
    states = []
    history = train_ppo(model, config, log=None, on_snapshot=lambda it, st: states.append(st))
    assert len(history) == 4
    assert states[1]['self_anchors'] == [f"neural:{os.path.join(config.league_dir, 'snapshot-000002.pt')}"]
    assert len(states[-1]['self_anchors']) == 2  # iterations 2 and 4 (league_anchor_max limits use, not memory)
    config.iterations = 5
    assert len(train_ppo(model, config, log=None, state=states[-1])) == 1


def test_sweep_resolves_run_anchors(tmp_path):
    run = {'name': 'x', 'ppo': {'anchors': ['run:best', 'board'], 'tier_weights': [1, 1, 1], 'universal': True}}
    config = ppo_config(run, str(tmp_path / 'x'))
    assert config.anchors == (f"neural:{tmp_path / 'best' / 'final.pt'}", 'board')
    assert config.tier_weights == (1.0, 1.0, 1.0)


# -- opponent inference device ------------------------------------------------------
def _save(path, hidden):
    from wonders.ml import save_checkpoint

    vocab = universal_vocab()
    save_checkpoint(str(path), new_model(hidden=hidden, vocab=vocab)[0], vocab.card_names, **vocab.meta())
    return str(path)


def test_opponent_device_routes_only_large_networks(tmp_path, monkeypatch):
    big, small = _save(tmp_path / 'big.pt', 64), _save(tmp_path / 'small.pt', 16)
    config = PPOConfig(opponent_device='cuda', opponent_gpu_min_hidden=32)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    assert opponent_device(config, f'neural:{big}') == 'cuda'
    assert opponent_device(config, f'neural:{small}') == 'cpu'
    assert opponent_device(config, 'board') == 'cpu'
    assert opponent_device(PPOConfig(), f'neural:{big}') == 'cpu'  # default stays on the CPU
    assert dict(opponent_spec(f'neural:{big}', 'cuda').kwargs)['device'] == 'cuda'
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert opponent_device(config, f'neural:{big}') == 'cpu'  # no CUDA: fall back


def test_agents_share_one_model_per_checkpoint(tmp_path):
    from wonders.ml import NeuralAgent

    path = _save(tmp_path / 'm.pt', 16)
    assert NeuralAgent(path).model is NeuralAgent(path, name='other').model


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_gpu_inference_matches_cpu(tmp_path):
    from wonders.env import WondersEnv
    from wonders.ml import NeuralAgent

    vocab = universal_vocab()
    path = _save(tmp_path / 'g.pt', 32)
    cpu, gpu = NeuralAgent(path), NeuralAgent(path, device='cuda')
    env = WondersEnv(
        n_players=4, expansions=('base', 'leaders'), card_names=vocab.card_names, decisions=vocab.decisions
    )
    env.encode_seats = set()
    env.reset(1)
    decisions = 0
    while not env.done and decisions < 60:
        actions = {}
        for seat in env.acting:
            obs = env.observation(seat)
            choice = cpu.decide(obs)
            assert gpu.decide(obs) == choice
            actions[seat] = env.action_space.index(choice)
            decisions += 1
        env.step(actions)
    assert decisions > 10
