"""Tests for the PPO opponent mix: tiers, external anchors and long-lived self-anchors."""

import os
import random
from collections import Counter

import pytest

torch = pytest.importorskip('torch')

from wonders.env.vocab import universal_vocab  # noqa: E402
from wonders.ml.imitation import new_model  # noqa: E402
from wonders.ml.ppo import PPOConfig, draw_opponents, train_ppo  # noqa: E402
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
