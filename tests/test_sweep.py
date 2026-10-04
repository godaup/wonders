"""Tests for the sweep infrastructure: vocabularies, mixed-table PPO, league, runner, eval."""

import json
import os
import random

import pytest

torch = pytest.importorskip('torch')

from click.testing import CliRunner  # noqa: E402

from wonders import cli  # noqa: E402
from wonders.agents import GreedyAgent, HeuristicAgent  # noqa: E402
from wonders.agents.match import play_match  # noqa: E402
from wonders.cards.content import load_content  # noqa: E402
from wonders.engine.record import PlayerSetup  # noqa: E402
from wonders.env import WondersEnv  # noqa: E402
from wonders.env.vocab import universal_vocab, vocab_for  # noqa: E402
from wonders.ml import NeuralAgent, save_checkpoint  # noqa: E402
from wonders.ml.evaluate import ALL_CELLS, evaluate, parse_cells  # noqa: E402
from wonders.ml.imitation import collect, new_model  # noqa: E402
from wonders.ml.ppo import PPOConfig, train_ppo  # noqa: E402
from wonders.ml.sweep import expand, run_sweep  # noqa: E402

TINY = dict(n_envs=2, envs_per_worker=0, rollout_steps=12, minibatch=16, epochs=1)


def test_vocabularies():
    base, universal = vocab_for(('base',)), universal_vocab()
    assert len(base.card_names) == 75 and universal.covers(base.card_names)
    assert set(vocab_for(('base', 'leaders', 'cities')).card_names) == set(universal.card_names)
    assert 'draft' in universal.decisions and 'draft' not in base.decisions
    assert len(ALL_CELLS) == 20


def test_universal_model_plays_base_and_expansion_games(tmp_path):
    vocab = universal_vocab()
    model, names = new_model(hidden=16, vocab=vocab)
    path = str(tmp_path / 'u.pt')
    save_checkpoint(path, model, names, decisions=list(vocab.decisions))
    for expansions in (('base',), ('base', 'leaders', 'cities')):
        content = load_content(*expansions)
        setups = [
            PlayerSetup('N', 'The Pyramids of Giza'),
            PlayerSetup('H', 'The Colossus of Rhodes'),
            PlayerSetup('G', 'The Temple of Artemis in Ephesus'),
            PlayerSetup('X', 'The Lighthouse of Alexandria'),
        ]
        agents = [NeuralAgent(path), HeuristicAgent(), GreedyAgent(), HeuristicAgent()]
        game, _ = play_match(content, setups, agents, rng=random.Random(0), expansions=expansions)
        assert 0 <= game.winner() < 4
    env = WondersEnv(n_players=3, card_names=vocab.card_names, decisions=vocab.decisions)
    observations, _ = env.reset(0)
    assert len(observations[0]['observation']) == vocab.obs_size
    with pytest.raises(ValueError):
        WondersEnv(expansions=('leaders',), card_names=vocab_for(('base',)).card_names)


def test_collect_records_leader_decisions():
    vocab = universal_vocab()
    data = collect(
        2, tables=[(('base', 'leaders'), ('board', 'board', 'board', 'board'))], teachers=('board',), vocab=vocab
    )
    draft_column = vocab.obs_size - len(vocab.decisions) + vocab.decisions.index('draft')
    assert data.obs.shape[1] == vocab.obs_size
    assert int(data.obs[:, draft_column].sum()) >= 4 * 4 * 2  # every draft pick was recorded


def test_mixed_tables_league_and_resume(tmp_path):
    vocab = universal_vocab()
    model, _ = new_model(hidden=16, vocab=vocab)
    config = PPOConfig(
        tables=[(('base',), 3), (('base', 'leaders'), 4)],
        opponent_pool=('heuristic', 'greedy'),
        universal=True,
        iterations=2,
        snapshot_every=1,
        league=True,
        league_size=1,
        league_dir=str(tmp_path),
        **TINY,
    )
    states = []
    history = train_ppo(model, config, log=None, on_snapshot=lambda it, st: states.append(st))
    assert len(history) == 2 and 'win_rel' in history[-1]
    assert [s['iteration'] for s in states] == [1, 2]
    assert sum(p.startswith('neural:') for p in states[-1]['pool']) == 1  # league_size keeps the newest
    assert len(os.listdir(tmp_path)) == 2
    config.iterations = 3
    resumed = train_ppo(model, config, log=None, state=states[-1])
    assert len(resumed) == 1  # continued at iteration 3
    with pytest.raises(ValueError):  # a base-vocabulary model cannot train on universal tables
        train_ppo(new_model(hidden=16)[0], config, log=None)


def test_sweep_runner_end_to_end_and_resumable(tmp_path):
    spec = {
        'out': str(tmp_path / 'runs'),
        'runs': [
            {
                'name': 'imit',
                'kind': 'imitate',
                'games': 3,
                'epochs': 1,
                'hidden': 16,
                'universal': True,
                'tables': [[['base'], ['board', 'heuristic', 'greedy']]],
                'teachers': ['board'],
            },
            {
                'name': 'ppo',
                'kind': 'ppo',
                'init': 'imit',
                'steps': 24,
                'ppo': {'tables': [[['base'], 3]], 'opponent_pool': ['heuristic'], 'universal': True, **TINY},
                'grid': {'lr': [1e-4, 3e-4]},
                'eval': {'cells': 'base/3p', 'games': 3, 'lineups': ['heuristic']},
            },
        ],
    }
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(spec))
    assert [r['name'] for r in expand(spec['runs'])] == ['imit', 'ppo-lr0.0001', 'ppo-lr0.0003']
    logs = []
    run_sweep(str(path), processes=1, log=logs.append)
    run_dir = tmp_path / 'runs' / 'ppo-lr0.0001'
    for name in ('done', 'final.pt', 'metrics.jsonl', 'eval.json', 'eval.md', 'config.json'):
        assert (run_dir / name).exists(), name
    results = json.loads((run_dir / 'eval.json').read_text())
    assert results[0]['cell'] == 'base/3p' and results[0]['games'] == 3
    logs.clear()
    run_sweep(str(path), processes=1, log=logs.append)
    assert all('skipping' in line for line in logs)


def test_evaluate_skips_cells_outside_the_vocabulary(tmp_path):
    model, names = new_model(hidden=16)  # base vocabulary only
    path = str(tmp_path / 'b.pt')
    save_checkpoint(path, model, names)
    results = evaluate(path, cells=parse_cells('base/3p,leaders/4p'), lineups=('heuristic',), games=3)
    assert results[0]['cell'] == 'base/3p' and 'win_rel' in results[0]
    assert results[1]['cell'] == 'leaders/4p' and results[1]['skipped']


def test_cli_sweep_dry_run_and_eval(tmp_path):
    spec = tmp_path / 's.json'
    spec.write_text(
        json.dumps({'out': str(tmp_path), 'runs': [{'name': 'x', 'kind': 'ppo', 'steps': 10, 'grid': {'lr': [1, 2]}}]})
    )
    runner = CliRunner()
    listed = runner.invoke(cli.main, ['train', 'sweep', str(spec), '--dry-run'])
    assert listed.exit_code == 0 and 'x-lr1' in listed.output and 'x-lr2' in listed.output
    model, names = new_model(hidden=16)
    ckpt = str(tmp_path / 'm.pt')
    save_checkpoint(ckpt, model, names)
    out = runner.invoke(
        cli.main, ['train', 'eval', ckpt, '--cells', 'base/3p', '-n', '3', '-p', '1', '--lineups', 'heuristic']
    )
    assert out.exit_code == 0, out.output
    assert 'base/3p' in out.output
