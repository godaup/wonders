"""Tests for head-to-head evaluation, leader-policy routing and Leaders diagnostics."""

import json
import random

import pytest

torch = pytest.importorskip('torch')

from click.testing import CliRunner  # noqa: E402

from wonders import cli  # noqa: E402
from wonders.agents.bench import AGENTS  # noqa: E402
from wonders.agents.match import play_match  # noqa: E402
from wonders.cards.content import load_content  # noqa: E402
from wonders.engine.record import PlayerSetup  # noqa: E402
from wonders.env.vocab import universal_vocab, vocab_for  # noqa: E402
from wonders.ml import NeuralAgent, save_checkpoint  # noqa: E402
from wonders.ml.diagnostics import leaders_report, leaders_report_table, teacher_agreement  # noqa: E402
from wonders.ml.evaluate import compare_agents, head_to_head, head_to_head_table, parse_cells  # noqa: E402
from wonders.ml.imitation import new_model  # noqa: E402

LEADERS = ('base', 'leaders')
SETUPS = [
    PlayerSetup('N', 'The Pyramids of Giza'),
    PlayerSetup('H', 'The Colossus of Rhodes'),
    PlayerSetup('G', 'The Temple of Artemis in Ephesus'),
]


def _checkpoint(path, vocab, seed=0):
    torch.manual_seed(seed)
    model, names = new_model(hidden=16, vocab=vocab)
    save_checkpoint(str(path), model, names, **vocab.meta())
    return str(path)


@pytest.fixture(scope='module')
def universal(tmp_path_factory):
    return _checkpoint(tmp_path_factory.mktemp('u') / 'u.pt', universal_vocab())


@pytest.fixture(scope='module')
def universal_v2(tmp_path_factory):
    return _checkpoint(tmp_path_factory.mktemp('u2') / 'u2.pt', universal_vocab('table-v2'), seed=1)


@pytest.fixture(scope='module')
def base_only(tmp_path_factory):
    return _checkpoint(tmp_path_factory.mktemp('b') / 'b.pt', vocab_for(('base',)), seed=2)


def _count_hook_calls(agent, monkeypatch):
    calls = {'draft': 0, 'recruit': 0}
    original_draft, original_recruit = type(agent).draft_leader, type(agent).recruit

    def draft(self, options):
        calls['draft'] += 1
        return original_draft(self, options)

    def recruit(self, leaders, obs):
        calls['recruit'] += 1
        return original_recruit(self, leaders, obs)

    monkeypatch.setattr(type(agent), 'draft_leader', draft)
    monkeypatch.setattr(type(agent), 'recruit', recruit)
    return calls


@pytest.mark.parametrize('leader_policy, uses_hooks', [(True, False), (False, True)])
def test_neural_agent_plays_leaders_with_its_policy(universal, monkeypatch, leader_policy, uses_hooks):
    agent = NeuralAgent(universal, leader_policy=leader_policy)
    calls = _count_hook_calls(agent, monkeypatch)
    agents = [agent, AGENTS['heuristic'].build(random.Random(0)), AGENTS['greedy'].build(random.Random(1))]
    play_match(load_content(*LEADERS), SETUPS, agents, rng=random.Random(5), expansions=LEADERS)
    assert (calls['draft'] > 0) is uses_hooks
    assert (calls['recruit'] > 0) is uses_hooks


def test_base_only_checkpoint_keeps_the_default_hooks(base_only):
    assert NeuralAgent(base_only).leader_policy is False


def test_head_to_head(universal, universal_v2, base_only):
    cells = parse_cells('base/3p,leaders/4p')
    result = head_to_head([('v1', universal), ('v2', universal_v2), ('b', base_only)], cells=cells, games=4)
    base_cell, leaders_cell = result['cells']
    assert set(base_cell['models']) == {'v1', 'v2', 'b'}  # 3 models fill the 3-player table
    assert set(leaders_cell['models']) == {'v1', 'v2'}  # the base-only model is not eligible
    for pair in result['pairs']:
        assert 0 <= pair['a_above_b'] <= 1 and pair['ci'][0] <= pair['a_above_b'] <= pair['ci'][1]
    v1_v2 = next(p for p in result['pairs'] if (p['a'], p['b']) == ('v1', 'v2'))
    assert v1_v2['games'] == 8
    assert set(result['pairs_by_combo']) == {'base', 'leaders'}
    table = head_to_head_table(result, 'h2h')
    assert 'v1 vs v2' in table and 'leaders/4p' in table
    json.dumps(result)


def test_head_to_head_skips_cells_and_checks_labels(universal, base_only):
    result = head_to_head([('u', universal), ('b', base_only)], cells=parse_cells('cities/3p'), games=2)
    assert result['cells'][0]['skipped'] and result['pairs'] == []
    with pytest.raises(ValueError):
        head_to_head([('x', universal), ('x', base_only)], cells=parse_cells('base/3p'), games=2)


def test_compare_agents_rejects_duplicates_and_filler_clash():
    with pytest.raises(ValueError):
        compare_agents([AGENTS['board'], AGENTS['board']], parse_cells('base/3p'), games=1)
    with pytest.raises(ValueError):
        compare_agents([AGENTS['board'], AGENTS['heuristic']], parse_cells('base/3p'), games=1)


def test_teacher_agreement_counts_every_decision_kind(universal):
    agreement = teacher_agreement(universal, parse_cells('leaders/3p'), games=2)
    assert {'play', 'draft', 'recruit'} <= set(agreement)
    assert agreement['draft']['decisions'] == 2 * 3 * 4  # 2 games x 3 seats x 4 picks
    assert all(0.0 <= a['agreement'] <= 1.0 for a in agreement.values())


def test_leaders_report(universal):
    report = leaders_report(universal, 'u', cells=parse_cells('leaders/3p'), games=2, agreement_games=1)
    assert [c['cell'] for c in report['model_ablation']['cells']] == ['leaders/3p']
    names = set(report['model_ablation']['cells'][0]['models'])
    assert names == {'u', 'u[cost]', 'u[random]'}
    assert set(report['teacher_ablation']['cells'][0]['models']) == {'board', 'board[random-draft]', 'board[random]'}
    text = leaders_report_table(report, 'diag')
    assert 'Teacher agreement' in text and 'u vs u[cost]' in text
    json.dumps(report)


def test_cli_h2h_and_leaders(universal, universal_v2, tmp_path):
    runner = CliRunner()
    out = tmp_path / 'h2h.json'
    args = ['train', 'h2h', f'v1={universal}', f'v2={universal_v2}', '--cells', 'base/3p', '-n', '2', '-p', '1']
    res = runner.invoke(cli.main, [*args, '-o', str(out)])
    assert res.exit_code == 0, res.output
    assert 'v1 vs v2' in res.output and json.loads(out.read_text())['models'] == ['v1', 'v2']
    res = runner.invoke(cli.main, ['train', 'h2h', universal, '-n', '1'])
    assert res.exit_code != 0  # needs two checkpoints
    res = runner.invoke(cli.main, ['train', 'leaders', universal, '-n', '1', '--agreement-games', '1', '-p', '1'])
    assert res.exit_code == 0, res.output
    assert 'Leaders diagnostics: u' in res.output
    res = runner.invoke(
        cli.main, ['train', 'leaders', f'mine={universal}', '-n', '1', '--agreement-games', '1', '-p', '1']
    )
    assert res.exit_code == 0, res.output
    assert 'Leaders diagnostics: mine' in res.output


def test_exploitability_seats_one_model_against_target_copies(universal, universal_v2, tmp_path):
    from wonders.ml.evaluate import exploitability

    result = exploitability(('e', universal), ('t', universal_v2), cells=parse_cells('base/3p,leaders/4p'), games=4)
    assert [c['cell'] for c in result['cells']] == ['base/3p', 'leaders/4p']
    assert all(0 <= c['win_rel'] <= 4 and 1 <= c['mean_rank'] <= 4 for c in result['cells'])
    assert result['win_rel'] >= 0 and result['win_rel_se'] >= 0
    with pytest.raises(ValueError):
        exploitability(('x', universal), ('x', universal_v2), cells=parse_cells('base/3p'), games=1)
    out = tmp_path / 'x.json'
    res = CliRunner().invoke(
        cli.main,
        [
            'train',
            'exploit',
            f'e={universal}',
            f't={universal_v2}',
            '--cells',
            'base/3p',
            '-n',
            '2',
            '-p',
            '1',
            '-o',
            str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    assert 'e vs t field' in res.output and json.loads(out.read_text())['target'] == 't'


def test_head_to_head_records_its_seating(universal, universal_v2):
    models = [('v1', universal), ('v2', universal_v2)]
    assert head_to_head(models, cells=parse_cells('base/3p'), games=2)['seating'] == 'shuffle'
    assert head_to_head(models, cells=parse_cells('base/3p'), games=2, seating='rotate')['seating'] == 'rotate'
