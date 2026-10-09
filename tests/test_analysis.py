"""Tests for post-game analysis (skipped unless PyTorch is installed)."""

import random

import pytest

torch = pytest.importorskip('torch')

from wonders.agents import GreedyAgent, HeuristicAgent  # noqa: E402
from wonders.agents.match import play_match  # noqa: E402
from wonders.cards.content import load_content  # noqa: E402
from wonders.engine.record import PlayerSetup  # noqa: E402
from wonders.ml.analysis import analyse, label, reason, summary_text  # noqa: E402

SETUPS = [
    PlayerSetup('A', 'The Pyramids of Giza', 'A'),
    PlayerSetup('B', 'The Colossus of Rhodes', 'B'),
    PlayerSetup('C', 'The Temple of Artemis in Ephesus', 'A'),
]


def test_labels_need_a_clear_and_significant_loss():
    assert label(0.0, 0.0, best=True) == 'best'
    assert label(0.8, 0.1, best=False) == 'good'  # under 1 VP
    assert label(1.5, 1.0, best=False) == 'good'  # within noise
    assert label(1.5, 0.3, best=False) == 'inaccuracy'
    assert label(3.0, 0.3, best=False) == 'mistake'
    assert label(6.0, 0.3, best=False) == 'blunder'


def test_reasons_name_gains_and_costs():
    assert reason({'science': 2.0, 'treasury': -1.0}, {}) == 'gains 2.0 science; gives up 1.0 coins now'
    assert 'longer-term outlook' in reason({'civilian': -3.0}, {})


@pytest.fixture(scope='module')
def report():
    content = load_content('base')
    agents = [GreedyAgent(), HeuristicAgent(), HeuristicAgent()]
    _game, record = play_match(content, SETUPS, agents, random.Random(5))
    return analyse(record, 0, depth=1, worlds=2, candidates=2, seed=1, what_if=1, what_if_runs=2)


def test_analyse_rates_every_regular_turn(report):
    decisions = report['decisions']
    assert len(decisions) == 18 and all('loss' in d for d in decisions)  # 3 Ages x 6 turns, all rated
    assert {d['label'] for d in decisions} <= {'best', 'good', 'inaccuracy', 'mistake', 'blunder'}
    for d in decisions:
        assert d['loss'] >= 0 and any(c['card'] == d['played']['card'] for c in d['candidates'])
        if d['label'] == 'best':
            assert d['loss'] == 0
    assert len(report['graph']) == 18
    assert set(report['key_moments']) <= {
        d['index'] for d in decisions if d['label'] in ('inaccuracy', 'mistake', 'blunder')
    }
    assert 0.0 <= report['summary']['accuracy'] <= 1.0
    assert report['player'] == 'A' and 'Accuracy' not in summary_text(report)


def test_server_refuses_to_analyse_an_unfinished_game():
    from wonders.play.server import PlayServer
    from wonders.play.session import ConfigError

    play = PlayServer()
    with pytest.raises(ConfigError):
        play.start_analysis()
    play.new_game({'seats': [{'player': 'human'}, {'player': 'heuristic'}, {'player': 'greedy'}], 'seed': 1})
    with pytest.raises(ConfigError):
        play.start_analysis()
    assert play.analysis_state() == {'status': 'idle'}


def test_patterns_and_what_if(report):
    kinds = {'trade', 'military', 'science', 'wonder', 'coins', 'passing', 'discards', 'timing'}
    assert len(report['patterns']) <= 6
    for p in report['patterns']:
        assert p['kind'] in kinds and p['title'] and p['detail'] and p['vp'] >= 0
    assert [p['vp'] for p in report['patterns']] == sorted((p['vp'] for p in report['patterns']), reverse=True)
    replayed = [d for d in report['decisions'] if 'what_if' in d]
    assert len(replayed) <= 1  # what_if=1
    for d in replayed:
        w = d['what_if']
        assert w['runs'] in (1, 2)
        for side in ('played', 'best'):
            assert 1 <= w[side]['mean_rank'] <= 3 and 0 <= w[side]['first'] <= 1


def test_aspects_compare_the_player_with_the_table(report):
    asp = report['aspects']
    assert asp['players'] == 3
    keys = {c['key'] for c in asp['categories']}
    assert 'total' in keys
    for c in asp['categories']:
        assert 1 <= c['rank'] <= 3 and c['best_other'] >= c['average'] - 1e-9
    assert [s['age'] for s in asp['by_age']] == [1, 2, 3]
    assert sum(s['moves'] for s in asp['by_age']) == sum(s['moves'] for s in asp['by_kind'].values()) == 18
    assert sum(n for row in asp['kind_matrix'].values() for n in row.values()) == 18
    assert {m['result'] for m in asp['military']} <= {'win', 'tie', 'loss'} and len(asp['military']) == 6
    assert sum(c['mine'] for c in asp['cards'].values()) > 0
    assert (
        0 <= asp['wonder']['built'] <= asp['wonder']['total'] and len(asp['wonder']['ages']) == asp['wonder']['built']
    )
    assert asp['economy']['trade_paid'] >= 0 and 1 <= asp['economy']['coins_rank'] <= 3


def test_report_reads_in_german(report):
    import copy
    import json

    from wonders.i18n import Translator
    from wonders.ml.analysis import localize

    json.dumps(report)  # the kept messages are JSON-ready
    original = copy.deepcopy(report)
    german = localize(report, Translator('de'))
    assert report == original  # localizing copies
    for en, de in zip(report['decisions'], german['decisions']):
        if en['reason']:
            assert de['reason'] != en['reason'] and 'gains' not in de['reason']
    for en, de in zip(report['patterns'], german['patterns']):
        assert de['title'] != en['title'] and de['detail'] != en['detail']
