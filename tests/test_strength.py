"""Tests for card and board strength: the rate card, paired simulations, calibration and the builder's jobs."""

import json
import math
import random
from typing import Any, Dict

import pytest

from tests.test_builder import _call, _project, builder_url  # noqa: F401 - fixture
from tests.test_extensions import isolated  # noqa: F401 - fixture
from wonders.agents import strength as st
from wonders.cards.base import Card
from wonders.cards.content import build_card, load_content
from wonders.play import builder
from wonders.play import strength as bs

BASE = load_content('base', 'cities')


@pytest.fixture(scope='module')
def ref() -> st.Reference:
    return st.reference(players=(3, 5), games=2)


def _card(effects: Any, age: int = 2, cost: Dict[str, Any] = {}, color: str = 'blue', **extra: Any) -> Card:
    return build_card({'name': 'Test', 'color': color, 'age': age, 'cost': cost, 'effects': effects, **extra}, 'test')


def test_reference_positions_cover_every_age(ref):
    assert {a: len(v) for a, v in ref.after.items()} == {1: 4, 2: 4, 3: 4}
    assert all(g.age == 3 for g in ref.final)
    assert sum(len(p.cards) for p in ref.after[1][0].players) < sum(len(p.cards) for p in ref.final[0].players)


def test_rate_card_follows_the_rules_exchange_rates(ref):
    cal = st.Calibration()  # the priors

    def vp(n: int, **kw: Any) -> float:
        return st.rate_card(_card([{'type': 'victory_points', 'points': n}], **kw), ref, cal).value

    assert vp(5) - vp(3) == pytest.approx(2.0)  # one VP is one VP
    assert vp(3) - vp(3, cost={'coins': 3}) == pytest.approx(1.0)  # 3 coins are one VP
    assert vp(3) > vp(3, cost={'resources': {'stone': 2}})
    assert vp(3, chain_enables=['Other']) > vp(3)
    assert vp(0) == pytest.approx(-1.0)  # an empty card is worth less than the 3 coins of a discard
    coins = st.rate_card(_card([{'type': 'coins', 'amount': 6}]), ref, cal)
    assert coins.parts['coins'] == pytest.approx(2.0)


def test_rate_card_measures_military_science_and_production_in_positions(ref):
    cal = st.Calibration()
    shield = st.rate_card(_card([{'type': 'military', 'shields': 2}], age=1, color='red'), ref, cal)
    later = st.rate_card(_card([{'type': 'military', 'shields': 2}], age=3, color='red'), ref, cal)
    assert shield.parts['military'] > later.parts['military'] > 0  # early shields count in more conflicts
    science = st.rate_card(_card([{'type': 'science', 'symbol': 'gear'}], color='green'), ref, cal)
    assert science.parts['science'] > 0
    wood = st.rate_card(_card([{'type': 'production', 'resources': ['wood']}], age=1, color='brown'), ref, cal)
    assert wood.parts['production'] == pytest.approx(3 * cal.weights['production'])


def test_rate_card_handles_every_official_card_and_board(ref):
    content = load_content('base', 'leaders', 'cities', 'wonderpack')
    for card in content.cards:
        assert math.isfinite(st.rate_card(card, ref).value), card.name
    for wonder in content.wonders:
        rating = st.rate_wonder(wonder, ref)
        assert math.isfinite(rating.value) and len(st.rate_stages(wonder, ref)) == len(wonder.stages)


def test_on_build_effects_are_measured_on_copies(ref):
    plunder = _card([{'type': 'plunder', 'amount': 2, 'scope': 'others'}], color='black')
    before = [p.coins for g in ref.after[2] for p in g.players]
    rating = st.rate_card(plunder, ref, st.Calibration())
    assert rating.parts['on_build'] > 0  # the others lose coins: the margin grows
    assert [p.coins for g in ref.after[2] for p in g.players] == before  # the reference is untouched


def test_fit_weights_recovers_a_known_rate_card():
    rng = random.Random(0)
    truth = dict(st.PRIOR_WEIGHTS, points=1.3, production=0.9, resource_cost=-0.7)
    rows, targets = [], []
    for _ in range(300):
        feats = {k: rng.uniform(0, 4) for k in st.FEATURES}
        rows.append(feats)
        targets.append(sum(truth[k] * feats[k] for k in st.FEATURES) + 0.5 + rng.gauss(0, 0.2))
    weights, intercept = st.fit_weights(rows, targets, [0.2] * len(rows), strength=0.01, tau=0.2)
    for k in ('points', 'production', 'resource_cost'):
        assert weights[k] == pytest.approx(truth[k], abs=0.05)
    assert intercept == pytest.approx(0.5, abs=0.3)


def test_recalibration_moves_with_the_evidence():
    assert st.recalibrate([], [], []) == pytest.approx((0.0, 1.0))
    few = st.recalibrate([1.0], [3.0], [1.0])
    many = st.recalibrate([1.0, 2.0, 3.0] * 20, [3.0, 4.0, 5.0] * 20, [0.3] * 60)
    assert 0 < few[0] < many[0]
    assert many[0] + 2.0 * many[1] == pytest.approx(4.0, abs=0.2)  # near the simulations (a little shrinkage)


def test_paired_card_samples_are_deterministic_and_skip_unaffordable_builds():
    palace = BASE.card('Palace')
    base = load_content('base')
    a = st.card_pair(base, palace, 4, seed=11)
    assert a == st.card_pair(base, palace, 4, seed=11)
    lumber = BASE.card('Lumber Yard')
    estimate = st.simulate_card(lumber, samples=3, seed=2, players=(3,), content=base)
    assert estimate.samples == 3 and math.isfinite(estimate.value)
    assert json.dumps(st.Estimate(1.0, float('inf'), 1).as_dict())  # JSON-safe: no Infinity
    assert st.Estimate(1.0, float('inf'), 1).as_dict()['se'] is None


def test_natural_pairs_and_board_pairs():
    base = load_content('base')
    pairs = st.natural_pairs(base, ['Palace', 'Lumber Yard', 'Baths'], 3, seed=5, expansions=('base',))
    assert set(pairs) <= {'Palace', 'Lumber Yard', 'Baths'} and pairs
    rhodes = base.wonder('The Colossus of Rhodes', 'B')
    estimate = st.simulate_wonder(rhodes, samples=2, seed=1, players=(3,), content=base, calibration=st.Calibration())
    assert estimate.samples == 2


def test_builder_rates_unsaved_specs_and_compares_with_official_cards(isolated):  # noqa: F811
    project = _project()
    card = bs.rate(
        'card', {'name': 'X', 'color': 'blue', 'age': 2, 'effects': [{'type': 'victory_points', 'points': 5}]}
    )
    assert card['rating']['value'] > 0 and 'points' in card['rating']['parts']
    if st.default_calibration().cards:  # with the bundled calibration: official blue Age II cards to compare
        assert card['official']['min'] <= card['official']['median'] <= card['official']['max']
    side = bs.rate('wonder', project['wonders'][0], [])
    assert len(side['stages']) == 2
    with pytest.raises(Exception):
        bs.rate('card', {'name': 'X', 'color': 'nope'})


def test_builder_jobs_simulate_saved_items_and_the_whole_expansion(isolated, monkeypatch):  # noqa: F811
    monkeypatch.setattr(bs, 'CARD_SAMPLES', 2)
    monkeypatch.setattr(bs, 'WONDER_SAMPLES', 2)
    assert builder.save_project(_project())['check']['loaded']
    jobs = bs.Jobs(processes=1)
    with pytest.raises(ValueError):
        jobs.start('nope', 'items')
    assert jobs.start('docks', 'items', wait=True)['running'] is False
    view = bs.overview('docks', jobs)
    (dock,) = view['cards']
    assert dock['simulated']['samples'] == 2 and not dock['simulated']['stale']
    assert dock['estimate']['source'] == 'item'
    assert {w['name'] for w in view['wonders']} == {'The Lighthouse A', 'The Lighthouse B'}
    # Editing the card makes its simulation stale until it is simulated again.
    changed = _project()
    changed['cards'][0]['effects'] = [{'type': 'victory_points', 'points': 2}]
    builder.save_project(changed, 'docks')
    stale = bs.overview('docks')['cards'][0]
    assert stale['simulated']['stale'] and stale['estimate']['source'] == 'rate'
    done = jobs.start('docks', 'expansion', games=10, wait=True)
    assert 'error' not in done, done
    view = bs.overview('docks', jobs)
    assert view['cards'][0]['expansion']['samples'] > 0
    assert view['expansion']['games'] == 10 and 'cards' in view['expansion']['correction']
    assert view['cards'][0]['estimate']['source'] == 'expansion'
    assert 'corrected' in view['cards'][0]['rating']
    bs.forget('docks')
    assert bs.load('docks') == {'cards': {}, 'wonders': {}, 'expansion': None}


def test_server_rates_and_reports_strength(builder_url, monkeypatch):  # noqa: F811
    rated = _call(
        builder_url + '/api/builder/rate',
        {'kind': 'card', 'spec': _project()['cards'][0], 'colors': _project()['colors']},
    )
    assert rated['kind'] == 'card' and 'value' in rated['rating']
    assert 'error' in _call(builder_url + '/api/builder/rate', {'kind': 'card', 'spec': {'color': 'nope'}})
    _call(builder_url + '/api/builder/save', {'project': _project()})
    view = _call(builder_url + '/api/builder/strength?id=docks')
    assert view['cards'][0]['name'] == 'Dock' and view['job'] is None
    with pytest.raises(Exception):
        _call(builder_url + '/api/builder/strength', {'id': 'nope', 'mode': 'items'})
