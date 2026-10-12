"""Tests for the estimate of the wonder board sides' strength."""

import random

import pytest

from wonders.agents.wonder_values import (
    TABLES,
    _deal,
    estimate_wonder_values,
    fit_wonder_values,
    side_name,
    table_name,
    tables_for,
)
from wonders.cards.content import load_content

pytest.importorskip('numpy')


def test_tables_cover_every_bundled_board():
    boards = {w.name for combo, _n in TABLES for w in load_content(*combo).wonders}
    assert boards == {w.name for w in load_content('base', 'leaders', 'cities', 'wonderpack').wonders}
    assert table_name((('base', 'leaders'), 5)) == 'base+leaders/5p'


def test_deal_rotates_the_boards_over_the_same_decks():
    games = _deal((3, 0, ('base',), 4, 2, 'greedy'))
    assert len(games) == 2
    first, second = ([s for s, _m in g] for g in games)
    assert second == first[2:] + first[:2]
    for game in games:
        assert sum(m for _s, m in game) == pytest.approx(0.0)


def _synthetic(truth, tables, deals, rng, noise=8.0):
    """Deals whose margins follow the model exactly: table t adds ``shift[t]`` to side 0."""
    out = []
    for t, (_combo, n) in enumerate(tables):
        for _ in range(deals):
            sides = rng.sample(sorted(truth[t]), n)
            strength = [truth[t][s] for s in sides]
            game = [
                (s, strength[i] - (sum(strength) - strength[i]) / (n - 1) + rng.gauss(0, noise))
                for i, s in enumerate(sides)
            ]
            mean = sum(m for _s, m in game) / n
            out.append((t, [[(s, m - mean) for s, m in game]]))
    return out


def test_fit_recovers_side_values_and_table_shifts():
    tables = tables_for([('base',)], [3, 5])
    base = {f's{i}': v for i, v in enumerate([3.0, 1.0, 0.0, -1.0, -3.0, 0.0, 0.0])}
    truth = [dict(base), dict(base, s0=-3.0, s4=3.0)]  # the strongest and weakest side swap at 5p
    for values in truth:
        mean = sum(values.values()) / len(values)
        for s in values:
            values[s] -= mean
    fit = fit_wonder_values(_synthetic(truth, tables, 3000, random.Random(1)), tables)
    for t, table in enumerate(tables):
        for s, v in fit.tables[table_name(table)].items():
            assert v.value == pytest.approx(truth[t][s], abs=4 * v.se + 0.1)
            assert 0 < v.se < 0.5
    assert sum(v.value for v in fit.main.values()) == pytest.approx(0.0, abs=1e-9)
    assert fit.main['s1'].value == pytest.approx(1.0, abs=0.3)
    assert fit.tau > 1.0  # the swap is a real table effect, so the tables keep their own values


def test_pooling_shrinks_tables_towards_the_main_effect():
    tables = tables_for([('base',)], [3, 4])
    truth = [{f's{i}': v for i, v in enumerate([2.0, 1.0, 0.0, -1.0, -2.0])}] * 2
    deals = _synthetic(truth, tables, 300, random.Random(2))
    free, pooled = fit_wonder_values(deals, tables, pooling=1e-3), fit_wonder_values(deals, tables, pooling=1e6)
    a, b = (pooled.tables[table_name(t)]['s0'].value for t in tables)
    assert a == pytest.approx(b, abs=1e-3)
    a, b = (free.tables[table_name(t)]['s0'].value for t in tables)
    assert a != pytest.approx(b, abs=1e-3)


def test_estimate_smoke():
    tables = tables_for([('base',)], [3])
    values = estimate_wonder_values(games=4, tables=tables, rotations=2, seed=0)
    again = estimate_wonder_values(games=4, tables=tables, rotations=2, seed=0)
    assert values.as_dict() == again.as_dict()
    assert values.games == 4
    table = values.tables['base/3p']
    assert sum(v.games for v in table.values()) == 4 * 3
    assert sum(v.value for v in table.values()) == pytest.approx(0.0, abs=1e-9)
    assert all(s.endswith((' A', ' B')) for s in values.main)
    assert side_name('The Pyramids of Giza', 'B') == 'The Pyramids of Giza B'
    assert '| side | main | base/3p |' in values.markdown()
    with pytest.raises(ValueError):
        estimate_wonder_values(games=4, tables=tables, rotations=4)
