"""Tests for local storage, the Elo ratings and the achievements."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import pytest

from wonders.agents.bench import AGENTS
from wonders.play import rating
from wonders.play.achievements import ACHIEVEMENTS, achievements, earned
from wonders.play.rating import (
    AGENT_ELO,
    RatedGame,
    agent_elo,
    expected,
    fit,
    pairwise,
    places,
    player_history,
    rated_game,
)
from wonders.storage import HOME_VARIABLE, StorageError, Store, home


# -- storage -------------------------------------------------------------------------------------
def test_store_saves_lists_and_deletes_games(tmp_path: Path) -> None:
    store = Store(tmp_path)
    assert store.list_games() == []
    store.save_game({'id': 'b-2', 'created': '2026-10-09T10:00:00+00:00', 'log': [1], 'record': {}})
    store.save_game({'id': 'a-1', 'created': '2026-10-08T10:00:00+00:00', 'log': [], 'record': {}})
    games = store.list_games()
    assert [g['id'] for g in games] == ['a-1', 'b-2']  # oldest first
    assert all('log' not in g and 'record' not in g and g['schema'] == 1 for g in games)
    assert store.load_game('b-2')['log'] == [1]
    store.save_game({**store.load_game('b-2'), 'log': [1, 2]})  # overwrite in place
    assert store.load_game('b-2')['log'] == [1, 2] and store.load_game('b-2')['created'].startswith('2026-10-09')
    store.delete_game('a-1')
    assert [g['id'] for g in store.list_games()] == ['b-2']
    assert not list((tmp_path / 'games').glob('*.tmp'))  # atomic writes leave nothing behind


def test_store_rejects_bad_ids_and_skips_foreign_files(tmp_path: Path) -> None:
    store = Store(tmp_path)
    for bad in ('../escape', 'a/b', '', 'x' * 65):
        with pytest.raises(StorageError):
            store.load_game(bad)
    with pytest.raises(StorageError, match='no saved game'):
        store.load_game('missing')
    with pytest.raises(StorageError):
        store.delete_game('missing')
    store.games_dir.mkdir(parents=True)
    (store.games_dir / 'broken.json').write_text('{not json')
    (store.games_dir / 'future.json').write_text(json.dumps({'id': 'future', 'schema': 99}))
    assert store.list_games() == []


def test_home_follows_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from wonders.extensions import extensions_home

    monkeypatch.setenv(HOME_VARIABLE, str(tmp_path))
    monkeypatch.delenv('WONDERS_EXTENSIONS', raising=False)
    assert home() == tmp_path and Store().games_dir == tmp_path / 'games'
    assert extensions_home() == tmp_path / 'extensions'


# -- rating --------------------------------------------------------------------------------------
def test_places_and_pairwise_results() -> None:
    assert places([50, 60, 50, 40], [3, 1, 3, 9]) == [2, 1, 2, 4]  # an exact tie shares the place
    assert places([50, 50], [4, 3]) == [1, 2]  # coins break ties
    assert pairwise([2, 1, 2, 4], 0) == [(1, 0.0), (2, 0.5), (3, 1.0)]
    assert expected(1500, 1500) == pytest.approx(0.5)
    assert expected(1900, 1500) == pytest.approx(1 - expected(1500, 1900))


def test_every_agent_has_a_rating() -> None:
    assert set(AGENTS) | {'neural'} <= set(AGENT_ELO)
    assert AGENT_ELO['random'] == 1000  # the anchor
    assert AGENT_ELO['neural'] > AGENT_ELO['board'] > AGENT_ELO['heuristic'] > AGENT_ELO['random']
    assert agent_elo('neural:something-else') == rating.DEFAULT_AGENT_ELO


def test_player_rating_follows_the_results() -> None:
    win = RatedGame('w', (1, 2, 3), 0, ('', 'heuristic', 'board'))
    loss = RatedGame('l', (3, 1, 2), 0, ('', 'heuristic', 'board'))
    history = player_history([win, win, loss])
    assert [p.id for p in history] == ['w', 'w', 'l'] and history[0].before == rating.START
    assert history[0].after > history[0].before and history[2].after < history[2].before
    assert history[1].before == history[0].after and history[-1].games == 3
    # beating the random agent is worth less than beating the network
    weak = player_history([RatedGame('a', (1, 2, 3), 0, ('', 'random', 'random'))])[0]
    strong = player_history([RatedGame('b', (1, 2, 3), 0, ('', 'neural', 'neural'))])[0]
    assert 0 < weak.after - weak.before < strong.after - strong.before <= rating.K_PROVISIONAL
    # after the provisional games the rating moves more slowly
    settled = player_history([loss] * rating.PROVISIONAL + [win])
    assert settled[-1].after - settled[-1].before < history[0].after - history[0].before


def test_fit_recovers_the_strength_order() -> None:
    truth = {'random': 1000.0, 'b': 1400.0, 'c': 1800.0}
    rng = random.Random(0)
    results: List[Any] = []
    for _ in range(3000):
        lineup = rng.sample(list(truth), 3)
        perf = [truth[name] / 400 * 2.3 + rng.gauss(0, 1.3) for name in lineup]  # noisy strengths
        results.append((lineup, places(perf, [0, 0, 0])))
    fitted = fit(results)
    assert fitted['random'] == pytest.approx(1000.0)
    assert fitted['random'] < fitted['b'] < fitted['c']


# -- achievements --------------------------------------------------------------------------------
def _game(**result: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        'places': [1, 2, 3],
        'colors': [{'blue': 3}, {'red': 2}, {}],
        'scores': [dict.fromkeys(('military', 'treasury', 'wonders', 'civilian', 'commercial', 'guilds', 'science'), 2)]
        * 3,
        'discard_builds': [0, 0, 0],
    }
    return {
        'id': 'g',
        'finished': True,
        'finished_at': '2026-10-09T12:00:00+00:00',
        'human': 0,
        'players': [{'agent': 'human'}, {'agent': 'board'}, {'agent': 'heuristic'}],
        'result': {**base, **result},
    }


def test_achievements_are_checked_from_the_saved_result() -> None:
    assert {a.id for a in ACHIEVEMENTS} == {'pacifist', 'renaissance', 'dumpster'}
    assert earned(_game()) == ['pacifist']  # won without red cards
    assert earned(_game(colors=[{'red': 1}, {}, {}])) == []
    assert earned(_game(places=[2, 1, 3])) == []
    allround = dict.fromkeys(('military', 'treasury', 'wonders', 'civilian', 'commercial', 'guilds', 'science'), 5)
    assert 'renaissance' in earned(_game(scores=[allround, {}, {}], places=[3, 1, 2]))
    assert earned(_game(discard_builds=[3, 0, 0], places=[2, 1, 3])) == ['dumpster']
    assert earned({**_game(), 'finished': False}) == []


def test_achievement_summary_counts_games() -> None:
    first = {**_game(), 'id': 'one', 'finished_at': '2026-10-01T00:00:00+00:00'}
    second = {**_game(), 'id': 'two'}
    summary = {a['id']: a for a in achievements([first, second])}
    assert summary['pacifist']['count'] == 2 and summary['pacifist']['games'] == ['one', 'two']
    assert summary['pacifist']['first'] == '2026-10-01T00:00:00+00:00'
    assert summary['dumpster']['count'] == 0 and summary['dumpster']['first'] is None
    assert rated_game(first) == RatedGame('one', (1, 2, 3), 0, ('', 'board', 'heuristic'))
    assert rated_game({**first, 'finished': False}) is None
