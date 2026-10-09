"""Tests for local storage, the Elo ratings and the achievements."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import pytest

from wonders.agents.bench import AGENT_NAMES, AGENTS, agent_id, agent_name
from wonders.play import rating
from wonders.play.achievements import ACHIEVEMENTS, GROUPS, POINTS, TIERS, achievements, earned
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
CATS = ('military', 'treasury', 'wonders', 'civilian', 'commercial', 'guilds', 'science')


def _game(players: Any = ('human', 'board', 'heuristic'), expansions: Any = ('base',), **result: Any) -> Dict[str, Any]:
    n = len(players)
    base: Dict[str, Any] = {
        'totals': [50] + [40] * (n - 1),
        'coins': [3] * n,
        'places': list(range(1, n + 1)),
        'colors': [{'blue': 3, 'red': 1}] + [{}] * (n - 1),
        'scores': [dict.fromkeys(CATS, 2)] * n,
        'cards': [18] * n,
        'chains': [0] * n,
        'stages': [1] * n,
        'stages_total': [3] * n,
        'tokens': [[1, -1]] * n,
        'debt': [0] * n,
        'discards': [1] * n,
        'discard_builds': [0] * n,
        'traded': 4,
    }
    return {
        'id': 'g',
        'finished': True,
        'finished_at': '2026-10-09T12:00:00+00:00',
        'human': 0,
        'expansions': list(expansions),
        'players': [{'agent': a, 'wonder': f'W{i}'} for i, a in enumerate(players)],
        'result': {**base, **result},
    }


def test_the_achievement_catalogue_is_well_formed() -> None:
    ids = [a.id for a in ACHIEVEMENTS]
    assert len(ids) == len(set(ids)) >= 30
    for a in ACHIEVEMENTS:
        assert a.tier in TIERS and a.group in GROUPS and a.icon and a.name and a.description.endswith(('.', '!'))
        assert (a.check is None) != (a.career is None)
    tiers = [a.tier for a in ACHIEVEMENTS]
    assert all(tiers.count(t) >= 2 for t in TIERS)  # a spread of difficulty
    assert {a.group for a in ACHIEVEMENTS} == set(GROUPS)
    summary = achievements([])
    assert all(item['count'] == 0 and item['points'] == POINTS[item['tier']] for item in summary)


def test_feats_are_checked_from_the_saved_result() -> None:
    plain = _game()
    assert earned(plain) == ['first_steps']  # a 10-point win with nothing special
    assert earned({**plain, 'finished': False}) == []
    assert 'pacifist' in earned(_game(colors=[{'blue': 3}, {}, {}]))
    assert 'wonderwall' in earned(_game(stages=[3, 0, 0]))
    assert 'full_spectrum' in earned(
        _game(colors=[dict.fromkeys(('brown', 'grey', 'yellow', 'red', 'blue', 'green', 'purple'), 1), {}, {}])
    )
    assert 'chain_reaction' in earned(_game(chains=[4, 0, 0]))
    assert 'landslide' in earned(_game(totals=[60, 45, 30]))
    assert 'photo_finish' in earned(_game(totals=[50, 49, 30])) and 'photo_finish' in earned(_game(totals=[50, 50, 30]))
    assert 'photo_finish' not in earned(_game(totals=[50, 49, 30], places=[2, 1, 3]))
    assert 'last_drachma' in earned(_game(coins=[0, 3, 3]))
    assert 'sparta' in earned(_game(tokens=[[1, 1, 3, 3, 5, 5], [], []]))
    assert 'sparta' not in earned(_game(tokens=[[1, 1, 3, 3, 5, 5, -1], [], []]))
    assert 'glass_jaw' in earned(_game(tokens=[[-1] * 5, [], []]))
    assert 'autarky' in earned(_game(traded=0)) and 'autarky' not in earned(_game(traded=0, places=[2, 1, 3]))
    assert 'less_is_more' in earned(_game(cards=[14, 18, 18]))
    assert 'recycling' in earned(_game(discards=[4, 0, 0]))
    assert 'dumpster' in earned(_game(discard_builds=[3, 0, 0], places=[3, 1, 2]))  # winning not needed
    assert 'seventy' in earned(_game(totals=[70, 40, 40]))
    allround = dict.fromkeys(CATS, 5)
    assert 'renaissance' in earned(_game(scores=[allround, {}, {}], places=[3, 1, 2]))


def test_expansion_and_opponent_feats() -> None:
    assert 'hall_of_fame' in earned(_game(expansions=('base', 'leaders'), colors=[{'white': 4}, {}, {}]))
    assert 'self_made' in earned(_game(expansions=('base', 'leaders')))
    assert 'self_made' not in earned(_game())  # no leaders to skip in a base game
    assert 'shady_dealings' in earned(_game(expansions=('base', 'cities'), colors=[{'black': 4}, {}, {}]))
    assert 'too_big_to_fail' in earned(_game(debt=[3, 0, 0]))
    seven = ('human', 'board', 'random', 'greedy', 'civic', 'heuristic', 'table')
    assert 'packed_house' in earned(_game(seven, ('base', 'leaders', 'cities')))
    assert 'packed_house' not in earned(_game(seven, ('base', 'leaders')))
    assert 'fortunes_fool' in earned(_game(('human', 'random', 'board'), places=[2, 1, 3]))
    assert 'giant_slayer' in earned(_game(('human', 'neural', 'board'), places=[2, 3, 1]))
    assert 'giant_slayer' not in earned(_game(('human', 'neural', 'board'), places=[2, 1, 3]))
    boss = ('human', 'neural', 'adaptive', 'rival')
    assert 'boss_rush' in earned(_game(boss))
    assert 'boss_rush' not in earned(_game(boss, places=[2, 1, 3, 4]))


def test_career_goals_count_across_games() -> None:
    games = [{**_game(('human',) + ('random',) * (n - 1)), 'id': f'g{n}'} for n in (3, 4, 5, 6, 7)]
    for k, g in enumerate(games):
        g['players'][0]['wonder'] = f'Wonder {k % 3}'  # three different winning wonders
    summary = {a['id']: a for a in achievements(games, [1500, 1600, 1760, 1700, 1800])}
    assert summary['full_house']['count'] == 1 and summary['full_house']['games'] == ['g7']
    assert (summary['globetrotter']['progress'], summary['globetrotter']['goal']) == (3, 4)
    assert summary['globetrotter']['count'] == 0
    assert summary['rogues_gallery']['progress'] == 1  # only ever beat the random agent
    assert summary['rising_star']['games'] == ['g5'] and summary['olympian']['progress'] == 1800
    assert summary['first_steps']['count'] == 5 and summary['first_steps']['first'] == '2026-10-09T12:00:00+00:00'
    assert rated_game(games[0]) == RatedGame('g3', (1, 2, 3), 0, ('', 'random', 'random'))
    assert rated_game({**games[0], 'finished': False}) is None


def test_agents_have_display_names() -> None:
    assert set(AGENTS) | {'neural'} <= set(AGENT_NAMES)
    assert len(set(AGENT_NAMES.values())) == len(AGENT_NAMES)
    assert agent_name('neural') == 'Pythia' and agent_name('my-plugin') == 'my-plugin'
    assert agent_id('pythia') == 'neural' and agent_id('Odysseus') == 'heuristic' and agent_id('board') == 'board'
