"""Tests for the meta rules, the effect schemas and the GUI's expansion builder (wonders.play.builder)."""

import io
import json
import random
import threading
import urllib.error
import urllib.request
import zipfile
from dataclasses import fields
from typing import Any, Dict, Iterator, Optional

import pytest

import wonders.extensions as extensions
from tests.test_extensions import isolated  # noqa: F401 - fixture
from wonders.cards.base import Card, Color, Cost
from wonders.cards.content import EFFECT_BUILDERS, build_age_deck, build_effect, load_content
from wonders.cards.library import wonder_gizah_a, wonder_rhodes_a
from wonders.cards.schema import EFFECT_SCHEMAS, KINDS, EffectSchema, Param
from wonders.effects.library import VictoryPoints
from wonders.engine.actions import Action
from wonders.engine.game import Game
from wonders.engine.player import Player
from wonders.play import builder
from wonders.play.server import make_server
from wonders.ruleset import EXPANSIONS, RULES, Expansion, Rules, Ruleset, check_rules, rule_order, rules_of


# -- meta rules -------------------------------------------------------------------------------
def test_rule_specs_match_the_rules_defaults():
    assert [(f.name, f.default) for f in fields(Rules)] == [(r.name, r.default) for r in RULES]
    assert Ruleset.of().rules == Rules() and Rules().changed() == {}
    assert Ruleset.of('leaders').rules.changed() == {'starting_coins': 6}


def test_rules_apply_in_a_fixed_order_and_deltas_add_up(monkeypatch):
    monkeypatch.setitem(
        EXPANSIONS, 'cheap', Expansion('cheap', 'Cheap', rules={'starting_coins': 2, 'discard_coins': 1})
    )
    monkeypatch.setitem(EXPANSIONS, 'rich', Expansion('rich', 'Rich', rules={'starting_coins': {'add': 2}}))
    monkeypatch.setitem(EXPANSIONS, 'zany', Expansion('zany', 'Zany', rules={'starting_coins': {'add': -1}}))
    assert rule_order(['zany', 'cheap', 'leaders', 'base']) == ['base', 'leaders', 'cheap', 'zany']
    # Whatever order they are chosen in: base 3, Leaders sets 6, extensions by id.
    for combo in (('base', 'leaders', 'rich'), ('rich', 'leaders')):
        assert rules_of(combo).starting_coins == 8
    assert rules_of(('leaders', 'rich', 'zany')).starting_coins == 7
    assert rules_of(('rich', 'cheap', 'leaders')).starting_coins == 4  # cheap sets 2 after leaders, rich adds 2
    assert rules_of(('cheap', 'leaders')).discard_coins == 1
    assert Ruleset.of('rich').starting_coins == 5


def test_rules_are_clamped_to_their_range(monkeypatch):
    monkeypatch.setitem(EXPANSIONS, 'poor', Expansion('poor', 'Poor', rules={'coins_per_point': {'add': -10}}))
    monkeypatch.setitem(EXPANSIONS, 'meek', Expansion('meek', 'Meek', rules={'defeat_points': {'add': 5}}))
    assert rules_of(('poor', 'meek')).coins_per_point == 1 and rules_of(('meek',)).defeat_points == 0


def test_check_rules():
    assert check_rules({'discard_coins': 2, 'defeat_points': -2}) == []
    assert check_rules({'starting_coins': {'add': -3}, 'victory_age_3': {'add': 20}}) == []
    problems = check_rules({'nope': 1, 'discard_coins': 'x', 'coins_per_point': 0, 'extra_guilds': True})
    assert len(problems) == 4 and 'unknown rule' in problems[0]
    problems = check_rules({'starting_coins': {'add': 31}, 'discard_coins': {'plus': 1}, 'extra_guilds': {'add': 1.5}})
    assert len(problems) == 3 and 'between -30 and 30' in problems[0]
    assert check_rules([1]) == ['rules must be an object of rule name to number']


def _game(monkeypatch, **rules: int) -> Game:
    monkeypatch.setitem(EXPANSIONS, 'tweak', Expansion('tweak', 'Tweak', rules=rules))
    return Game([Player('P0', wonder_gizah_a()), Player('P1', wonder_rhodes_a())], expansions=('base', 'tweak'))


def test_the_engine_applies_meta_rules(monkeypatch):
    g = _game(monkeypatch, discard_coins=5, coins_per_point=2, victory_age_1=4, defeat_points=-3, science_set_points=10)
    g.players[0].coins = 0
    g.apply(0, Action.discard(Card('junk', Color.GREY)))
    assert g.players[0].coins == 5
    assert g.score(0).treasury == 2
    g.players[0].add_card(Card('Barracks', Color.RED, Cost(), (build_effect({'type': 'military', 'shields': 1}),)))
    g.resolve_conflict(1)
    assert g.players[0].tokens == [4] and g.players[1].tokens == [-3]  # two players: one conflict
    for symbol in ('tablet', 'gear', 'compass'):
        g.players[1].add_card(Card(symbol, Color.GREEN, Cost(), (build_effect({'type': 'science', 'symbol': symbol}),)))
    assert g.score(1).science == 3 + 10


def test_extra_guilds_rule_sizes_age_three():
    content = load_content('base')
    assert len(build_age_deck(content, 3, 4, random.Random(0), extra_guilds=0)) == 4 * 7 - 2
    assert len(build_age_deck(content, 3, 4, random.Random(0))) == 4 * 7


def test_bundled_games_keep_the_base_rules():
    assert Game([Player('P0', wonder_gizah_a()), Player('P1', wonder_rhodes_a())]).rules == Rules()


# -- effect schemas ---------------------------------------------------------------------------
def test_every_effect_type_has_a_schema_whose_example_builds():
    assert set(EFFECT_SCHEMAS) == set(EFFECT_BUILDERS)
    for name, schema in EFFECT_SCHEMAS.items():
        build_effect({'type': name, **schema.example})
        assert all(p.kind in KINDS for p in schema.params), name
        assert set(schema.example) <= {p.name for p in schema.params} | {'description'}, name


def test_plugins_can_describe_their_effects(isolated):  # noqa: F811
    schema = EffectSchema('Double', 'Points', 'Twice the points.', (Param('points', 'int', 'Points', required=True),))
    extensions.REGISTRY.register_effect('double', lambda s: VictoryPoints(2 * s['points']), schema=schema)
    try:
        info = {e['type']: e['schema'] for e in builder.builder_info()['effects']}
        assert info['double']['params'][0]['name'] == 'points'
    finally:
        EFFECT_SCHEMAS.pop('double', None)


# -- builder ----------------------------------------------------------------------------------
def _side(side: str, name: str = 'The Lighthouse') -> Dict[str, Any]:
    stage = {'cost': {'resources': {'stone': 2}}, 'effects': [{'type': 'victory_points', 'points': 3}]}
    return {'name': name, 'side': side, 'starting_resource': 'glass', 'stages': [stage, stage]}


def _project(**over: Any) -> Dict[str, Any]:
    project = {
        'id': 'docks',
        'name': 'Docks',
        'version': '0.1.0',
        'rules': {'starting_coins': {'add': 2}, 'discard_coins': 2},
        'colors': [{'id': 'harbour', 'label': 'Harbour', 'fill': '#1f8a8a', 'score': 'civilian', 'tradeable': True}],
        'cards': [
            {
                'name': 'Dock',
                'color': 'harbour',
                'age': 1,
                'at_players': [3, 4, 5, 6, 7],
                'cost': {'coins': 0, 'resources': {'wood': 1, 'ore': 0}},
                'effects': [{'type': 'production', 'resources': ['wood'], 'description': ''}],
                'chain_enables': [],
            }
        ],
        'wonders': [_side('A'), _side('B')],
    }
    project.update(over)
    return project


def test_save_load_play_and_delete_a_project(isolated):  # noqa: F811
    result = builder.save_project(_project())
    assert result['check']['ok'] and result['check']['loaded']
    folder = isolated / 'home' / 'docks'
    data = json.loads((folder / 'cards.json').read_text())
    assert data['cards'][0] == {
        'name': 'Dock',
        'color': 'harbour',
        'age': 1,
        'at_players': [3, 4, 5, 6, 7],
        'cost': {'resources': {'wood': 1}},
        'effects': [{'type': 'production', 'resources': ['wood']}],
    }  # empty values are dropped
    manifest = json.loads((folder / 'manifest.json').read_text())
    assert manifest['expansions'] == [{'id': 'docks', 'name': 'Docks', 'file': 'cards.json'}]
    assert extensions.validate(folder) == []
    assert builder.load_project('docks')['rules'] == {'starting_coins': {'add': 2}, 'discard_coins': 2}
    assert Ruleset.of('docks').rules.changed() == {'starting_coins': 5, 'discard_coins': 2}
    assert Ruleset.of('docks', 'leaders').starting_coins == 8  # the delta adds to Leaders' 6
    assert load_content('docks').card('Dock').color is Color('harbour')
    assert [p['id'] for p in builder.projects()] == ['docks']

    report = builder.balance('docks', games=4, players=3)
    assert report['cards'][0]['name'] == 'Dock' and report['wonders'][0]['name'] == 'The Lighthouse'
    assert set(report['base']) >= {'total', 'military'} and report['expected_win_rate'] == 0.33

    names = zipfile.ZipFile(io.BytesIO(builder.export_project('docks'))).namelist()
    assert sorted(names) == ['docks/cards.json', 'docks/manifest.json']
    builder.delete_project('docks')
    assert not folder.exists() and 'docks' not in EXPANSIONS


@pytest.mark.filterwarnings('ignore:extension folder')
def test_a_project_with_errors_is_saved_but_not_loaded(isolated):  # noqa: F811
    bad = _project(cards=[{'name': 'Broken', 'color': 'nope', 'age': 1, 'at_players': [3]}])
    result = builder.save_project(bad)
    assert not result['check']['ok'] and not result['check']['loaded']
    assert (isolated / 'home' / 'docks' / 'cards.json').is_file() and 'docks' not in EXPANSIONS
    with pytest.raises(builder.BuilderError, match='exists already'):
        builder.save_project(_project())  # a new project never overwrites another
    assert builder.save_project(_project(), previous_id='docks')['check']['loaded']  # fixed: now it loads


def test_renaming_a_project_moves_its_folder(isolated):  # noqa: F811
    builder.save_project(_project())
    builder.save_project(_project(id='harbours'), previous_id='docks')
    assert [p['id'] for p in builder.projects()] == ['harbours']
    assert 'docks' not in EXPANSIONS and 'harbours' in EXPANSIONS


def test_check_pins_problems_to_items(isolated):  # noqa: F811
    project = _project(
        id='Bad Id',
        rules={'discard_coins': 99},
        colors=[{'id': 'brown', 'fill': '#000000'}],
        cards=[
            {
                'name': 'Altar',
                'color': 'blue',
                'age': 1,
                'at_players': [3],
                'effects': [{'type': 'coins', 'amount': 1}],
            },
            {'name': 'Odd', 'color': 'blue', 'age': 1, 'at_players': [2], 'effects': [{'type': 'nope'}]},
            {'name': 'Link', 'color': 'blue', 'age': 1, 'at_players': [3], 'chain_requires': 'Nowhere'},
            {'name': 'Quiet', 'color': 'blue', 'age': 1},
        ],
        wonders=[_side('A', 'Half'), _side('A', 'The Pyramids of Giza'), _side('B', 'The Pyramids of Giza')],
    )
    report = builder.check_project(project)
    found: Dict[Any, str] = {}
    for p in report['problems']:
        key = (p['section'], p['index'], p['level'])
        found[key] = found.get(key, '') + p['message'] + '; '
    assert 'bad id' in found[('general', None, 'error')]
    assert 'between' in found[('rules', None, 'error')]
    assert 'already defined' in found[('colors', 0, 'error')]
    assert 'clashes' in found[('cards', 0, 'error')]
    assert 'nope' in found[('cards', 1, 'error')]
    assert 'Nowhere' in found[('cards', 2, 'error')]
    assert 'never dealt' in found[('cards', 3, 'warning')]
    assert 'lacks side B' in found[('wonders', 0, 'error')]
    assert 'clashes' in found[('wonders', 1, 'error')]
    assert not report['ok'] and report['errors'] >= 8


def test_previews_render_unsaved_specs(isolated):  # noqa: F811
    project = _project()
    svg = builder.preview('card', project['cards'][0], project['colors'])
    assert svg.startswith('<svg') and '#1f8a8a' in svg
    # Editing a colour updates it in place while the builder is open.
    recoloured = [{**project['colors'][0], 'fill': '#aa0000'}]
    assert '#aa0000' in builder.preview('card', project['cards'][0], recoloured)
    assert builder.preview('wonder', _side('A')).startswith('<svg')
    with pytest.raises(ValueError):
        builder.preview('card', {'name': 'x', 'color': 'blue', 'effects': [{'type': 'nope'}]})


def test_bad_ids_cannot_escape_the_home(isolated):  # noqa: F811
    for bad in ('../x', 'a/b', '', 'UP'):
        with pytest.raises(builder.BuilderError):
            builder.load_project(bad)


def test_a_broken_folder_is_not_left_half_loaded(isolated):  # noqa: F811
    folder = extensions.scaffold(isolated / 'home' / 'broken', 'broken')
    data = json.loads((folder / 'cards.json').read_text())
    data['cards'][0]['effects'] = [{'type': 'no_such_effect'}]
    (folder / 'cards.json').write_text(json.dumps(data))
    with pytest.warns(UserWarning, match='no_such_effect'):
        extensions.ensure_loaded()
    assert 'broken' not in EXPANSIONS and 'broken' not in extensions.REGISTRY.extensions


# -- server -----------------------------------------------------------------------------------
@pytest.fixture
def builder_url(isolated) -> Iterator[str]:  # noqa: F811
    httpd = make_server(port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{httpd.server_address[1]}'
    finally:
        httpd.shutdown()
        httpd.server_close()


def _call(url: str, body: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Any:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers or {'Content-Type': 'application/json'})
    try:
        res = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as err:
        err.msg = f'{err.msg}: {err.read().decode()}'  # the server's reason
        raise
    with res:
        payload = res.read()
        return json.loads(payload) if res.headers['Content-Type'] == 'application/json' else payload


def test_server_builds_and_offers_an_expansion(builder_url):
    state = _call(builder_url + '/api/builder')
    assert state['projects'] == [] and len(state['info']['rules']) == len(RULES)
    assert _call(builder_url + '/api/builder/check', {'project': _project()})['ok']
    preview = _call(
        builder_url + '/api/builder/preview',
        {'kind': 'card', 'spec': _project()['cards'][0], 'colors': _project()['colors']},
    )
    assert preview['svg'].startswith('<svg')
    assert 'error' in _call(builder_url + '/api/builder/preview', {'kind': 'card', 'spec': {'color': 'nope'}})
    saved = _call(builder_url + '/api/builder/save', {'project': _project()})
    assert saved['check']['loaded'] and saved['projects'][0]['id'] == 'docks'
    options = _call(builder_url + '/api/options')
    assert any(e['id'] == 'docks' for e in options['expansions']) and options['colors']['harbour']['fill'] == '#1f8a8a'
    assert _call(builder_url + '/api/builder/project?id=docks')['project']['name'] == 'Docks'
    assert _call(builder_url + '/api/card.svg?name=Dock').startswith(b'<svg')
    assert zipfile.ZipFile(io.BytesIO(_call(builder_url + '/api/builder/export?id=docks'))).namelist()
    assert _call(builder_url + '/api/builder/balance', {'id': 'docks', 'games': 2, 'players': 3})['games'] == 2
    assert _call(builder_url + '/api/builder/delete', {'id': 'docks'})['projects'] == []


def test_server_refuses_cross_site_builder_requests(builder_url):
    for headers in ({'Content-Type': 'text/plain'}, {'Content-Type': 'application/json', 'Host': 'evil.example:80'}):
        with pytest.raises(urllib.error.HTTPError) as err:
            _call(builder_url + '/api/builder/save', {'project': _project()}, headers)
        assert err.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as err:
        _call(builder_url + '/api/builder', headers={'Host': 'evil.example'})
    assert err.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as err:
        _call(builder_url + '/api/builder/save', {'project': _project(id='../up')})
    assert err.value.code == 400
