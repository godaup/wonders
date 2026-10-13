"""Tests for the catalog: community extensions shipped with the game, searched and switched on and off."""

import json
import random
import shutil
import threading
import urllib.error
from pathlib import Path
from typing import Iterator

import pytest
from click.testing import CliRunner

import wonders.extensions as extensions
from tests.test_builder import _call
from tests.test_extensions import isolated  # noqa: F401 - fixture
from wonders import cli
from wonders.agents import GreedyAgent
from wonders.agents.match import play_match
from wonders.cards.content import load_content
from wonders.engine.record import GameRecord, PlayerSetup
from wonders.play.server import make_server
from wonders.ruleset import EXPANSIONS

REPOSITORY = Path(__file__).resolve().parent.parent / 'extensions'


@pytest.fixture
def shelf(isolated):  # noqa: F811
    """A catalog holding two scaffolded extensions."""
    catalog = isolated / 'catalog'
    extensions.scaffold(catalog / 'harbour', 'harbour', 'Harbour Town', 'Ada')
    extensions.scaffold(catalog / 'mines', 'mines', 'Deep Mines', 'Bo')
    return catalog


def test_the_repository_extensions_are_the_default_catalog(monkeypatch):
    monkeypatch.delenv(extensions.CATALOG_VARIABLE, raising=False)
    homes = extensions.catalog_homes()  # a checkout reads extensions/, an installed package its copy
    assert homes in ([REPOSITORY], [Path(extensions.__file__).resolve().parent / 'catalog'])
    entry = next(e for e in extensions.catalog() if e.manifest.id == 'rinascita')
    assert entry.cards == 23 and 'Uraniborg' in entry.wonders and entry.colors == ['Arts']
    assert [e.manifest.id for e in extensions.catalog('plague COLUMN')] == ['rinascita']  # every word, any case


def test_the_catalog_is_searched_by_name_author_and_card(shelf):
    assert [e.manifest.id for e in extensions.catalog()] == ['mines', 'harbour']  # by name
    assert [e.manifest.id for e in extensions.catalog('ada')] == ['harbour']
    assert [e.manifest.id for e in extensions.catalog('deep workshop')] == ['mines']  # a card name matches too
    assert extensions.catalog('harbour mines') == []
    assert not any(e.active or e.loaded for e in extensions.catalog())


def test_switching_on_loads_now_and_at_every_start(shelf):
    entry = extensions.activate('harbour')
    assert entry.active and entry.loaded == 'catalog' and 'harbour' in EXPANSIONS and 'mines' not in EXPANSIONS
    assert extensions.active_ids() == ['harbour']
    # A new start: discovery loads what is switched on, nothing else.
    extensions.unload('harbour')
    extensions.REGISTRY = extensions.Registry()
    extensions._LOADED = False
    registry = extensions.ensure_loaded()
    assert registry.extensions['harbour'].source == 'catalog' and 'mines' not in registry.extensions
    off = extensions.activate('harbour', False)
    assert not off.active and not off.loaded and 'harbour' not in EXPANSIONS and extensions.active_ids() == []
    with pytest.raises(ValueError, match='no extension'):
        extensions.activate('nope')


def test_your_own_folder_keeps_precedence(shelf, isolated):  # noqa: F811
    shutil.copytree(shelf / 'harbour', isolated / 'home' / 'harbour')
    entry = extensions.activate('harbour')
    assert entry.active and entry.loaded == 'folder'
    assert extensions.REGISTRY.extensions['harbour'].location == str(isolated / 'home' / 'harbour')
    extensions.activate('harbour', False)  # switching off never unloads your own folder
    assert 'harbour' in EXPANSIONS


def test_a_broken_extension_stays_off(shelf):
    data = json.loads((shelf / 'mines' / 'cards.json').read_text())
    data['cards'][0]['effects'] = [{'type': 'no_such_effect'}]
    (shelf / 'mines' / 'cards.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='failed to load'):
        extensions.activate('mines')
    assert extensions.active_ids() == [] and 'mines' not in EXPANSIONS


def test_a_record_loads_its_catalog_expansion_while_switched_off(shelf):
    extensions.activate('harbour')
    expansions = ('base', 'harbour')
    boards = ['The Pyramids of Giza', 'The Colossus of Rhodes', 'The Temple of Artemis in Ephesus']
    setups = [PlayerSetup(n, w, 'A') for n, w in zip('ABC', boards)]
    game, record = play_match(
        load_content(*expansions), setups, [GreedyAgent() for _ in setups], random.Random(3), expansions
    )
    extensions.activate('harbour', False)
    assert 'harbour' not in EXPANSIONS
    again = GameRecord.from_json(record.to_json())
    content = load_content(*again.expansions)  # the catalog provides it again, without switching it on
    assert 'harbour' in EXPANSIONS and extensions.active_ids() == [] and again.check_extensions() == []
    assert [again.replay(content).score(i).total for i in range(3)] == [game.score(i).total for i in range(3)]


def test_cli_searches_and_switches(shelf):
    runner = CliRunner()
    found = runner.invoke(cli.main, ['ext', 'catalog', 'harbour'])
    assert found.exit_code == 0 and 'harbour 0.1.0 [off] Harbour Town by Ada' in found.output
    assert runner.invoke(cli.main, ['ext', 'catalog', 'nothing-like-it']).exit_code != 0
    on = runner.invoke(cli.main, ['ext', 'enable', 'harbour'])
    assert on.exit_code == 0 and 'harbour is on' in on.output and extensions.active_ids() == ['harbour']
    assert '[on]' in runner.invoke(cli.main, ['ext', 'catalog']).output
    assert runner.invoke(cli.main, ['ext', 'disable', 'harbour']).exit_code == 0 and extensions.active_ids() == []
    assert runner.invoke(cli.main, ['ext', 'enable', 'nope']).exit_code != 0


@pytest.fixture
def catalog_url(shelf) -> Iterator[str]:
    httpd = make_server(port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{httpd.server_address[1]}'
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_server_searches_and_switches(catalog_url):
    found = _call(catalog_url + '/api/catalog?q=deep')['extensions']
    assert [x['id'] for x in found] == ['mines'] and found[0]['cards'] == 1 and not found[0]['active']
    assert not any(e['id'] == 'mines' for e in _call(catalog_url + '/api/options')['expansions'])
    on = _call(catalog_url + '/api/catalog/activate', {'id': 'mines', 'active': True})['extension']
    assert on['active'] and on['loaded'] == 'catalog'
    assert any(e['id'] == 'mines' for e in _call(catalog_url + '/api/options')['expansions'])  # the menu offers it
    _call(catalog_url + '/api/catalog/activate', {'id': 'mines', 'active': False})
    assert not any(e['id'] == 'mines' for e in _call(catalog_url + '/api/options')['expansions'])
    with pytest.raises(urllib.error.HTTPError, match='no extension'):
        _call(catalog_url + '/api/catalog/activate', {'id': 'nope'})
    with pytest.raises(urllib.error.HTTPError, match='only local JSON'):  # no cross-site form posts
        _call(catalog_url + '/api/catalog/activate', {'id': 'mines'}, headers={'Content-Type': 'text/plain'})
