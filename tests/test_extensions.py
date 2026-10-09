"""Tests for the extension mechanism (wonders.extensions)."""

import json
import random

import pytest
from click.testing import CliRunner

import wonders.extensions as extensions
from wonders import cli
from wonders.agents.bench import AGENTS, AgentSpec
from wonders.agents.match import play_match
from wonders.cards.content import EFFECT_BUILDERS, load_content
from wonders.effects.library import VictoryPoints
from wonders.engine.record import GameRecord, PlayerSetup
from wonders.ruleset import EXPANSIONS, Ruleset


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """A fresh registry and extensions home; every global the extensions touch is restored afterwards."""
    snapshot = (dict(EXPANSIONS), dict(EFFECT_BUILDERS), dict(AGENTS))
    monkeypatch.setattr(extensions, 'REGISTRY', extensions.Registry())
    monkeypatch.setattr(extensions, '_LOADED', False)
    monkeypatch.setenv(extensions.HOME_VARIABLE, str(tmp_path / 'home'))
    monkeypatch.delenv(extensions.DISABLE_VARIABLE, raising=False)
    yield tmp_path
    for live, saved in zip((EXPANSIONS, EFFECT_BUILDERS, AGENTS), snapshot):
        live.clear()
        live.update(saved)


def _cards(folder):
    return json.loads((folder / 'cards.json').read_text())


def test_scaffold_is_valid_and_loads_from_the_home(isolated):
    home = isolated / 'home'
    folder = extensions.scaffold(home / 'mini', 'mini', 'Mini')
    assert extensions.validate(folder) == []
    registry = extensions.ensure_loaded()
    assert 'mini' in registry.extensions and 'mini' in EXPANSIONS
    ext = registry.extensions['mini']
    assert ext.source == 'folder' and len(ext.digest) == 12 and ext.provides == ['expansion:mini']
    content = load_content('base', 'mini')
    assert content.card('Mini Workshop').expansion == 'mini'
    assert Ruleset.of('mini').expansions == ('base', 'mini')


def test_validate_reports_problems(isolated):
    folder = extensions.scaffold(isolated / 'bad', 'bad')
    data = _cards(folder)
    data['cards'] += [
        {'name': 'Altar', 'color': 'blue', 'age': 1, 'effects': []},  # clashes with a base card
        {'name': 'Odd', 'color': 'blue', 'age': 1, 'effects': [{'type': 'no_such_effect'}]},
        {'name': 'Linked', 'color': 'blue', 'age': 1, 'chain_requires': 'Nowhere'},
    ]
    (folder / 'cards.json').write_text(json.dumps(data))
    problems = extensions.validate(folder)
    assert any('Altar' in p and 'clashes' in p for p in problems)
    assert any('no_such_effect' in p for p in problems)
    assert any('Nowhere' in p for p in problems)
    (folder / 'manifest.json').write_text('{"name": "no id"}')
    assert extensions.validate(folder) == ['manifest.json: manifest lacks id']


def test_games_with_an_extension_expansion_pin_it_in_the_record(isolated):
    folder = extensions.scaffold(isolated / 'home' / 'mini', 'mini')
    extensions.ensure_loaded()
    expansions = ('base', 'mini')
    content = load_content(*expansions)
    setups = [
        PlayerSetup(n, w, 'A')
        for n, w in zip('ABC', ['The Pyramids of Giza', 'The Colossus of Rhodes', 'The Temple of Artemis in Ephesus'])
    ]
    from wonders.agents import GreedyAgent

    game, record = play_match(content, setups, [GreedyAgent() for _ in setups], random.Random(1), expansions)
    pin = record.extensions['mini']
    assert pin['hash'] == extensions.REGISTRY.extensions['mini'].digest
    again = GameRecord.from_json(record.to_json())
    assert again.extensions == record.extensions and again.check_extensions() == []
    assert [again.replay(content).score(i).total for i in range(3)] == [game.score(i).total for i in range(3)]
    data = _cards(folder)
    data['cards'][0]['effects'][0]['points'] = 3  # the extension changes after the game
    (folder / 'cards.json').write_text(json.dumps(data))
    installed = extensions.REGISTRY.extensions['mini']
    installed.digest = extensions._digest(folder, installed.manifest)  # as a fresh install would see it
    assert 'differs' in again.check_extensions()[0]
    with pytest.warns(UserWarning, match='replay may differ'):
        again.replay(content)


def test_plugins_add_effects_and_agents(isolated):
    from wonders.agents import GreedyAgent

    def register(registry):
        registry.register_effect('double_points', lambda spec: VictoryPoints(2 * spec['points']))
        registry.register_agent(AgentSpec.of('greedy-too', GreedyAgent), blurb='Greedy, from a plugin.')

    ext = extensions.load_plugin(extensions.Manifest('demo', 'Demo', version='1.2'), register)
    assert ext.provides == ['effect:double_points', 'agent:greedy-too']
    assert 'greedy-too' in AGENTS and extensions.REGISTRY.agent_blurbs['greedy-too']
    folder = extensions.scaffold(isolated / 'uses-plugin', 'uses')
    data = _cards(folder)
    data['cards'][0]['effects'] = [{'type': 'double_points', 'points': 2, 'description': '4 VP (doubled)'}]
    (folder / 'cards.json').write_text(json.dumps(data))
    assert extensions.validate(folder) == []
    with pytest.raises(ValueError):
        extensions.REGISTRY.register_effect('double_points', lambda spec: None)  # names are unique
    with pytest.raises(ValueError):
        extensions.load_plugin(extensions.Manifest('demo', 'Again'), register)


def test_models_and_locales_from_folders(isolated):
    folder = isolated / 'home' / 'pack'
    folder.mkdir(parents=True)
    (folder / 'net.pt').write_bytes(b'not really a checkpoint')
    (folder / 'de.json').write_text(json.dumps({'menu.start': 'Spiel starten'}))
    (folder / 'manifest.json').write_text(
        json.dumps(
            {
                'id': 'pack',
                'name': 'Pack',
                'models': [{'name': 'mynet', 'file': 'net.pt', 'description': 'A test model'}],
                'locales': [{'lang': 'de', 'file': 'de.json'}],
            }
        )
    )
    registry = extensions.ensure_loaded()
    assert registry.models['mynet'].path.endswith('net.pt') and registry.models['mynet'].extension == 'pack'
    assert registry.locales['de']['menu.start'] == 'Spiel starten'


def test_broken_folders_warn_instead_of_failing(isolated):
    folder = isolated / 'home' / 'broken'
    folder.mkdir(parents=True)
    (folder / 'manifest.json').write_text('{not json')
    with pytest.warns(UserWarning, match='failed to load'):
        extensions.ensure_loaded()


def test_cli_new_validate_list(isolated):
    runner = CliRunner()
    made = runner.invoke(cli.main, ['ext', 'new', 'cli-demo', '--name', 'CLI Demo'])
    assert made.exit_code == 0, made.output
    folder = isolated / 'home' / 'cli-demo'
    checked = runner.invoke(cli.main, ['ext', 'validate', str(folder)])
    assert checked.exit_code == 0 and 'valid extension' in checked.output
    listed = runner.invoke(cli.main, ['ext', 'list'])
    assert listed.exit_code == 0 and 'cli-demo 0.1.0 [folder] CLI Demo' in listed.output
    assert 'expansion:cli-demo' in listed.output
    (folder / 'cards.json').write_text('{}')
    data = json.loads((folder / 'manifest.json').read_text())
    data['expansions'][0]['file'] = 'missing.json'
    (folder / 'manifest.json').write_text(json.dumps(data))
    failed = runner.invoke(cli.main, ['ext', 'validate', str(folder)])
    assert failed.exit_code != 0 and 'missing file' in failed.output


def test_gui_menu_and_session_with_an_extension(isolated):
    from wonders.play.session import GameConfig, HumanSession, SeatConfig, menu_options

    extensions.scaffold(isolated / 'home' / 'mini', 'mini', 'Mini')
    assert any(e['id'] == 'mini' and e['implemented'] for e in menu_options()['expansions'])
    seats = [SeatConfig(), SeatConfig('greedy'), SeatConfig('heuristic')]
    session = HumanSession(GameConfig(expansions=['base', 'mini'], seats=seats, seed=2))
    while not session.done:
        session.act(0)
    assert 'mini' in json.loads(session.record_json())['extensions']
