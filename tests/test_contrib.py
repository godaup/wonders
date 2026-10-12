"""Tests for the contribution checks (wonders.contrib) behind the contribution workflows."""

import json
import os
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

import wonders.contrib as contrib
import wonders.extensions as extensions
from wonders import cli
from wonders.agents.bench import AGENTS
from wonders.cards.base import Color
from wonders.cards.content import EFFECT_BUILDERS
from wonders.ruleset import EXPANSIONS

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """A fresh registry; every global the extensions touch is restored afterwards."""
    snapshot = (dict(EXPANSIONS), dict(EFFECT_BUILDERS), dict(AGENTS))
    registry = extensions.Registry()
    monkeypatch.setattr(Color, '_members', dict(Color._members))  # colours the extensions define
    monkeypatch.setattr(extensions, 'REGISTRY', registry)
    monkeypatch.setattr(contrib, 'REGISTRY', registry)
    monkeypatch.setattr(extensions, '_LOADED', False)
    monkeypatch.setenv(extensions.HOME_VARIABLE, str(tmp_path / 'home'))
    monkeypatch.delenv(extensions.DISABLE_VARIABLE, raising=False)
    yield tmp_path
    for live, saved in zip((EXPANSIONS, EFFECT_BUILDERS, AGENTS), snapshot):
        live.clear()
        live.update(saved)


def _extension(parent: Path, ext_id: str = 'mini') -> Path:
    return extensions.scaffold(parent / 'extensions' / ext_id, ext_id, ext_id.title(), author='tester')


def test_changed_maps_paths_to_contributions(tmp_path):
    folder = _extension(tmp_path, 'netpack')
    manifest = json.loads((folder / 'manifest.json').read_text())
    manifest['models'] = [{'name': 'net', 'file': 'net.pt'}, {'name': 'other', 'file': 'other.pt'}]
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    (tmp_path / 'plugins' / 'tool').mkdir(parents=True)
    (tmp_path / 'wonders' / 'i18n').mkdir(parents=True)
    (tmp_path / 'wonders' / 'i18n' / 'fr.json').write_text('{}')
    paths = [
        'extensions/netpack/net.pt',
        'extensions/netpack/README.md',
        'extensions/gone/manifest.json',  # deleted: nothing to check
        'extensions/Bad Name/x.json',
        'plugins/tool/tool/__init__.py',
        'wonders/i18n/fr.json',
        'wonders/i18n/README.md',
        'wonders/cli.py',
        'README.md',
    ]
    result = contrib.changed(paths, tmp_path)
    assert result['extensions'] == ['extensions/netpack']
    assert result['plugins'] == ['tool']
    assert result['locales'] == ['wonders/i18n/fr.json']
    assert result['models'] == [{'extension': 'netpack', 'model': 'net'}]  # only the changed checkpoint
    assert result['invalid'] == ['extensions/Bad Name/x.json'] and result['any']
    assert contrib.changed(['extensions/netpack/manifest.json'], tmp_path)['models'] == [
        {'extension': 'netpack', 'model': 'net'},
        {'extension': 'netpack', 'model': 'other'},
    ]
    assert not contrib.changed(['wonders/engine/game.py'], tmp_path)['any']


def test_changed_cli_writes_github_output_and_rejects_bad_names(tmp_path):
    output = tmp_path / 'out.txt'
    runner = CliRunner()
    ok = runner.invoke(cli.main, ['ext', 'changed', '--root', str(tmp_path), '--github-output', str(output)], input='')
    assert ok.exit_code == 0, ok.output
    assert 'models=[]' in output.read_text() and 'any=false' in output.read_text()
    bad = runner.invoke(cli.main, ['ext', 'changed', '--root', str(tmp_path), 'plugins/$(x)/a.py'])
    assert bad.exit_code == 1 and 'names must be' in bad.output


def test_a_scaffolded_extension_passes_and_games_are_played(isolated):
    folder = _extension(isolated)
    report = contrib.check(folder, games=1)
    assert report.passed, report.markdown()
    texts = [text for _, text in report.findings]
    assert any('leaders+cities+mini/7p: 1 game(s) played and replayed' in t for t in texts)
    assert any(level == 'warning' and 'hands of 8 instead of 7' in t for level, t in report.findings)  # one per player
    assert report.markdown().startswith('#### ✅ extension')


def test_an_extension_that_leaves_cards_undealt_fails(isolated):
    folder = _extension(isolated)
    data = json.loads((folder / 'cards.json').read_text())
    data['cards'][0]['at_players'] = [3]  # one copy at every count: a card left over at 4-7 players
    data['rules'] = {'extra_guilds': {'add': 1}}
    (folder / 'cards.json').write_text(json.dumps(data))
    report = contrib.check(folder, games=1)
    assert not report.passed
    (error,) = [t for level, t in report.findings if level == 'error']
    assert 'left over when dealing' in error and 'Age I at 4 players' in error and 'Age III at 3 players' in error
    assert report.markdown().startswith('#### ❌ extension')


def test_extension_check_reports_contribution_problems(isolated):
    folder = _extension(isolated)
    manifest = json.loads((folder / 'manifest.json').read_text())
    del manifest['rules_version']
    manifest['author'] = ''
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    (folder / 'helper.py').write_text('print(1)')
    other = _extension(isolated, 'other')
    other_manifest = json.loads((other / 'manifest.json').read_text())
    other_manifest['expansions'][0]['id'] = 'mini'
    (other / 'manifest.json').write_text(json.dumps(other_manifest))
    renamed = isolated / 'extensions' / 'renamed'
    folder.rename(renamed)
    errors = contrib.check_extension(renamed).errors
    assert any('lacks author' in e for e in errors)
    assert any('rules_version' in e for e in errors)
    assert any('helper.py' in e and 'data only' in e for e in errors)
    assert any('named after the manifest id' in e for e in errors)
    assert any('mini also used by other' in e for e in errors)


def test_locale_check(tmp_path):
    good = tmp_path / 'fr.json'
    good.write_text(json.dumps({'_language': 'Français', '{n} Shields': '{n} boucliers', 'Coins': 'Pièces'}))
    report = contrib.check_locale(good)
    assert report.passed and report.title == 'Français (fr)'
    assert any(level == 'warning' and 'not translated yet' in text for level, text in report.findings)
    bad = tmp_path / 'french.json'
    bad.write_text('{"{n} Shields": "{count} boucliers", "Coins": "", "Not a game text": "x"}')
    report = contrib.check_locale(bad)
    errors = report.errors
    assert any('language code' in e for e in errors)
    assert any('_language' in e for e in errors)
    assert any('{fields}' in e for e in errors) and any('empty translation' in e for e in errors)
    assert any('does not show' in text for _, text in report.findings)
    assert contrib.check_locale(ROOT / 'wonders' / 'i18n' / 'de.json').passed


def test_example_plugin_registers_and_its_content_plays(isolated, monkeypatch):
    monkeypatch.setenv(extensions.DISABLE_VARIABLE, '1')  # no entry-point discovery: we load it by hand
    monkeypatch.syspath_prepend(str(ROOT / 'plugins' / 'example'))
    import wonders_example

    try:
        info = extensions.plugin_manifest(wonders_example.register)  # MANIFEST lives in the module
        assert info is not None and info['id'] == 'example'
        ext = extensions.load_plugin(extensions.Manifest.from_dict(info), wonders_example.register)
        assert ext.provides == ['effect:tithe', 'expansion:tithe', 'agent:taxman']
        report = contrib.Report('plugin', 'example')
        contrib._check_provided(report, ext.provides, games=1)
        assert report.passed, report.markdown()
        assert any('agent taxman at leaders+cities/5p' in text for _, text in report.findings)
    finally:
        sys.modules.pop('wonders_example', None)


def test_extensions_homes_take_several_folders(monkeypatch, tmp_path):
    monkeypatch.setenv(extensions.HOME_VARIABLE, os.pathsep.join([str(tmp_path / 'a'), str(tmp_path / 'b')]))
    assert extensions.extensions_homes() == [tmp_path / 'a', tmp_path / 'b']
    assert extensions.extensions_home() == tmp_path / 'a'


def test_scorecards_merge_render_and_feed_the_leaderboard(isolated):
    pytest.importorskip('torch')
    from wonders.ml import pretrained_path

    shards = [
        contrib.scorecard(pretrained_path(), cells=cells, games=2, processes=1) for cells in ('base/3p', 'leaders/3p')
    ]
    [card] = contrib.merge_scorecards(shards)
    assert card['cells'] == ['base/3p', 'leaders/3p']
    [pair] = card['result']['pairs']
    assert pair['games'] == 4
    text = contrib.scorecard_markdown(card)
    assert 'Verdict:' in text and '| base | 2 |' in text and '| leaders | 2 |' in text
    data, markdown = isolated / 'leaderboard.json', isolated / 'LEADERBOARD.md'
    contrib.update_leaderboard(card, data, markdown, date='2026-10-09')
    contrib.update_leaderboard(card, data, markdown, date='2026-10-10')  # replaces, does not duplicate
    entries = json.loads(data.read_text())['entries']
    assert len(entries) == 1 and entries[0]['date'] == '2026-10-10' and entries[0]['games'] == 4
    assert '| 1 | `ppo_universal_v2` |' in markdown.read_text()


def test_summary_of_reports_without_scorecards_needs_no_pytorch(tmp_path):
    (tmp_path / 'locale.md').write_text('#### ✅ locale `wonders/i18n/de.json`\n', encoding='utf-8')
    text = contrib.summary(tmp_path)  # a translation-only pull request: no model, no scorecard
    assert '✅ all checks passed' in text and contrib.merge_scorecards([]) == []


def test_an_extension_locale_may_translate_its_colour_labels(isolated):
    folder = _extension(isolated)
    data = json.loads((folder / 'cards.json').read_text())
    data['colors'] = [{'id': 'arts', 'label': 'Arts', 'fill': '#c2477a'}, {'id': 'relic', 'fill': '#888888'}]
    (folder / 'cards.json').write_text(json.dumps(data))
    (folder / 'de.json').write_text(json.dumps({'Arts': 'Künste', 'Relic': 'Reliquie', 'Mini Workshop': 'Werkstatt'}))
    manifest = json.loads((folder / 'manifest.json').read_text())
    manifest['locales'] = [{'lang': 'de', 'file': 'de.json'}]
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    report = contrib.check(folder, games=1)
    assert report.passed, report.markdown()
    assert not any('does not show' in text for _, text in report.findings)  # labels are shown on the cards
