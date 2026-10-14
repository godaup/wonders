"""Translations: the catalogs are consistent, and everything the game shows has a German translation."""

import ast
import json
import re
import string
from importlib import resources
from pathlib import Path
from typing import List, Set

import pytest

from wonders import extensions
from wonders.agents.bench import CATEGORIES
from wonders.cards.content import load_content
from wonders.i18n import Translator, catalog, english, languages, msg
from wonders.play.achievements import ACHIEVEMENTS
from wonders.play.builder import builder_info
from wonders.play.session import ARMADA_CATEGORIES, GameConfig, HumanSession, SeatConfig, menu_options
from wonders.viz.render import card_svg, gallery_html, wonder_svg

ROOT = Path(__file__).resolve().parent.parent
QUOTED = r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\""


def _fields(text: str) -> List[str]:
    return sorted(name for _, name, _, _ in string.Formatter().parse(text) if name)


def _bundled() -> List[str]:
    return sorted(p.name[:-5] for p in resources.files('wonders.i18n').iterdir() if p.name.endswith('.json'))


@pytest.mark.parametrize('lang', _bundled())
def test_bundled_catalogs_are_well_formed(lang: str) -> None:
    text = resources.files('wonders.i18n').joinpath(f'{lang}.json').read_text(encoding='utf-8')
    pairs = json.loads(text, object_pairs_hook=list)
    keys = [k for k, _ in pairs]
    assert len(keys) == len(set(keys)), sorted({k for k in keys if keys.count(k) > 1})
    for key, value in pairs:
        if key.startswith('_'):
            continue
        assert value.strip(), key
        assert _fields(key.split('|')[0]) == _fields(value), key  # same {fields}, or formatting breaks
    assert dict(pairs)['_language']


def test_translator_formats_numbers_contexts_and_messages() -> None:
    de = Translator('de')
    assert de('{n} Shields', n=3) == '3 Schilde'
    assert de('{vp:.1f} {category}', vp=2.5, category='Militär') == '2,5 Militär'  # decimal comma
    assert de('Loom') == 'Webstuhl' and de('Loom', 'resource') == 'Stoff'  # one English word, two meanings
    assert de('Not in any catalog', 'context') == 'Not in any catalog'  # falls back to English
    message = {
        'join': '; ',
        'items': [
            msg('gains {items}', items=[msg('{vp:.1f} {category}', vp=1.0, category=msg('science', 'category'))])
        ],
    }
    assert english(message) == 'gains 1.0 science'
    assert de.render(message) == 'bringt 1,0 Wissenschaft'
    assert (
        de.render(msg('{card} (to {name})', card=msg('Library'), name='Library')) == 'Bibliothek (an Library)'
    )  # plain names stay
    assert Translator('xx')('Coins') == 'Coins'
    assert languages()['de'] == 'Deutsch' and next(iter(languages())) == 'en'


def test_every_card_board_session_and_menu_text_has_a_german_translation() -> None:
    missing: Set[str] = set()
    t = Translator('de', missing=missing)
    content = load_content('base', 'leaders', 'cities', 'wonderpack', 'armada')
    second = load_content('base2', 'leaders2', 'cities2', 'armada2')
    for card in content.choices() + second.choices():
        card_svg(card, t)
    for wonder in content.wonders + second.wonders:
        wonder_svg(wonder, 1, t)
    gallery_html(content.cards[:1], content.wonders[:1], t=t)
    menu_options((), t)
    builder_info(t)  # the expansion builder's effect, rule and choice labels
    for achievement in ACHIEVEMENTS:
        t(achievement.name)
        t(achievement.description)
    for expansions, edition in (
        (['base'], 1),
        (['base', 'leaders', 'cities'], 1),
        (['base', 'leaders', 'wonderpack'], 1),
        (['base'], 2),
        (['leaders', 'cities'], 2),
        (['armada', 'cities'], 2),
        (['armada'], 1),
    ):
        seats = [SeatConfig(), SeatConfig('greedy'), SeatConfig('heuristic')]
        session = HumanSession(GameConfig(expansions=expansions, seats=seats, seed=3, edition=edition), t=t)
        session.state(t)
        for name in session.cards:
            session.card_info(name, t)
        while not session.done:
            actions = session.state(t)['observation']['actions']
            session.act(next((a['index'] for a in actions if a.get('naval')), 0))  # Armada: sail when possible
        session.state(t)
    assert not missing, sorted(missing)
    svg = card_svg(content.card('Library'), Translator('de'))
    assert 'Bibliothek' in svg and 'WIRKUNG' in svg


def test_post_game_analysis_messages_have_a_german_translation() -> None:
    tree = ast.parse((ROOT / 'wonders' / 'ml' / 'analysis.py').read_text(encoding='utf-8'))
    choices = {  # title = 'a' if ... else 'b'; msg(title, ...)
        node.targets[0].id: [node.value.body, node.value.orelse]
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.IfExp) and isinstance(node.targets[0], ast.Name)
    }
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'msg' and node.args:
            first = node.args[0]
            options = choices.get(first.id, []) if isinstance(first, ast.Name) else [first]
            context = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else None
            for option in options:
                if isinstance(option, ast.Constant) and isinstance(option.value, str) and context is None:
                    keys.add(option.value)
    # The category names are a literal dict: read it from the source, as importing wonders.ml needs PyTorch.
    names = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) == 'CATEGORY_NAMES'
    )
    keys |= {f'{name}|category' for name in names.values()} | {f'{s}|science' for s in ('tablet', 'gear', 'compass')}
    assert len(keys) > 25
    assert sorted(keys - set(catalog('de'))) == []


def _gui_keys() -> Set[str]:
    html = (ROOT / 'wonders' / 'play' / 'static' / 'index.html').read_text(encoding='utf-8')
    markup, script = html[: html.index('<script>')], html[html.index('<script>') :]
    keys: Set[str] = set()
    call = re.compile(rf"\b(?:t|tn|N)\(\s*(?:[^'\"()]*?\?\s*)?({QUOTED})(?:\s*:\s*({QUOTED}))?")
    for match in call.finditer(script):
        keys |= {ast.literal_eval(g) for g in match.groups() if g}
    keys |= set(re.findall(r'data-i18n[^>]*>([^<]+)<', markup))
    for names, rest in re.findall(r'data-i18n-attrs="([^"]+)"([^>]*)>', markup):
        keys |= {re.search(f'{a}="([^"]*)"', rest).group(1) for a in names.split(',')}  # type: ignore[union-attr]
    # Words looked up in a context: short colour groups, ratings, move kinds, battles, score columns.
    short = re.search(r'const SHORT_GROUPS = \{(.*?)\};', script, re.S).group(1)  # type: ignore[union-attr]
    groups = {ast.literal_eval(s) for s in re.findall(rf"N\(({QUOTED})\)", short)}
    keys = (keys - groups) | {f'{g}|short' for g in groups} | {'Other|short'}
    order = re.search(r'const LABEL_ORDER = \[(.*?)\];', script).group(1)  # type: ignore[union-attr]
    keys |= {f'{ast.literal_eval(s)}|rating' for s in re.findall(QUOTED, order)}
    keys |= {f'{k}|kind' for k in ('build', 'wonder', 'discard', 'free')}
    keys |= {f'{r}|military' for r in ('win', 'tie', 'loss')}
    keys |= {f'{c.capitalize()}|score' for c in CATEGORIES + ARMADA_CATEGORIES} | {'Wonder|action', 'you'}
    return keys


def test_every_gui_text_has_a_german_translation() -> None:
    keys = _gui_keys()
    assert len(keys) > 200
    assert sorted(k for k in keys - set(catalog('de')) if k.strip() not in '◐−+') == []


def test_an_extension_adds_a_language(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = extensions.Registry()
    monkeypatch.setattr(extensions, 'REGISTRY', registry)
    monkeypatch.setattr(extensions, '_LOADED', True)
    registry.register_locale('fr', {'_language': 'Français', 'Coins': 'Pièces'})
    registry.register_locale('de', {'Coins': 'Taler'})  # extends (and overrides) the bundled German
    assert languages()['fr'] == 'Français'
    assert Translator('fr')('Coins') == 'Pièces'
    assert Translator('de')('Coins') == 'Taler' and Translator('de')('Shields') == 'Schilde'
