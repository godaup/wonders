"""The expansion builder behind the GUI: create and edit data extensions without writing JSON.

A *project* is one data extension folder in the extensions home (``~/.wonders/extensions/<id>/``):
a ``manifest.json`` plus one expansion file holding the expansion's colours, cards, wonder boards
and meta rules. The builder only ever writes that format — the one a person writes by hand — so
validation, the game menu, game records (with content hashes), ``wonders ext check`` and pull
requests to ``extensions/`` all work unchanged. The API exchanges a project as one flat object::

    {"id": "docks", "name": "Docks", "version": "0.1.0", "author": "", "license": "CC-BY-4.0",
     "description": "", "rules": {"discard_coins": 2}, "colors": [...], "cards": [...], "wonders": [...]}

``cards`` / ``wonders`` / ``colors`` use the expansion-file format (``docs/extensions.md``); a wonder
board is two entries of the same name, sides ``A`` and ``B``.
"""

from __future__ import annotations

import io
import json
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from wonders import extensions
from wonders.cards.base import OFFICIAL_COLORS, Card, Color, Resource, ScienceSymbol
from wonders.cards.content import Content, build_age_deck, build_card, build_wonder, load_content
from wonders.cards.schema import describe
from wonders.extensions import MANIFEST, extensions_home, reload_folder, unload
from wonders.i18n import ENGLISH, Translator
from wonders.ruleset import EXPANSIONS, RULES, RULES_VERSION, check_rules

EXPANSION_FILE = 'cards.json'
META_KEYS = ('name', 'version', 'author', 'license', 'description')
CARD_KEYS = ('name', 'color', 'age', 'guild', 'at_players', 'cost', 'chain_requires', 'chain_enables', 'effects')
WONDER_KEYS = ('name', 'side', 'starting_resource', 'stages', 'effects')
BALANCE_LINEUP = ('heuristic', 'board', 'greedy', 'builder')
MAX_BALANCE_GAMES = 400

# Readable names of the choice values the effect forms offer (keyed by parameter name and value).
CHOICE_LABELS = {
    'scope': {
        'self': 'you',
        'neighbours': 'your neighbours',
        'both': 'you and your neighbours',
        'others': 'all others',
    },
    'side': {'left': 'left neighbour', 'right': 'right neighbour', 'both': 'both neighbours'},
    'category': {'raw': 'raw materials', 'manufactured': 'manufactured goods', 'both': 'all resources'},
    'what': {
        'colors': 'cards of the colours',
        'wonder_stages': 'wonder stages',
        'defeat_tokens': 'defeat tokens',
        'tokens': 'military tokens',
        'victory_tokens': 'victory tokens',
        'coins': 'coins (per 3)',
    },
}
SCORE_LABELS = {
    'civilian': 'Civilian',
    'commercial': 'Commercial',
    'guilds': 'Guilds',
    'leaders': 'Leaders',
    'cities': 'Cities',
}


class BuilderError(ValueError):
    """A builder request that cannot be carried out."""


# --- what the forms offer --------------------------------------------------------------------


def builder_info(t: Translator = ENGLISH) -> Dict[str, Any]:
    """Everything the builder page needs for its forms: effect schemas, meta rules, ids, known names."""
    effects = []
    for entry in describe():
        schema = entry['schema']
        if schema is not None:
            schema['label'], schema['help'], schema['group'] = t(schema['label']), t(schema['help']), t(schema['group'])
            for param in schema['params']:
                param['label'] = t(param['label'])
                param['help'] = t(param['help']) if param['help'] else ''
                labels = CHOICE_LABELS.get(param['name'], {})
                param['choice_labels'] = {c: t(labels.get(c, c)) for c in param['choices']}
        effects.append(entry)
    known = _known_content()
    return {
        'effects': effects,
        'rules': [
            {
                'name': r.name,
                'default': r.default,
                'min': r.minimum,
                'max': r.maximum,
                'label': t(r.label),
                'help': t(r.help),
            }
            for r in RULES
        ],
        'resources': [r.value for r in Resource if r is not Resource.GOLD],
        'symbols': [s.value for s in ScienceSymbol],
        'official_colors': [c.value for c in OFFICIAL_COLORS],
        'scores': {k: t(v, 'score') for k, v in SCORE_LABELS.items()},
        'known_cards': sorted({c.name for c in known.cards}),
        'home': str(extensions_home()),
        'rules_version': RULES_VERSION,
    }


def _known_content(without: str = '') -> Content:
    """Every loaded expansion's cards and wonders, except the expansion ``without``."""
    from wonders.extensions import ensure_loaded

    ensure_loaded()
    content = Content()
    for exp in EXPANSIONS.values():
        if exp.implemented and exp.id != without:
            try:
                part = load_content(exp.id)
            except Exception:  # a broken extension must not break the builder
                continue
            content.cards += part.cards
            content.wonders += part.wonders
    return content


# --- projects on disk ------------------------------------------------------------------------


def _folder(project_id: str) -> Path:
    from wonders.contrib import NAME

    if not isinstance(project_id, str) or not NAME.match(project_id):
        raise BuilderError(f'bad expansion id {project_id!r}: lowercase letters, digits, ".", "_" and "-"')
    home = extensions_home().resolve()
    folder = (home / project_id).resolve()
    if folder.parent != home:
        raise BuilderError(f'bad expansion id {project_id!r}')
    return folder


def projects() -> List[Dict[str, Any]]:
    """The data extensions in the extensions home, by id."""
    home = extensions_home()
    out: List[Dict[str, Any]] = []
    if home.is_dir():
        for folder in sorted(p for p in home.iterdir() if (p / MANIFEST).is_file()):
            try:
                project = load_project(folder.name)
            except Exception as exc:
                out.append({'id': folder.name, 'name': folder.name, 'error': str(exc)})
                continue
            out.append(
                {
                    'id': project['id'],
                    'name': project['name'],
                    'version': project['version'],
                    'cards': len(project['cards']),
                    'wonders': len({w.get('name') for w in project['wonders']}),
                    'loaded': project['id'] in EXPANSIONS,
                }
            )
    return out


def load_project(project_id: str) -> Dict[str, Any]:
    """A project from its folder (the folder name is its id; the first expansion is the one edited)."""
    folder = _folder(project_id)
    if not (folder / MANIFEST).is_file():
        raise BuilderError(f'no expansion {project_id!r} in {folder.parent}')
    manifest = json.loads((folder / MANIFEST).read_text(encoding='utf-8'))
    entry = (manifest.get('expansions') or [{}])[0]
    raw: Dict[str, Any] = {}
    path = folder / entry.get('file', EXPANSION_FILE)
    if path.is_file():
        raw = json.loads(path.read_text(encoding='utf-8'))
    rules = dict(raw.get('rules') or {})
    if 'starting_coins' in entry:
        rules.setdefault('starting_coins', entry['starting_coins'])
    return {
        'id': manifest.get('id', project_id),
        **{k: manifest.get(k, '') for k in META_KEYS},
        'rules': rules,
        'colors': list(raw.get('colors', [])),
        'cards': list(raw.get('cards', [])),
        'wonders': list(raw.get('wonders', [])),
    }


def _empty(value: Any) -> bool:
    return value is None or value == '' or value == [] or value == {}


def _clean_effects(effects: Any) -> List[Dict[str, Any]]:
    out = []
    for effect in effects or []:
        if isinstance(effect, dict):
            out.append({k: v for k, v in effect.items() if not _empty(v)})
    return out


def _clean_cost(cost: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(cost, dict):
        return None
    out: Dict[str, Any] = {}
    if cost.get('coins'):
        out['coins'] = cost['coins']
    resources = {r: n for r, n in (cost.get('resources') or {}).items() if n}
    if resources:
        out['resources'] = resources
    return out or None


def clean_card(spec: Dict[str, Any]) -> Dict[str, Any]:
    """A card spec as the file keeps it: known keys in a fixed order, empty values dropped."""
    card = dict(spec)
    card['cost'] = _clean_cost(card.get('cost'))
    card['effects'] = _clean_effects(card.get('effects'))
    if not card.get('guild'):
        card.pop('guild', None)
    extra = {k: v for k, v in card.items() if k not in CARD_KEYS and not _empty(v)}
    ordered = {k: card[k] for k in CARD_KEYS if k in card and not _empty(card[k])}
    if 'effects' not in ordered:
        ordered['effects'] = []
    return {**ordered, **extra}


def clean_wonder(spec: Dict[str, Any]) -> Dict[str, Any]:
    """A wonder side as the file keeps it."""
    wonder = dict(spec)
    wonder['stages'] = [
        {k: v for k, v in (('cost', _clean_cost(s.get('cost'))), ('effects', _clean_effects(s.get('effects')))) if v}
        for s in wonder.get('stages') or []
        if isinstance(s, dict)
    ]
    wonder['effects'] = _clean_effects(wonder.get('effects'))
    extra = {k: v for k, v in wonder.items() if k not in WONDER_KEYS and not _empty(v)}
    ordered = {k: wonder[k] for k in WONDER_KEYS if k in wonder and not _empty(wonder[k])}
    ordered.setdefault('stages', [])
    return {**ordered, **extra}


def clean_color(spec: Dict[str, Any]) -> Dict[str, Any]:
    keys = ('id', 'label', 'fill', 'score', 'tradeable')
    return {k: spec[k] for k in keys if k in spec and not _empty(spec[k]) and spec[k] is not False}


def clean_project(project: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(project, dict):
        raise BuilderError('expected a project object')
    return {
        'id': str(project.get('id', '')),
        **{k: str(project.get(k) or '') for k in META_KEYS},
        'rules': {k: v for k, v in (project.get('rules') or {}).items() if not _empty(v)},
        'colors': [clean_color(c) for c in project.get('colors') or [] if isinstance(c, dict)],
        'cards': [clean_card(c) for c in project.get('cards') or [] if isinstance(c, dict)],
        'wonders': [clean_wonder(w) for w in project.get('wonders') or [] if isinstance(w, dict)],
    }


def save_project(project: Dict[str, Any], previous_id: str = '') -> Dict[str, Any]:
    """Write a project to its folder and (re)load it, so the menu offers it right away.

    Without ``previous_id`` the project is new and gets a new folder (an existing one is refused);
    with it, that project is updated, and renamed when the id changed. Other files of the folder
    (models, locales, more expansions) and unknown manifest keys are kept.
    """
    project = clean_project(project)
    pid = project['id']
    folder = _folder(pid)
    if not project['name'].strip():
        raise BuilderError('the expansion needs a name')
    owner = extensions.REGISTRY.expansion_owner.get(pid)
    if pid in EXPANSIONS and owner != pid and pid != previous_id:
        raise BuilderError(f'the id {pid!r} is taken by another expansion')
    if previous_id and previous_id != pid:
        old = _folder(previous_id)
        if folder.exists():
            raise BuilderError(f'a folder {folder.name!r} already exists')
        if old.is_dir():
            unload(previous_id)
            old.rename(folder)
    elif not previous_id and folder.exists():  # a new project must not overwrite another one
        raise BuilderError(f'an expansion folder {pid!r} exists already: choose another id')
    folder.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {}
    if (folder / MANIFEST).is_file():
        manifest = json.loads((folder / MANIFEST).read_text(encoding='utf-8'))
    expansions = list(manifest.get('expansions') or [])
    first = dict(expansions[0]) if expansions else {}
    first.pop('starting_coins', None)  # kept in the expansion file's rules instead
    first.update(id=pid, name=project['name'], file=first.get('file') or EXPANSION_FILE)
    manifest.update(
        id=pid,
        **{k: project[k] for k in META_KEYS},
        rules_version=RULES_VERSION,
        expansions=[first] + expansions[1:],
    )
    manifest.setdefault('version', '0.1.0')
    if not manifest['version']:
        manifest['version'] = '0.1.0'
    data: Dict[str, Any] = {}
    for key in ('rules', 'colors', 'cards', 'wonders'):
        if project[key] or key in ('cards', 'wonders'):
            data[key] = project[key]
    (folder / first['file']).write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    (folder / MANIFEST).write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    report = check_project(project)
    unload(pid)
    if report['ok']:  # with errors it stays saved but unplayable until fixed
        try:
            reload_folder(folder)
        except Exception as exc:
            report['problems'].append(_problem('general', None, f'not loaded: {exc}'))
            report.update(ok=False, errors=report['errors'] + 1)
    report['loaded'] = pid in EXPANSIONS
    return {'project': load_project(pid), 'check': report}


def delete_project(project_id: str) -> None:
    folder = _folder(project_id)
    if not (folder / MANIFEST).is_file():
        raise BuilderError(f'no expansion {project_id!r}')
    unload(project_id)
    shutil.rmtree(folder)


def export_project(project_id: str) -> bytes:
    """The project folder as a zip (the folder itself inside, as ``extensions/`` expects it)."""
    folder = _folder(project_id)
    if not (folder / MANIFEST).is_file():
        raise BuilderError(f'no expansion {project_id!r}')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                archive.write(path, f'{folder.name}/{path.relative_to(folder).as_posix()}')
    return buffer.getvalue()


# --- checking --------------------------------------------------------------------------------


def _problem(section: str, index: Optional[int], message: str, level: str = 'error') -> Dict[str, Any]:
    return {'section': section, 'index': index, 'message': message, 'level': level}


def define_colors(colors: Sequence[Dict[str, Any]]) -> List[str]:
    """Define (or update) a project's colours so its cards can be built; returns the problems."""
    problems = []
    for spec in colors:
        try:
            Color.from_spec(spec, redefine=True)
        except Exception as exc:
            problems.append(str(exc))
    return problems


def check_project(project: Dict[str, Any]) -> Dict[str, Any]:
    """Problems of a project, each pinned to a section and item: errors stop it loading, warnings do not."""
    project = clean_project(project)
    pid = project['id']
    known = _known_content(without=pid)
    out = _check_meta(project) + _check_colors(project['colors'])
    problems, mine = _check_cards(project, {c.name for c in known.cards})
    out += problems + _check_wonders(project['wonders'], {w.name for w in known.wonders})
    extra = project['rules'].get('extra_guilds', 2)
    if isinstance(extra, int) and not check_rules({'extra_guilds': extra}):
        out += _deck_warnings(mine, extra)
    errors = sum(p['level'] == 'error' for p in out)
    return {'ok': not errors, 'errors': errors, 'warnings': len(out) - errors, 'problems': out}


def _check_meta(project: Dict[str, Any]) -> List[Dict[str, Any]]:
    from wonders.contrib import NAME, VERSION

    pid, out = project['id'], []
    if not NAME.match(pid):
        out.append(_problem('general', None, f'bad id {pid!r}: lowercase letters, digits, ".", "_" and "-"'))
    elif pid in EXPANSIONS and extensions.REGISTRY.expansion_owner.get(pid) != pid:
        out.append(_problem('general', None, f'the id {pid!r} is taken by another expansion'))
    if not project['name'].strip():
        out.append(_problem('general', None, 'the expansion needs a name'))
    if project['version'] and not VERSION.match(project['version']):
        out.append(_problem('general', None, f'version {project["version"]!r} should look like 0.1.0', 'warning'))
    return out + [_problem('rules', None, p) for p in check_rules(project['rules'])]


def _check_colors(colors: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    ids = Counter(c.get('id') for c in colors)
    for i, spec in enumerate(colors):
        if ids[spec.get('id')] > 1:
            out.append(_problem('colors', i, f'colour id {spec.get("id")!r} is used twice'))
        try:
            Color.from_spec(spec, redefine=True)
        except Exception as exc:
            out.append(_problem('colors', i, _reason(exc)))
    return out


def _check_cards(project: Dict[str, Any], known: Set[str]) -> Tuple[List[Dict[str, Any]], List[Card]]:
    out: List[Dict[str, Any]] = []
    mine: List[Card] = []
    specs = project['cards']
    names = Counter(c.get('name') for c in specs)
    for i, spec in enumerate(specs):
        name = spec.get('name')
        if not name:
            out.append(_problem('cards', i, 'the card needs a name'))
            continue
        try:
            card = build_card(spec, project['id'])
        except Exception as exc:
            out.append(_problem('cards', i, _reason(exc)))
            continue
        mine.append(card)
        if names[name] > 1:
            out.append(_problem('cards', i, f'{name!r} is used by more than one card'))
        if name in known:
            out.append(_problem('cards', i, f'{name!r} clashes with a card of another expansion'))
        out += _card_warnings(card, i)
        for link in [spec.get('chain_requires')] + list(spec.get('chain_enables') or []):
            if link and link not in known and link not in names:
                out.append(_problem('cards', i, f'chain to unknown card {link!r}'))
    return out, mine


def _check_wonders(specs: Sequence[Dict[str, Any]], known: Set[str]) -> List[Dict[str, Any]]:
    out = []
    sides: Dict[str, List[str]] = {}
    for i, spec in enumerate(specs):
        name = spec.get('name')
        if not name:
            out.append(_problem('wonders', i, 'the wonder needs a name'))
            continue
        side = spec.get('side', 'A')
        if side in sides.get(name, []):
            out.append(_problem('wonders', i, f'{name!r} has two sides {side}'))
        sides.setdefault(name, []).append(side)
        if side not in ('A', 'B'):
            out.append(_problem('wonders', i, f'side must be A or B, not {side!r}'))
        if name in known:
            out.append(_problem('wonders', i, f'{name!r} clashes with a wonder of another expansion'))
        try:
            build_wonder(spec)
        except Exception as exc:
            out.append(_problem('wonders', i, _reason(exc)))
        if not spec.get('stages'):
            out.append(_problem('wonders', i, 'the board has no stages', 'warning'))
    for name, got in sides.items():
        missing = sorted({'A', 'B'} - set(got))
        if missing:
            index = next(i for i, w in enumerate(specs) if w.get('name') == name)
            out.append(_problem('wonders', index, f'{name!r} lacks side {", ".join(missing)} (games pick A or B)'))
    return out


def _reason(exc: Exception) -> str:
    if isinstance(exc, KeyError):
        return f'missing field {exc}'
    if isinstance(exc, (TypeError, AttributeError)):
        return f'malformed value ({exc})'
    return str(exc)


def _card_warnings(card: Any, index: int) -> List[Dict[str, Any]]:
    out = []
    drawn = card.guild or card.color in (Color.WHITE, Color.BLACK)
    if card.color is not Color.WHITE and card.age not in (1, 2, 3):
        out.append(_problem('cards', index, 'the card needs an Age (1, 2 or 3)'))
    bad = [n for n in card.at_players if not 3 <= n <= 7]
    if bad:
        out.append(_problem('cards', index, f'player counts {bad} are outside 3–7'))
    if not drawn and not card.at_players:
        out.append(_problem('cards', index, 'no player counts: the card is never dealt', 'warning'))
    if card.guild and card.age != 3:
        out.append(_problem('cards', index, 'guilds are drawn in Age III only', 'warning'))
    if not card.effects:
        out.append(_problem('cards', index, 'the card has no effect', 'warning'))
    return out


def _deck_warnings(cards: Sequence[Any], extra_guilds: int) -> List[Dict[str, Any]]:
    """Hand sizes are deck // players: warn when the new cards change them or leave cards undealt."""
    base = load_content('base')
    ours = Content(cards=base.cards + list(cards), wonders=base.wonders)
    notes = []
    for n in range(3, 8):
        for age in (1, 2, 3):
            mine = len(build_age_deck(ours, age, n, random.Random(0), extra_guilds))
            theirs = len(build_age_deck(base, age, n, random.Random(0)))
            if mine // n != theirs // n or mine % n:
                notes.append(f'Age {age} at {n} players: {mine} cards, hands of {mine // n} ({mine % n} undealt)')
    if not notes:
        return []
    more = f' (+{len(notes) - 4} more)' if len(notes) > 4 else ''
    message = 'deck sizes change: ' + '; '.join(notes[:4]) + more
    return [_problem('cards', None, message, 'warning')]


# --- previews --------------------------------------------------------------------------------


def preview(kind: str, spec: Dict[str, Any], colors: Sequence[Dict[str, Any]] = (), t: Translator = ENGLISH) -> str:
    """The card or wonder-side SVG of an unsaved spec (raises ``ValueError`` when it does not build)."""
    from wonders.viz.render import card_svg, wonder_svg

    define_colors([clean_color(c) for c in colors if isinstance(c, dict)])
    if kind == 'card':
        card = clean_card(spec)
        card.setdefault('name', '?')
        card.setdefault('color', 'blue')
        return card_svg(build_card(card, 'preview'), t)
    if kind == 'wonder':
        wonder = clean_wonder(spec)
        wonder.setdefault('name', '?')
        return wonder_svg(build_wonder(wonder), 0, t)
    raise BuilderError(f'unknown preview kind {kind!r}')


# --- balance ---------------------------------------------------------------------------------


def balance(project_id: str, games: int = 100, players: int = 4, seed: int = 0) -> Dict[str, Any]:
    """Scripted agents play the saved expansion (with the base game) and the base game alone, same seeds.

    Reports mean scores per category for both, and for every new card how often it was built and how
    its builders placed; for every new wonder how its players placed. A quick look, not a verdict.
    """
    from wonders.agents.bench import AGENTS
    from wonders.agents.match import play_match
    from wonders.engine.record import PlayerSetup
    from wonders.play.rating import places

    if project_id not in EXPANSIONS or extensions.REGISTRY.expansion_owner.get(project_id) != project_id:
        raise BuilderError(f'save {project_id!r} without errors first')
    games = max(1, min(int(games), MAX_BALANCE_GAMES))
    players = max(3, min(int(players), 7))
    lineup = [AGENTS[BALANCE_LINEUP[i % len(BALANCE_LINEUP)]] for i in range(players)]
    combos = {'expansion': ('base', project_id), 'base': ('base',)}
    new_cards = {c.name for c in load_content(project_id).cards}
    new_wonders = {w.name for w in load_content(project_id).wonders}
    result: Dict[str, Any] = {'games': games, 'players': players, 'lineup': [s.name for s in lineup]}
    for key, combo in combos.items():
        content = load_content(*combo)
        names = list(dict.fromkeys(w.name for w in content.wonders))
        if len(names) < players:
            raise BuilderError(f'only {len(names)} wonders for {players} players')
        totals: Counter = Counter()
        cards: Dict[str, Counter] = {n: Counter() for n in sorted(new_cards)}
        wonders: Dict[str, Counter] = {n: Counter() for n in sorted(new_wonders)}
        for g in range(games):
            rng = random.Random(seed * 100_003 + g)
            seats = lineup[g % players :] + lineup[: g % players]  # rotate seats
            setups = [PlayerSetup(s.name, w, rng.choice('AB')) for s, w in zip(seats, rng.sample(names, players))]
            agents = [s.build(random.Random(rng.random())) for s in seats]
            game, _ = play_match(content, setups, agents, rng, combo)
            scores = [game.score(i) for i in range(players)]
            place = places([s.total for s in scores], [p.coins for p in game.players])
            for s in scores:
                totals.update(s.as_dict())
            for i, p in enumerate(game.players):
                for card in p.cards:
                    if card.name in cards:
                        cards[card.name].update(built=1, place=place[i], wins=place[i] == 1)
                if p.wonder.name in wonders:
                    wonders[p.wonder.name].update(seats=1, place=place[i], wins=place[i] == 1, score=scores[i].total)
        n = games * players
        result[key] = {k: round(v / n, 2) for k, v in totals.items()}
        if key == 'expansion':
            result['cards'] = [
                {
                    'name': name,
                    'built_per_game': round(c['built'] / games, 2),
                    'mean_place': round(c['place'] / c['built'], 2) if c['built'] else None,
                    'win_rate': round(c['wins'] / c['built'], 2) if c['built'] else None,
                }
                for name, c in cards.items()
            ]
            result['wonders'] = [
                {
                    'name': name,
                    'seats': c['seats'],
                    'mean_place': round(c['place'] / c['seats'], 2) if c['seats'] else None,
                    'win_rate': round(c['wins'] / c['seats'], 2) if c['seats'] else None,
                    'mean_score': round(c['score'] / c['seats'], 1) if c['seats'] else None,
                }
                for name, c in wonders.items()
            ]
    result['expected_win_rate'] = round(1 / players, 2)
    return result
