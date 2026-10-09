"""Checks for contributed content: extension folders, translations, plugins and model scorecards.

What a contribution is and where it goes (``CONTRIBUTING.md``):

* ``extensions/<id>/`` — **data** extensions (manifest + JSON / checkpoints): expansions, trained models
  and the translations of their own cards. Checked by :func:`check_extension`.
* ``plugins/<name>/`` — **code** extensions: pip-installable packages with an entry point in the
  ``wonders.extensions`` group (new effect types, agents, …). Checked by :func:`check_plugin` once
  installed.
* ``wonders/i18n/<lang>.json`` — a translation of the game itself. Checked by :func:`check_locale`.

A model in an extension also gets a **scorecard** (:func:`scorecard`): a head-to-head against the
bundled network at the same tables; :func:`update_leaderboard` keeps ``LEADERBOARD.md``.

The workflows in ``.github/workflows/contributions.yml`` drive this through ``wonders ext changed |
check | scorecard | summary | leaderboard``; everything also runs locally.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import random
import re
import string
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from wonders.extensions import (
    ENTRY_POINT_GROUP,
    MANIFEST,
    REGISTRY,
    Manifest,
    ensure_loaded,
    load_folder,
    plugin_manifest,
    validate,
)

EXTENSIONS_DIR = 'extensions'  # top-level folders of the repository
PLUGINS_DIR = 'plugins'
CORE_LOCALES = ('wonders', 'i18n')
REFERENCE_LANG = 'de'  # the complete catalog: its keys are every text the game shows
NAME = re.compile(r'^[a-z0-9][a-z0-9._-]{0,63}$')  # ids, folder and model names (safe in paths and shells)
LANG = re.compile(r'^[a-z]{2,3}(-[A-Za-z0-9]{2,8})?$')
VERSION = re.compile(r'^\d+\.\d+(\.\d+)?([.-]?[a-z0-9]+)*$')
MAX_FILE_BYTES = 20 * 2**20
MAX_FOLDER_BYTES = 50 * 2**20
DATA_SUFFIXES = {'.json', '.pt', '.md', '.txt', '.png', '.jpg', '.svg'}
SMOKE_GAMES = 4
SMOKE_LINEUP = ('heuristic', 'random', 'greedy', 'board', 'builder', 'civic', 'scientist')
BUNDLED = 'bundled'
PathLike = Union[str, Path]


# -- reports ---------------------------------------------------------------------------------------------
@dataclass
class Report:
    """The outcome of checking one contribution: findings (``error`` fails the check) plus detail lines."""

    kind: str  # 'extension' | 'plugin' | 'locale'
    target: str
    title: str = ''
    findings: List[Tuple[str, str]] = field(default_factory=list)  # (level, text): 'ok' | 'warning' | 'error'

    def ok(self, text: str) -> None:
        self.findings.append(('ok', text))

    def warning(self, text: str) -> None:
        self.findings.append(('warning', text))

    def error(self, text: str) -> None:
        self.findings.append(('error', text))

    @property
    def passed(self) -> bool:
        return all(level != 'error' for level, _ in self.findings)

    @property
    def errors(self) -> List[str]:
        return [text for level, text in self.findings if level == 'error']

    def markdown(self) -> str:
        icon = {'ok': '✅', 'warning': '⚠️', 'error': '❌'}
        head = f"#### {'✅' if self.passed else '❌'} {self.kind} `{self.target}`"
        lines = [head + (f' — {self.title}' if self.title else ''), '']
        lines += [f'- {icon[level]} {text}' for level, text in self.findings]
        return '\n'.join(lines) + '\n'


def kind_of(target: PathLike) -> str:
    """``'extension'`` (folder with a manifest), ``'plugin'`` (folder with ``pyproject.toml``) or ``'locale'``."""
    path = Path(target)
    if path.is_dir() and (path / MANIFEST).is_file():
        return 'extension'
    if path.is_dir() and (path / 'pyproject.toml').is_file():
        return 'plugin'
    if path.is_file() and path.suffix == '.json':
        return 'locale'
    raise ValueError(f'{target}: not an extension folder, plugin package or locale file')


def check(target: PathLike, games: int = SMOKE_GAMES) -> Report:
    """Check any contribution (see :func:`kind_of`)."""
    kind = kind_of(target)
    if kind == 'extension':
        return check_extension(target, games)
    if kind == 'plugin':
        return check_plugin(target, games)
    return check_locale(target)


# -- which contributions a change touches ------------------------------------------------------------
def changed(paths: Iterable[str], root: PathLike = '.') -> Dict[str, Any]:
    """Map changed repository paths (``git diff --name-only``) to the contributions to check.

    Returns ``{'extensions', 'plugins', 'locales', 'models', 'invalid', 'any'}``: folder / file paths,
    ``models`` as ``{'extension', 'model'}`` for every model whose checkpoint or manifest changed, and
    ``invalid`` paths whose names are not allowed (they must match :data:`NAME`, so they are safe to
    pass around in workflows). Deleted contributions are left out.
    """
    root = Path(root)
    extensions: Dict[str, Set[str]] = {}
    plugins: Set[str] = set()
    locales: Set[str] = set()
    invalid: List[str] = []
    for raw in paths:
        raw = raw.strip()
        parts = PurePosixPath(raw).parts
        if len(parts) >= 3 and parts[0] in (EXTENSIONS_DIR, PLUGINS_DIR):
            name = parts[1]
            if not NAME.match(name):
                invalid.append(raw)
            elif (root / parts[0] / name).is_dir():
                if parts[0] == EXTENSIONS_DIR:
                    extensions.setdefault(name, set()).add('/'.join(parts[2:]))
                else:
                    plugins.add(name)
        elif len(parts) == 3 and parts[:2] == CORE_LOCALES and parts[2].endswith('.json'):
            if not LANG.match(parts[2][: -len('.json')]):
                invalid.append(raw)
            elif (root / raw).is_file():
                locales.add(raw)
    models: List[Dict[str, str]] = []
    for name, files in sorted(extensions.items()):
        try:
            manifest = json.loads((root / EXTENSIONS_DIR / name / MANIFEST).read_text(encoding='utf-8'))
            entries = list(manifest.get('models', []))
        except Exception:
            continue  # reported by the check
        for entry in entries:
            model = str(entry.get('name', ''))
            if not NAME.match(model):
                invalid.append(f'{EXTENSIONS_DIR}/{name}: model name {model!r}')
            elif MANIFEST in files or entry.get('file') in files:
                models.append({'extension': name, 'model': model})
    out = {
        'extensions': [f'{EXTENSIONS_DIR}/{name}' for name in sorted(extensions)],
        'plugins': sorted(plugins),
        'locales': sorted(locales),
        'models': models,
        'invalid': invalid,
    }
    out['any'] = any(out[k] for k in ('extensions', 'plugins', 'locales'))
    return out


# -- translations ----------------------------------------------------------------------------------------
def _fields(text: str) -> List[str]:
    return sorted(name for _, name, _, _ in string.Formatter().parse(text) if name)


def _reference_keys() -> Set[str]:
    from wonders.i18n import _bundled

    return {k for k in _bundled(REFERENCE_LANG) if not k.startswith('_')}


def _check_strings(
    report: Report,
    pairs: Sequence[Tuple[Any, Any]],
    label: str,
    require_language: bool,
    known: Set[str] = frozenset(),  # type: ignore[assignment]
    coverage: bool = False,
) -> None:
    keys = [k for k, _ in pairs]
    duplicates = sorted({str(k) for k in keys if keys.count(k) > 1})
    if duplicates:
        report.error(f'{label}: duplicate keys {", ".join(duplicates[:10])}')
    bad = 0
    for key, value in pairs:
        if not isinstance(key, str) or not isinstance(value, str):
            report.error(f'{label}: {key!r} must map text to text')
            bad += 1
            continue
        if key.startswith('_'):
            continue
        if not value.strip():
            report.error(f'{label}: empty translation for {key!r}')
            bad += 1
        else:
            try:
                same = _fields(key.split('|')[0]) == _fields(value)
            except ValueError as exc:
                report.error(f'{label}: {key!r}: {exc}')
                bad += 1
                continue
            if not same:
                report.error(f'{label}: {key!r} must keep the {{fields}} of the English text: {value!r}')
                bad += 1
    data = dict(pairs)
    if require_language and not data.get('_language'):
        report.error(f'{label}: a new language needs "_language" (its name in the language picker)')
    reference = _reference_keys()
    texts = {k for k in data if isinstance(k, str) and not k.startswith('_')}
    unknown = sorted(texts - reference - known)
    if unknown:
        shown = ', '.join(repr(k) for k in unknown[:8]) + (' …' if len(unknown) > 8 else '')
        report.warning(f'{label}: {len(unknown)} key(s) the game does not show (typo or stale?): {shown}')
    summary = f'{label}: {len(texts)} text(s)'
    if coverage:
        share = len(texts & reference) / len(reference)
        summary += f', {100 * share:.1f}% of the {len(reference)} texts the game shows'
        if share < 1:
            report.warning(f'{label}: {len(reference - texts)} text(s) not translated yet (they stay English)')
    if not bad:
        report.ok(summary)


def check_locale(path: PathLike) -> Report:
    """Check a translation of the game (``wonders/i18n/<lang>.json``): well-formed and how complete."""
    path = Path(path)
    lang = path.stem
    report = Report('locale', str(path), lang)
    if not LANG.match(lang):
        report.error(f'file name must be a language code such as fr.json or pt-BR.json, not {path.name}')
    try:
        pairs = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=list)
    except Exception as exc:
        report.error(f'not valid JSON: {exc}')
        return report
    if not isinstance(pairs, list):
        report.error('must be one JSON object of "English text": "translation"')
        return report
    language = dict(pairs).get('_language')
    report.title = f'{language} ({lang})' if language else lang
    _check_strings(report, pairs, path.name, require_language=True, coverage=True)
    return report


# -- extensions ------------------------------------------------------------------------------------------
def _check_manifest(report: Report, manifest: Manifest, strict: bool = True) -> None:
    from wonders.ruleset import RULES_VERSION

    if not NAME.match(manifest.id):
        report.error(f'id {manifest.id!r}: use lower-case letters, digits, ".", "_" and "-"')
    if not VERSION.match(str(manifest.version)):
        report.error(f'version {manifest.version!r} is not a version number such as 0.1.0')
    missing = [k for k in ('author', 'license', 'description') if not getattr(manifest, k)]
    if missing:
        (report.error if strict else report.warning)(f'manifest lacks {", ".join(missing)}')
    if manifest.rules_version is None:
        (report.error if strict else report.warning)(
            f'manifest lacks rules_version (the engine implements {RULES_VERSION})'
        )
    elif manifest.rules_version != RULES_VERSION:
        report.warning(f'made for rules version {manifest.rules_version}; the engine implements {RULES_VERSION}')


def _check_files(report: Report, folder: Path, manifest: Manifest) -> None:
    listed = {MANIFEST} | {e.get('file') for e in manifest.expansions + manifest.models + manifest.locales}
    total = 0
    for path in sorted(folder.rglob('*')):
        rel = path.relative_to(folder).as_posix()
        if path.is_symlink():
            report.error(f'{rel}: symbolic links are not allowed')
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        total += size
        if size > MAX_FILE_BYTES:
            report.error(f'{rel}: {size / 2**20:.1f} MB, the limit is {MAX_FILE_BYTES // 2**20} MB per file')
        if path.suffix.lower() not in DATA_SUFFIXES and path.name not in ('LICENSE', 'NOTICE'):
            report.error(f'{rel}: extension folders hold data only ({", ".join(sorted(DATA_SUFFIXES))})')
        elif rel not in listed and path.suffix in ('.json', '.pt'):
            report.warning(f'{rel}: not named in the manifest, so it is never loaded')
    if total > MAX_FOLDER_BYTES:
        report.error(f'{total / 2**20:.1f} MB in total, the limit is {MAX_FOLDER_BYTES // 2**20} MB')


def _check_siblings(report: Report, folder: Path, manifest: Manifest) -> None:
    """Ids, expansion ids and model names must be unique among the other extensions next to it."""
    if folder.resolve().parent.name == EXTENSIONS_DIR and folder.name != manifest.id:
        report.error(f'the folder must be named after the manifest id: {EXTENSIONS_DIR}/{manifest.id}')
    from wonders.ruleset import EXPANSIONS

    ours = {str(e.get('id')) for e in manifest.expansions}
    builtin = ours & {e for e in EXPANSIONS if e not in REGISTRY.expansion_owner}
    if builtin:
        report.error(f'expansion id {", ".join(sorted(builtin))} is reserved')
    models = {str(m.get('name')) for m in manifest.models}
    if BUNDLED in models or 'neural' in models:
        report.error('model names "bundled" and "neural" are reserved')
    for other in sorted(folder.resolve().parent.iterdir()):
        if other == folder.resolve() or not (other / MANIFEST).is_file():
            continue
        try:
            theirs = json.loads((other / MANIFEST).read_text(encoding='utf-8'))
        except Exception:
            continue
        if theirs.get('id') == manifest.id:
            report.error(f'id {manifest.id!r} is also used by {other.name}')
        clash = ours & {str(e.get('id')) for e in theirs.get('expansions', [])}
        clash |= models & {str(m.get('name')) for m in theirs.get('models', [])}
        if clash:
            report.error(f'{", ".join(sorted(map(str, clash)))} also used by {other.name}')


def _content_names(expansion_ids: Sequence[str]) -> Set[str]:
    """Card and wonder names plus effect descriptions an extension's locale may translate."""
    names: Set[str] = set()
    for exp in expansion_ids:
        raw = REGISTRY.expansion_raw(exp) or {}
        for item in raw.get('cards', []) + raw.get('wonders', []):
            names.add(str(item.get('name', '')))
            for spec in item.get('effects', []) + [e for s in item.get('stages', []) for e in s.get('effects', [])]:
                if spec.get('description'):
                    names.add(str(spec['description']))
    return names


def _load_folder(report: Report, folder: Path, manifest: Manifest) -> bool:
    ensure_loaded()
    loaded = REGISTRY.extensions.get(manifest.id)
    if loaded is not None:
        if Path(loaded.location).resolve() == folder.resolve():
            return True
        report.error(f'another extension with id {manifest.id!r} is installed ({loaded.location})')
        return False
    try:
        load_folder(folder)
    except Exception as exc:
        report.error(f'does not load: {exc}')
        return False
    return True


def check_extension(folder: PathLike, games: int = SMOKE_GAMES) -> Report:
    """Check a data extension folder: manifest, files, cards and decks, locales and models, plus games.

    Every expansion is played (scripted agents, alone and with Leaders + Cities, 3 and 7 players) and
    every game is replayed from its record; every model is loaded and played.
    """
    folder = Path(folder)
    report = Report('extension', folder.as_posix())
    try:
        manifest = Manifest.from_dict(json.loads((folder / MANIFEST).read_text(encoding='utf-8')))
    except Exception as exc:
        report.error(f'{MANIFEST}: {exc}')
        return report
    report.title = f'{manifest.name} {manifest.version}'
    _check_manifest(report, manifest)
    _check_siblings(report, folder, manifest)
    _check_files(report, folder, manifest)
    for problem in validate(folder):
        report.error(problem)
    if not report.passed or not _load_folder(report, folder, manifest):
        return report
    expansion_ids = [e['id'] for e in manifest.expansions]
    for exp in expansion_ids:
        _check_expansion(report, exp, games)
    from wonders.i18n import languages

    known = _content_names(expansion_ids)
    for entry in manifest.locales:
        lang = str(entry.get('lang', ''))
        try:
            pairs = json.loads((folder / entry['file']).read_text(encoding='utf-8'), object_pairs_hook=list)
        except Exception as exc:
            report.error(f"{entry.get('file')}: not valid JSON ({exc})")
            continue
        if not LANG.match(lang):
            report.error(f'locale {lang!r}: not a language code')
        _check_strings(report, pairs, f'locale {lang}', require_language=lang not in languages(), known=known)
    for entry in manifest.models:
        _check_model(report, folder / entry['file'], entry['name'], games)
        if not entry.get('description'):
            report.warning(f"model {entry['name']}: add a description (how it was trained)")
    return report


def _check_expansion(report: Report, expansion: str, games: int) -> None:
    from wonders.cards.content import load_content

    content = load_content('base', expansion)
    cards = [c for c in content.cards if c.expansion == expansion]
    for card in cards:
        counts = [n for n in card.at_players if not 3 <= n <= 7]
        if counts:
            report.error(f'card {card.name!r}: at_players {counts} outside 3–7')
    report.ok(f'expansion {expansion}: {len(cards)} card(s)')
    _check_decks(report, expansion)
    for combo in (('base', expansion), ('base', 'leaders', 'cities', expansion)):
        for n in (3, 7):
            _smoke(report, combo, [SMOKE_LINEUP[i % len(SMOKE_LINEUP)] for i in range(n)], games)


def _check_decks(report: Report, expansion: str) -> None:
    """Warn when the expansion changes hand sizes or leaves cards undealt (hands are ``len(deck) // n``)."""
    from wonders.cards.content import build_age_deck, load_content

    with_it, without = load_content('base', expansion), load_content('base')
    notes = []
    for n in range(3, 8):
        for age in (1, 2, 3):
            ours = len(build_age_deck(with_it, age, n, random.Random(0)))
            theirs = len(build_age_deck(without, age, n, random.Random(0)))
            if ours // n != theirs // n or ours % n:
                notes.append(f'Age {age} at {n}p: {ours} cards ({theirs} in the base game)')
    if notes:
        report.warning(
            'decks change hand sizes or leave cards undealt (hands are deck // players): '
            + '; '.join(notes[:5])
            + (' …' if len(notes) > 5 else '')
        )


def _smoke(report: Report, combo: Sequence[str], lineup: Sequence[Any], games: int, label: str = '') -> None:
    """Play ``games`` seeded games and replay each from its JSON record; report failures."""
    from wonders.agents.bench import AGENTS
    from wonders.agents.match import play_match
    from wonders.cards.content import load_content
    from wonders.engine.record import GameRecord, PlayerSetup

    specs = [AGENTS[a] if isinstance(a, str) else a for a in lineup]
    key = f"{'+'.join(e for e in combo if e != 'base') or 'base'}/{len(specs)}p"
    try:
        content = load_content(*combo)
        names = list(dict.fromkeys(w.name for w in content.wonders))
        for g in range(games):
            rng = random.Random(1_000 + g)
            setups = [
                PlayerSetup(spec.name, wonder, rng.choice('AB'))
                for spec, wonder in zip(specs, rng.sample(names, len(specs)))
            ]
            agents = [spec.build(random.Random(rng.random())) for spec in specs]
            game, record = play_match(content, setups, agents, rng=rng, expansions=tuple(combo))
            replayed = GameRecord.from_json(record.to_json()).replay(content)
            scores = [game.score(i).total for i in range(game.n)]
            if [replayed.score(i).total for i in range(replayed.n)] != scores:
                report.error(f'{label}{key}: game {g} does not replay to the same scores')
                return
    except Exception as exc:
        report.error(f'{label}{key}: {type(exc).__name__}: {exc}')
        return
    report.ok(f'{label}{key}: {games} game(s) played and replayed')


def _check_model(report: Report, path: Path, name: str, games: int) -> None:
    """Load a checkpoint (``weights_only``), report what it covers, and play it."""
    from wonders.ruleset import RULES_VERSION

    try:
        from wonders.ml.evaluate import COMBOS, _covers, cell_key, neural_spec
        from wonders.ml.model import load_checkpoint, rules_version
    except ImportError:
        report.warning(f"model {name}: not checked here (needs PyTorch: pip install 'wonders[ml]')")
        return
    try:
        model, card_names, meta = load_checkpoint(str(path))
    except Exception as exc:
        report.error(f'model {name}: does not load ({exc})')
        return
    covers = [combo for combo in COMBOS if _covers(card_names, combo)]
    if not covers:
        report.error(f'model {name}: its vocabulary covers none of the bundled expansion combinations')
        return
    from wonders.ml import pretrained_path

    if _sha256(path) == _sha256(pretrained_path()):
        report.warning(f'model {name}: identical to the bundled network')
    trained = rules_version(meta)
    if trained != RULES_VERSION:
        report.warning(f'model {name}: trained under rules version {trained}; the engine implements {RULES_VERSION}')
    params = sum(p.numel() for p in model.parameters())
    plays = ', '.join(cell_key((c, 3)).split('/')[0] for c in covers)
    report.ok(f"model {name}: {params:,} parameters, encoder {meta.get('encoder', 'card-v1')}, plays {plays}")
    spec = neural_spec(f'neural:{name}', str(path))
    _smoke(report, covers[-1], [spec, 'heuristic', 'board'], max(1, games // 2), label=f'model {name} at ')


# -- plugins ---------------------------------------------------------------------------------------------
def _pyproject(folder: Path) -> Dict[str, Any]:
    import tomllib

    with open(folder / 'pyproject.toml', 'rb') as fh:
        return tomllib.load(fh)


def check_plugin(folder: PathLike, games: int = SMOKE_GAMES) -> Report:
    """Check an installed plugin package: its entry points load and what they register works.

    The package in ``folder`` must be installed first (``pip install plugins/<name>``); its own tests
    are run separately (``pytest plugins/<name>``).
    """
    folder = Path(folder)
    report = Report('plugin', folder.as_posix())
    try:
        project = _pyproject(folder).get('project', {})
    except Exception as exc:
        report.error(f'pyproject.toml: {exc}')
        return report
    dist_name = project.get('name', '')
    report.title = f"{dist_name} {project.get('version', '')}".strip()
    if not project.get('entry-points', {}).get(ENTRY_POINT_GROUP):
        report.error(f'pyproject.toml declares no entry point in the {ENTRY_POINT_GROUP!r} group')
        return report
    if not any(re.match(r'^wonders\b', str(d)) for d in project.get('dependencies', [])):
        report.warning('add "wonders" to the dependencies')
    try:
        dist = metadata.distribution(dist_name)
    except metadata.PackageNotFoundError:
        report.error(f'{dist_name} is not installed: pip install {folder.as_posix()}')
        return report
    ensure_loaded()
    for ep in (e for e in dist.entry_points if e.group == ENTRY_POINT_GROUP):
        try:
            target = ep.load()
            info = plugin_manifest(target)
            manifest = Manifest.from_dict({'version': dist.version, **(info or {'id': ep.name, 'name': ep.name})})
        except Exception as exc:
            report.error(f'entry point {ep.name}: does not load ({type(exc).__name__}: {exc})')
            continue
        if info is None:
            report.warning(f'entry point {ep.name}: define MANIFEST (id, name, rules_version, …) in the module')
        _check_manifest(report, manifest, strict=False)
        ext = REGISTRY.extensions.get(manifest.id)
        if ext is None:
            from wonders.extensions import load_plugin

            try:
                ext = load_plugin(manifest, target)
            except Exception as exc:
                report.error(f'entry point {ep.name}: register() failed ({type(exc).__name__}: {exc})')
                continue
        report.ok(f"entry point {ep.name}: registers {', '.join(ext.provides) or 'nothing'}")
        _check_provided(report, ext.provides, games)
    return report


def _check_provided(report: Report, provides: Sequence[str], games: int) -> None:
    from wonders.agents.bench import AGENTS

    for item in provides:
        kind, _, value = item.partition(':')
        if kind == 'agent':
            for combo, n in ((('base',), 3), (('base', 'leaders', 'cities'), 5)):
                lineup = [AGENTS[value]] + [SMOKE_LINEUP[i % len(SMOKE_LINEUP)] for i in range(n - 1)]
                _smoke(report, combo, lineup, games, label=f'agent {value} at ')
        elif kind == 'expansion':
            _check_expansion(report, value, games)
        elif kind == 'model':
            entry = REGISTRY.models[value]
            _check_model(report, Path(entry.path), value, games)
        elif kind == 'locale':
            from wonders.i18n import languages

            pairs = list(REGISTRY.locales.get(value, {}).items())
            known = _content_names(list(REGISTRY.expansion_data))
            _check_strings(report, pairs, f'locale {value}', require_language=value not in languages(), known=known)


# -- model scorecards ------------------------------------------------------------------------------------
def _sha256(path: PathLike) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def model_entry(source: PathLike, model: Optional[str] = None) -> Tuple[str, Path, Dict[str, Any]]:
    """``(name, checkpoint, info)`` for a model in an extension folder (or a bare ``.pt`` file)."""
    source = Path(source)
    if source.is_file():
        return source.stem, source, {}
    manifest = Manifest.from_dict(json.loads((source / MANIFEST).read_text(encoding='utf-8')))
    entries = manifest.models if model is None else [m for m in manifest.models if m.get('name') == model]
    if len(entries) != 1:
        names = ', '.join(m.get('name', '?') for m in manifest.models) or 'none'
        raise ValueError(f'{source}: name one model with --model (models: {names})')
    info = {
        'extension': manifest.id,
        'version': manifest.version,
        'author': manifest.author,
        'description': entries[0].get('description', ''),
    }
    return entries[0]['name'], source / entries[0]['file'], info


def scorecard(
    source: PathLike,
    model: Optional[str] = None,
    cells: str = 'all',
    games: int = 200,
    seed: int = 4242,
    processes: int = 1,
    filler: str = 'heuristic',
) -> Dict[str, Any]:
    """Play a contributed model against the bundled network at the same tables (needs PyTorch).

    Seats are shuffled every game and the other seats are ``filler`` agents. Returns the
    :func:`~wonders.ml.evaluate.head_to_head` result plus what identifies the run (checkpoint hashes,
    rules version, seed), so :func:`merge_scorecards` and :func:`update_leaderboard` can use it.
    """
    from wonders.ml import pretrained_path
    from wonders.ml.evaluate import head_to_head, parse_cells
    from wonders.ruleset import RULES_VERSION

    name, checkpoint, info = model_entry(source, model)
    result = head_to_head(
        [(name, str(checkpoint)), (BUNDLED, pretrained_path())],
        cells=parse_cells(cells),
        games=games,
        seed=seed,
        filler=filler,
        processes=processes,
        seating='shuffle',
    )
    return {
        'model': name,
        **info,
        'source': Path(source).as_posix(),
        'checkpoint_sha256': _sha256(checkpoint),
        'opponent': BUNDLED,
        'opponent_sha256': _sha256(pretrained_path()),
        'rules_version': RULES_VERSION,
        'games_per_cell': games,
        'seed': seed,
        'cells': [cells],
        'result': result,
    }


def merge_scorecards(cards: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge scorecards of the same model run in shards (e.g. one CI job per expansion combination)."""
    from wonders.ml.evaluate import merge_head_to_head

    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for card in cards:
        groups.setdefault((card.get('extension', ''), card['model'], card['checkpoint_sha256']), []).append(card)
    merged = []
    for group in groups.values():
        out = dict(group[0])
        out['cells'] = [c for card in group for c in card['cells']]
        out['result'] = merge_head_to_head([card['result'] for card in group])
        merged.append(out)
    return merged


def _pair(result: Dict[str, Any], a: str, b: str, pairs: Optional[Sequence[Dict[str, Any]]] = None):
    """``(share of games a ranked above b, (lo, hi), games)`` from a head-to-head pair list."""
    for p in result['pairs'] if pairs is None else pairs:
        if (p['a'], p['b']) == (a, b):
            return p['a_above_b'], tuple(p['ci']), p['games']
        if (p['a'], p['b']) == (b, a):
            return 1 - p['a_above_b'], (1 - p['ci'][1], 1 - p['ci'][0]), p['games']
    return None


def _weighted(result: Dict[str, Any], label: str) -> Dict[str, Any]:
    """Games-weighted first-place share, mean score and category means of ``label`` over all cells."""
    from wonders.agents.bench import CATEGORIES

    games = first = score = 0.0
    categories = {c: 0.0 for c in CATEGORIES}
    for cell in result['cells']:
        stats = cell.get('models', {}).get(label)
        if not stats:
            continue
        n_players = int(cell['cell'].rsplit('/', 1)[1].rstrip('p'))
        g = cell['games']
        games += g
        first += stats['win_rel'] / n_players * g
        score += stats['mean_score'] * g
        for c, value in stats.get('categories', {}).items():
            categories[c] += value * g
    if not games:
        return {'games': 0, 'first': 0.0, 'score': 0.0, 'categories': categories}
    return {
        'games': games,
        'first': first / games,
        'score': score / games,
        'categories': {c: v / games for c, v in categories.items()},
    }


def verdict(share: float, ci: Sequence[float]) -> str:
    if ci[0] > 0.5:
        return '🏆 **new state of the art** — ranked above the bundled network (lower bound above 50%)'
    if ci[1] < 0.5:
        return 'weaker than the bundled network (upper bound below 50%)'
    return 'on par with the bundled network (the interval includes 50%)'


def scorecard_markdown(card: Dict[str, Any]) -> str:
    """The PR comment / job summary for one (merged) scorecard."""
    from wonders.agents.bench import CATEGORIES

    result, name = card['result'], card['model']
    label = f"{card.get('extension', '')}/{name}" if card.get('extension') else name
    lines = [f'#### 📊 Scorecard: `{label}` vs the bundled network', '']
    overall = _pair(result, name, BUNDLED)
    if overall is None:
        return '\n'.join(lines + ['No cell could seat both networks (vocabulary).', ''])
    share, ci, games = overall
    lines += [f'Verdict: {verdict(share, ci)}.', '']
    ours, theirs = _weighted(result, name), _weighted(result, BUNDLED)
    lines += [
        f'| | `{name}` | bundled |',
        '|---|---|---|',
        f'| ranked above the other (95% CI) | **{100 * share:.1f}%** ({100 * ci[0]:.1f}–{100 * ci[1]:.1f}) '
        f'| {100 * (1 - share):.1f}% |',
        f"| first place | {100 * ours['first']:.1f}% | {100 * theirs['first']:.1f}% |",
        f"| mean score | {ours['score']:.1f} | {theirs['score']:.1f} |",
        f'| games | {games} | |',
        '',
        '| combination | games | ranked above bundled (95% CI) |',
        '|---|---|---|',
    ]
    for combo, pairs in sorted(result['pairs_by_combo'].items()):
        found = _pair(result, name, BUNDLED, pairs)
        if found:
            s, (lo, hi), g = found
            lines.append(f'| {combo} | {g} | {100 * s:.1f}% ({100 * lo:.1f}–{100 * hi:.1f}) |')
    by_players: Dict[int, List[int]] = {}
    for cell in result['cells']:
        for pair, value in cell.get('pairs', {}).items():
            a, _, b = pair.partition(' > ')
            if {a, b} == {name, BUNDLED}:
                above = round((value if a == name else 1 - value) * cell['games'])
                counts = by_players.setdefault(int(cell['cell'].rsplit('/', 1)[1].rstrip('p')), [0, 0])
                counts[0] += above
                counts[1] += cell['games']
    from wonders.agents.bench import wilson_interval

    lines += ['', '| players | games | ranked above bundled (95% CI) |', '|---|---|---|']
    for n, (above, g) in sorted(by_players.items()):
        lo, hi = wilson_interval(above, g)
        lines.append(f'| {n} | {g} | {100 * above / g:.1f}% ({100 * lo:.1f}–{100 * hi:.1f}) |')
    lines += ['', f'| category (mean VP) | `{name}` | bundled | Δ |', '|---|---|---|---|']
    for c in CATEGORIES:
        a, b = ours['categories'][c], theirs['categories'][c]
        if not a and not b:
            continue  # e.g. leaders without the Leaders expansion
        lines.append(f'| {c} | {a:.2f} | {b:.2f} | {a - b:+.2f} |')
    cells = ','.join(card['cells'])
    model_arg = f" --model {name}" if card.get('extension') else ''
    lines += [
        '',
        f"Shuffled seats, other seats {result['filler']}; {card['games_per_cell']} games per cell, seed "
        f"{card['seed']}; checkpoint `{card['checkpoint_sha256']}`, bundled `{card['opponent_sha256']}`, rules "
        f"version {card['rules_version']}. Reproduce (one command per combination, then `wonders ext summary`):",
        '',
        f"`wonders ext scorecard {card['source']}{model_arg} --cells {cells} "
        f"--games {card['games_per_cell']} --seed {card['seed']}`",
        '',
    ]
    return '\n'.join(lines)


def summary(directory: PathLike) -> str:
    """One Markdown report from a folder of check reports (``*.md``) and scorecards (``*.json``)."""
    directory = Path(directory)
    reports = sorted(directory.rglob('*.md'))
    cards = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(directory.rglob('*.json'))]
    failed = sum('#### ❌' in p.read_text(encoding='utf-8') for p in reports)
    head = '❌ some checks failed' if failed else '✅ all checks passed'
    lines = ['## Contribution checks', '', head + '.', '']
    lines += [p.read_text(encoding='utf-8') for p in reports]
    lines += [scorecard_markdown(card) for card in merge_scorecards(cards)]
    lines += ['<sub>Run by `.github/workflows/contributions.yml` with the base branch as referee.</sub>', '']
    return '\n'.join(lines)


# -- leaderboard -----------------------------------------------------------------------------------------
LEADERBOARD_INTRO = """# Leaderboard

Contributed networks against the bundled one (`wonders/ml/pretrained`), measured on merge by
`.github/workflows/leaderboard.yml`: the same tables, seats shuffled every game, the other seats
heuristic agents, every expansion combination at 3–7 players. *Ranked above* is the share of games
the model finished above the bundled network. A model whose lower bound is above 50% is a candidate
for the next bundled network. Contribute one: see `extensions/README.md`.
"""


def update_leaderboard(card: Dict[str, Any], data_path: PathLike, markdown_path: PathLike, date: str = '') -> None:
    """Add or replace a merged scorecard's entry in ``leaderboard.json`` and re-render ``LEADERBOARD.md``."""
    data_path, markdown_path = Path(data_path), Path(markdown_path)
    data = json.loads(data_path.read_text(encoding='utf-8')) if data_path.is_file() else {'entries': []}
    found = _pair(card['result'], card['model'], BUNDLED)
    if found is None:
        raise ValueError('the scorecard has no games between the model and the bundled network')
    share, ci, games = found
    stats = _weighted(card['result'], card['model'])
    entry = {
        'extension': card.get('extension', ''),
        'model': card['model'],
        'version': card.get('version', ''),
        'author': card.get('author', ''),
        'ranked_above': round(share, 4),
        'ci': [round(ci[0], 4), round(ci[1], 4)],
        'games': games,
        'first_place': round(stats['first'], 4),
        'mean_score': round(stats['score'], 2),
        'checkpoint_sha256': card['checkpoint_sha256'],
        'opponent_sha256': card['opponent_sha256'],
        'rules_version': card['rules_version'],
        'date': date or datetime.date.today().isoformat(),
    }
    entries = [
        e for e in data['entries'] if (e.get('extension'), e.get('model')) != (entry['extension'], entry['model'])
    ]
    entries.append(entry)
    entries.sort(key=lambda e: (-e['ranked_above'], e['model']))
    data['entries'] = entries
    data_path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    markdown_path.write_text(leaderboard_markdown(data, card['opponent_sha256']), encoding='utf-8')


def leaderboard_markdown(data: Dict[str, Any], current: str = '') -> str:
    lines = [LEADERBOARD_INTRO]
    if not data.get('entries'):
        return '\n'.join(lines + ['No contributed models yet.', ''])
    lines += [
        '| # | model | extension | author | ranked above bundled (95% CI) | games | first place | mean score |'
        ' rules | measured |',
        '|---|---|---|---|---|---|---|---|---|---|',
    ]
    stale = False
    for i, e in enumerate(data['entries'], 1):
        mark = ''
        if current and e['opponent_sha256'] != current:
            mark, stale = ' †', True
        lo, hi = e['ci']
        lines.append(
            f"| {i} | `{e['model']}` | {e['extension']} {e['version']} | {e['author']} | "
            f"{100 * e['ranked_above']:.1f}% ({100 * lo:.1f}–{100 * hi:.1f}) | {e['games']} | "
            f"{100 * e['first_place']:.1f}% | {e['mean_score']:.1f} | {e['rules_version']} | {e['date']}{mark} |"
        )
    if stale:
        lines += ['', '† measured against an earlier bundled network.']
    return '\n'.join(lines) + '\n'
