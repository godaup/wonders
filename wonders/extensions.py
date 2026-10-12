"""Extensions: custom expansions, card effects, agents, models and translations.

Everything the game can be extended with goes through one :class:`Registry` (:data:`REGISTRY`):

* **expansions** — cards and wonder boards in the same JSON format as the bundled ones;
* **card effects** — new effect types for those JSON files (code);
* **agents** — scripted agents for the CLI, the benchmark and the GUI (code);
* **models** — trained network checkpoints, usable as ``neural:<name>``;
* **locales** — translations (a new language, or more strings for one; see :mod:`wonders.i18n`).

Extensions come from two sources, discovered once and lazily (:func:`ensure_loaded`):

* **Folders** (data only) — every sub-folder of the extensions homes (``$WONDERS_EXTENSIONS``, several
  separated by ``os.pathsep``; default ``~/.wonders/extensions``) holding a ``manifest.json``. They may
  bring expansions, models and locales, never code, so they are safe to load automatically.
* **The catalog** (data only) — the reviewed community extensions of the repository's ``extensions/``
  folder, shipped with the package (:func:`catalog_homes`). Each one loads only once it is switched on
  (:func:`activate`; the switches live in ``$WONDERS_HOME/extensions.json``), or when a game record
  needs one of its expansions (:func:`provide`).
* **Plugins** (code) — installed Python packages that declare an entry point in the
  ``wonders.extensions`` group pointing to ``register(registry)``. They may also add effects and
  agents. Code only arrives through an explicit ``pip install``; nothing is ever downloaded.

Each extension has a manifest (id, name, version, author, licence, the rules version it targets)
and a content hash, which game records keep so a game replays only with identical content.
``WONDERS_NO_EXTENSIONS=1`` disables discovery (e.g. for reproducible tests).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

ENTRY_POINT_GROUP = 'wonders.extensions'
MANIFEST = 'manifest.json'
HOME_VARIABLE = 'WONDERS_EXTENSIONS'
CATALOG_VARIABLE = 'WONDERS_CATALOG'
SETTINGS = 'extensions.json'  # under the storage home: which catalog extensions are switched on
DISABLE_VARIABLE = 'WONDERS_NO_EXTENSIONS'


@dataclass
class Manifest:
    """What an extension declares about itself (``manifest.json``)."""

    id: str
    name: str
    version: str = '0.0.0'
    author: str = ''
    license: str = ''
    description: str = ''
    rules_version: Optional[int] = None  # the rules version it was made for (wonders.ruleset.RULES_VERSION)
    expansions: List[Dict[str, Any]] = field(default_factory=list)  # {id, name, file[, starting_coins]}
    models: List[Dict[str, Any]] = field(default_factory=list)  # {name, file, description}
    locales: List[Dict[str, Any]] = field(default_factory=list)  # {lang, file}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Manifest':
        missing = [k for k in ('id', 'name') if not data.get(k)]
        if missing:
            raise ValueError(f'manifest lacks {", ".join(missing)}')
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)


@dataclass
class Extension:
    """A loaded extension: its manifest, where it came from and a hash of its content."""

    manifest: Manifest
    source: str  # 'builtin' | 'folder' | 'catalog' | 'plugin'
    location: str = ''
    digest: str = ''
    provides: List[str] = field(default_factory=list)  # e.g. ['expansion:myexp', 'model:mynet', 'effect:x']


@dataclass
class ModelEntry:
    name: str
    path: str
    description: str = ''
    extension: str = ''


class Registry:
    """The one place extensions register into; the built-in registries are updated alongside."""

    def __init__(self) -> None:
        self.extensions: Dict[str, Extension] = {}
        self.expansion_data: Dict[str, Union[Path, Dict[str, Any]]] = {}  # expansion id -> JSON file or data
        self.expansion_owner: Dict[str, str] = {}  # expansion id -> extension id
        self.models: Dict[str, ModelEntry] = {}
        self.locales: Dict[str, Dict[str, str]] = {}
        self.agent_blurbs: Dict[str, str] = {}
        self._current: Optional[Extension] = None  # the extension registering right now

    # -- registration API (used by folders and plugins) ---------------------------------------
    def register_effect(
        self, type_name: str, builder: Callable[[Dict[str, Any]], Any], schema: Optional[Any] = None
    ) -> None:
        """A new effect type for card / wonder JSON: ``builder(spec) -> Effect``.

        ``schema`` (a :class:`~wonders.cards.schema.EffectSchema`) describes its fields, so the GUI's
        expansion builder offers a form for it; without one the builder edits it as JSON.
        """
        from wonders.cards.content import EFFECT_BUILDERS
        from wonders.cards.schema import EFFECT_SCHEMAS

        if type_name in EFFECT_BUILDERS:
            raise ValueError(f'effect type {type_name!r} already exists')
        EFFECT_BUILDERS[type_name] = builder
        if schema is not None:
            EFFECT_SCHEMAS[type_name] = schema
        self._provides(f'effect:{type_name}')

    def register_agent(self, spec: Any, blurb: str = '') -> None:
        """A scripted agent (a :class:`~wonders.agents.bench.AgentSpec`) for the CLI, benchmarks and GUI."""
        from wonders.agents.bench import AGENTS

        if spec.name in AGENTS or spec.name == 'neural' or spec.name.startswith('neural:'):
            raise ValueError(f'agent name {spec.name!r} is taken')
        AGENTS[spec.name] = spec
        if blurb:
            self.agent_blurbs[spec.name] = blurb
        self._provides(f'agent:{spec.name}')

    def register_expansion(
        self,
        expansion_id: str,
        name: str,
        data: Union[str, Path, Dict[str, Any]],
        starting_coins: Optional[int] = None,
    ) -> None:
        """An expansion: cards, wonders, colours and meta rules as a JSON file (path) or already-parsed data.

        The data's ``"rules"`` override meta rules (:data:`wonders.ruleset.RULES`); ``starting_coins`` is the
        older shorthand for ``rules['starting_coins']``.
        """
        from wonders.ruleset import EXPANSIONS, Expansion, check_rules

        if expansion_id in EXPANSIONS:
            raise ValueError(f'expansion id {expansion_id!r} is taken')
        source = Path(data) if isinstance(data, (str, Path)) else data
        raw = json.loads(source.read_text(encoding='utf-8')) if isinstance(source, Path) else source
        rules = dict(raw.get('rules') or {})
        if starting_coins is not None:
            rules.setdefault('starting_coins', starting_coins)
        problems = check_rules(rules)
        if problems:
            raise ValueError(f'expansion {expansion_id!r}: ' + '; '.join(problems))
        EXPANSIONS[expansion_id] = Expansion(expansion_id, name, rules=rules)
        self.expansion_data[expansion_id] = source
        self.expansion_owner[expansion_id] = self._current.manifest.id if self._current else ''
        self._provides(f'expansion:{expansion_id}')

    def register_model(self, name: str, path: Union[str, Path], description: str = '') -> None:
        """A trained checkpoint, usable as ``neural:<name>`` (loaded with ``weights_only``, never as code)."""
        if name in self.models or name == 'neural':
            raise ValueError(f'model name {name!r} is taken')
        owner = self._current.manifest.id if self._current else ''
        self.models[name] = ModelEntry(name, str(path), description, owner)
        self._provides(f'model:{name}')

    def register_locale(self, lang: str, strings: Dict[str, str]) -> None:
        """Translated strings for a language (merged with what is already registered for it)."""
        self.locales.setdefault(lang, {}).update({str(k): str(v) for k, v in strings.items()})
        self._provides(f'locale:{lang}')

    # -- queries ------------------------------------------------------------------------------------
    def expansion_raw(self, expansion_id: str) -> Optional[Dict[str, Any]]:
        """The JSON data of an extension expansion (``None`` for built-in ones)."""
        data = self.expansion_data.get(expansion_id)
        if data is None:
            return None
        if isinstance(data, Path):
            return json.loads(data.read_text(encoding='utf-8'))
        return data

    def pins(self, expansion_ids: Any) -> Dict[str, Dict[str, str]]:
        """``{extension id: {version, hash}}`` for the extensions behind ``expansion_ids`` (for game records)."""
        out: Dict[str, Dict[str, str]] = {}
        for exp in expansion_ids:
            owner = self.expansion_owner.get(exp)
            if owner and owner in self.extensions:
                ext = self.extensions[owner]
                out[owner] = {'version': ext.manifest.version, 'hash': ext.digest}
        return out

    def _provides(self, item: str) -> None:
        if self._current is not None:
            self._current.provides.append(item)


REGISTRY = Registry()
_LOADED = False


def extensions_homes() -> List[Path]:
    """Where folder extensions live: ``$WONDERS_EXTENSIONS``, else ``extensions`` under the storage home.

    The variable may name several folders separated by ``os.pathsep``, e.g. a checkout's ``extensions/``
    and your own.
    """
    from wonders.storage import home

    configured = [Path(p) for p in os.environ.get(HOME_VARIABLE, '').split(os.pathsep) if p]
    return configured or [home() / 'extensions']


def extensions_home() -> Path:
    """The first extensions folder: where ``wonders ext new`` creates extensions."""
    return extensions_homes()[0]


def _digest(folder: Path, manifest: Manifest) -> str:
    h = hashlib.sha256((folder / MANIFEST).read_bytes())
    for entry in manifest.expansions + manifest.models + manifest.locales:
        path = folder / entry['file']
        if path.is_file():
            h.update(path.read_bytes())
    return h.hexdigest()[:12]


def read_manifest(folder: Union[str, Path]) -> Manifest:
    """The manifest of an extension folder."""
    return Manifest.from_dict(json.loads((Path(folder) / MANIFEST).read_text(encoding='utf-8')))


def load_folder(folder: Union[str, Path], registry: Optional[Registry] = None, source: str = 'folder') -> Extension:
    """Load one data-only extension folder (``manifest.json`` + its files) into the registry."""
    registry = registry or REGISTRY
    folder = Path(folder)
    manifest = read_manifest(folder)
    if manifest.id in registry.extensions:
        raise ValueError(f'extension {manifest.id!r} is already loaded')
    ext = Extension(manifest, source, str(folder), _digest(folder, manifest))
    registry._current = ext
    registry.extensions[manifest.id] = ext
    try:
        for e in manifest.expansions:
            registry.register_expansion(e['id'], e.get('name', e['id']), folder / e['file'], e.get('starting_coins'))
            _build_check(registry, e['id'])
        for m in manifest.models:
            registry.register_model(m['name'], folder / m['file'], m.get('description', ''))
        for loc in manifest.locales:
            registry.register_locale(loc['lang'], json.loads((folder / loc['file']).read_text(encoding='utf-8')))
    except Exception:
        unload(manifest.id, registry)  # nothing half-loaded: the menu and renderers load every expansion
        raise
    finally:
        registry._current = None
    return ext


def _build_check(registry: Registry, expansion_id: str) -> None:
    """Build an expansion's colours, cards and wonders once, so a broken file fails to load at all."""
    from wonders.cards.content import build_card, build_wonder, define_colors

    raw = registry.expansion_raw(expansion_id) or {}
    define_colors(raw)
    for spec in raw.get('cards', []):
        try:
            build_card(spec, expansion_id)
        except Exception as exc:
            raise ValueError(f"card {spec.get('name', '?')!r}: {exc}") from exc
    for spec in raw.get('wonders', []):
        try:
            build_wonder(spec)
        except Exception as exc:
            raise ValueError(f"wonder {spec.get('name', '?')!r}: {exc}") from exc


def unload(extension_id: str, registry: Optional[Registry] = None) -> None:
    """Forget a folder extension's expansions and models (its locale strings stay merged)."""
    from wonders.ruleset import EXPANSIONS

    registry = registry or REGISTRY
    ext = registry.extensions.pop(extension_id, None)
    if ext is None:
        return
    for item in ext.provides:
        kind, _, name = item.partition(':')
        if kind == 'expansion':
            EXPANSIONS.pop(name, None)
            registry.expansion_data.pop(name, None)
            registry.expansion_owner.pop(name, None)
        elif kind == 'model':
            registry.models.pop(name, None)


def reload_folder(folder: Union[str, Path], registry: Optional[Registry] = None) -> Extension:
    """Load a folder extension again after its files changed (e.g. saved by the expansion builder)."""
    registry = registry or REGISTRY
    folder = Path(folder)
    manifest = Manifest.from_dict(json.loads((folder / MANIFEST).read_text(encoding='utf-8')))
    old = registry.extensions.get(manifest.id)
    if old is not None and old.source != 'folder':
        raise ValueError(f'extension {manifest.id!r} is a plugin, not a folder')
    unload(manifest.id, registry)
    return load_folder(folder, registry)


def load_plugin(
    manifest: Manifest, register: Callable[[Registry], None], registry: Optional[Registry] = None
) -> Extension:
    """Run a plugin's ``register(registry)`` under its manifest (also handy for tests)."""
    registry = registry or REGISTRY
    if manifest.id in registry.extensions:
        raise ValueError(f'extension {manifest.id!r} is already loaded')
    ext = Extension(manifest, 'plugin', getattr(register, '__module__', ''), manifest.version)
    registry._current = ext
    try:
        register(registry)
    finally:
        registry._current = None
    registry.extensions[manifest.id] = ext
    return ext


def plugin_manifest(target: Any) -> Optional[Dict[str, Any]]:
    """The ``MANIFEST`` dict of an entry point's target: the ``register`` function's module (or the target)."""
    info = getattr(target, 'MANIFEST', None)
    if info is None:
        info = getattr(sys.modules.get(getattr(target, '__module__', '') or ''), 'MANIFEST', None)
    return info if isinstance(info, dict) else None


def _load_entry_points(registry: Registry) -> None:
    for ep in metadata.entry_points(group=ENTRY_POINT_GROUP):
        try:
            target = ep.load()
            info = plugin_manifest(target) or {'id': ep.name, 'name': ep.name}
            version = ''
            if ep.dist is not None:
                version = ep.dist.version
            manifest = Manifest.from_dict({'version': version, **info})
            load_plugin(manifest, target, registry)
        except Exception as exc:  # a broken plugin must not break the game
            warnings.warn(f'extension plugin {ep.name!r} failed to load: {exc}', stacklevel=2)


def ensure_loaded() -> Registry:
    """Discover folder extensions and plugins once (no-op afterwards or with ``WONDERS_NO_EXTENSIONS``)."""
    global _LOADED
    if _LOADED:
        return REGISTRY
    _LOADED = True
    if os.environ.get(DISABLE_VARIABLE):
        return REGISTRY
    _load_entry_points(REGISTRY)  # first: folders may use the effect types plugins add
    for home in extensions_homes():
        if not home.is_dir():
            continue
        for folder in sorted(p for p in home.iterdir() if (p / MANIFEST).is_file()):
            try:
                load_folder(folder)
            except Exception as exc:  # a broken folder must not break the game
                warnings.warn(f'extension folder {folder} failed to load: {exc}', stacklevel=2)
    active = set(active_ids())
    for entry in catalog():
        if entry.manifest.id in active and entry.manifest.id not in REGISTRY.extensions:  # your own folder wins
            try:
                load_folder(entry.folder, REGISTRY, 'catalog')
            except Exception as exc:
                warnings.warn(f'catalog extension {entry.manifest.id!r} failed to load: {exc}', stacklevel=2)
    return REGISTRY


# -- the catalog: community extensions shipped with the package, switched on and off -------------------------


def catalog_homes() -> List[Path]:
    """Where the catalog lives: ``$WONDERS_CATALOG`` (``os.pathsep``-separated), else the package's own copy.

    A built package carries the repository's ``extensions/`` as ``wonders/catalog``; an editable install
    (a checkout) reads ``extensions/`` next to the package directly.
    """
    if os.environ.get(CATALOG_VARIABLE) is not None:
        return [Path(p) for p in os.environ[CATALOG_VARIABLE].split(os.pathsep) if p]
    package = Path(__file__).resolve().parent
    for folder in (package / 'catalog', package.parent / 'extensions'):
        if folder.is_dir():
            return [folder]
    return []


@dataclass
class CatalogEntry:
    """One extension of the catalog: what it brings and whether it is switched on and loaded."""

    manifest: Manifest
    folder: Path
    active: bool = False  # switched on (``$WONDERS_HOME/extensions.json``)
    loaded: str = ''  # where the loaded extension of this id comes from ('catalog', 'folder', 'plugin') or ''
    cards: int = 0
    wonders: List[str] = field(default_factory=list)  # board names (each with sides A and B)
    colors: List[str] = field(default_factory=list)  # labels of the card colours it adds
    names: List[str] = field(default_factory=list)  # card names: found by the search too

    def matches(self, query: str) -> bool:
        """Whether every word of ``query`` occurs in the id, name, author, description, cards or boards."""
        m = self.manifest
        expansions = [str(e.get('name', '')) for e in m.expansions]
        text = ' '.join([m.id, m.name, m.author, m.description, *expansions, *self.wonders, *self.colors, *self.names])
        text = text.casefold()
        return all(word in text for word in query.casefold().split())

    def to_dict(self) -> Dict[str, Any]:
        m = self.manifest
        return {
            'id': m.id,
            'name': m.name,
            'version': m.version,
            'author': m.author,
            'license': m.license,
            'description': m.description,
            'expansions': [{'id': e['id'], 'name': e.get('name', e['id'])} for e in m.expansions],
            'cards': self.cards,
            'wonders': self.wonders,
            'colors': self.colors,
            'models': [str(x.get('name', '')) for x in m.models],
            'locales': [str(x.get('lang', '')) for x in m.locales],
            'active': self.active,
            'loaded': self.loaded,
        }


def _catalog_entry(folder: Path, registry: Registry, active: Set[str]) -> CatalogEntry:
    manifest = read_manifest(folder)
    loaded = registry.extensions.get(manifest.id)
    entry = CatalogEntry(manifest, folder, manifest.id in active, loaded.source if loaded else '')
    for e in manifest.expansions:
        try:
            raw = json.loads((folder / e['file']).read_text(encoding='utf-8'))
        except (OSError, ValueError, KeyError):
            continue
        entry.cards += len(raw.get('cards', []))
        entry.names += [str(c.get('name', '')) for c in raw.get('cards', [])]
        entry.wonders += [w for w in dict.fromkeys(str(w.get('name', '')) for w in raw.get('wonders', []))]
        entry.colors += [str(c.get('label') or str(c.get('id', '')).capitalize()) for c in raw.get('colors', [])]
    return entry


def catalog(query: str = '', registry: Optional[Registry] = None) -> List[CatalogEntry]:
    """The catalog's extensions (by name) matching ``query`` (every word, case-insensitive; empty: all)."""
    registry = registry or REGISTRY
    active = set(active_ids())
    entries: Dict[str, CatalogEntry] = {}
    for home in catalog_homes():
        if not home.is_dir():
            continue
        for folder in sorted(p for p in home.iterdir() if (p / MANIFEST).is_file()):
            try:
                entry = _catalog_entry(folder, registry, active)
            except (OSError, ValueError) as exc:
                warnings.warn(f'catalog folder {folder} is unreadable: {exc}', stacklevel=2)
                continue
            entries.setdefault(entry.manifest.id, entry)  # the first catalog home wins
    found = [e for e in entries.values() if e.matches(query)]
    return sorted(found, key=lambda e: (e.manifest.name.casefold(), e.manifest.id))


def _settings_path() -> Path:
    from wonders.storage import home

    return home() / SETTINGS


def active_ids() -> List[str]:
    """The catalog extensions switched on (unreadable settings count as none)."""
    try:
        data = json.loads(_settings_path().read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return []
    active = data.get('active') if isinstance(data, dict) else None
    return [str(i) for i in active] if isinstance(active, list) else []


def activate(extension_id: str, on: bool = True, registry: Optional[Registry] = None) -> CatalogEntry:
    """Switch a catalog extension on (it loads now and at every start) or off (it is unloaded).

    Raises:
        ValueError: the catalog has no such extension, or it fails to load (it then stays off).
    """
    from wonders.storage import write_json

    registry = registry or REGISTRY
    entry = next((e for e in catalog(registry=registry) if e.manifest.id == extension_id), None)
    if entry is None:
        raise ValueError(f'the catalog has no extension {extension_id!r}')
    discover = registry is REGISTRY and not os.environ.get(DISABLE_VARIABLE)
    if discover:
        ensure_loaded()  # first: an extension of the same id in your own folder keeps precedence
    loaded = registry.extensions.get(extension_id)
    if on and loaded is None and (discover or registry is not REGISTRY):
        try:
            load_folder(entry.folder, registry, 'catalog')
        except Exception as exc:
            raise ValueError(f'{extension_id!r} failed to load: {exc}') from exc
    elif not on and loaded is not None and loaded.source == 'catalog':
        unload(extension_id, registry)
    ids = [i for i in active_ids() if i != extension_id] + ([extension_id] if on else [])
    write_json(_settings_path(), {'active': sorted(ids)})
    return _catalog_entry(entry.folder, registry, set(ids))


def provide(expansion_ids: Any) -> None:
    """Make expansions known: discover extensions, then load catalog extensions that bring missing ones.

    A game record or saved game that used a catalog expansion replays even after it was switched off;
    loading it this way does not switch it on.
    """
    from wonders.ruleset import EXPANSIONS

    missing = {i for i in expansion_ids if i not in EXPANSIONS}
    if not missing:
        return
    ensure_loaded()
    missing = {i for i in missing if i not in EXPANSIONS}
    if not missing or os.environ.get(DISABLE_VARIABLE):
        return
    for entry in catalog():
        brings = {str(e.get('id')) for e in entry.manifest.expansions}
        if brings & missing and entry.manifest.id not in REGISTRY.extensions:
            try:
                load_folder(entry.folder, REGISTRY, 'catalog')
            except Exception as exc:
                warnings.warn(f'catalog extension {entry.manifest.id!r} failed to load: {exc}', stacklevel=2)


def _rule_and_colour_problems(filename: str, raw: Dict[str, Any], entry: Dict[str, Any]) -> List[str]:
    """Problems with an expansion file's meta rules and colours (which it defines on the way)."""
    from wonders.cards.base import Color
    from wonders.ruleset import check_rules

    problems = [f'{filename}: {p}' for p in check_rules(raw.get('rules') or {})]
    if 'starting_coins' in entry:
        problems += [f'manifest: {p}' for p in check_rules({'starting_coins': entry['starting_coins']})]
    for color in raw.get('colors', []):
        try:
            Color.from_spec(color)
        except Exception as exc:
            problems.append(f"colour {color.get('id', '?') if isinstance(color, dict) else color!r}: {exc}")
    return problems


def _validate_expansion(folder: Path, entry: Dict[str, Any], known: Set[str]) -> List[str]:
    """Problems in one expansion file of an extension folder."""
    from wonders.cards.content import build_card, build_wonder

    path = folder / entry.get('file', '')
    if not path.is_file():
        return []  # reported as a missing file
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        return [f'{path.name}: not valid JSON ({exc})']
    problems = _rule_and_colour_problems(path.name, raw, entry)
    names: List[str] = []
    for spec in raw.get('cards', []):
        try:
            card = build_card(spec, entry['id'])
        except Exception as exc:
            problems.append(f"card {spec.get('name', '?')!r}: {exc}")
            continue
        if card.name in known:
            problems.append(f'card {card.name!r} clashes with an existing card')
        names.append(card.name)
    for spec in raw.get('wonders', []):
        try:
            build_wonder(spec)
        except Exception as exc:
            problems.append(f"wonder {spec.get('name', '?')!r}: {exc}")
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        problems.append(f'duplicate card names: {", ".join(duplicates)}')
    everything = known | set(names)
    for spec in raw.get('cards', []):
        for link in [spec.get('chain_requires')] + list(spec.get('chain_enables', [])):
            if link and link not in everything:
                problems.append(f"card {spec.get('name')!r}: chain to unknown card {link!r}")
    return problems


def validate(folder: Union[str, Path]) -> List[str]:
    """Problems with an extension folder (an empty list means it is fine).

    Checks the manifest, that every file exists and parses, that every card and wonder builds (known
    effect types, resources, colours), that names do not clash with the base game or each other, and
    that chain links point to existing cards.
    """
    from wonders.cards.content import load_content

    folder = Path(folder)
    try:
        manifest = Manifest.from_dict(json.loads((folder / MANIFEST).read_text(encoding='utf-8')))
    except Exception as exc:
        return [f'{MANIFEST}: {exc}']
    problems = [
        f"missing file {entry.get('file')!r}"
        for entry in manifest.expansions + manifest.models + manifest.locales
        if not (folder / entry.get('file', '')).is_file()
    ]
    known = {c.name for c in load_content('base', 'leaders', 'cities').cards}
    for entry in manifest.expansions:
        problems += _validate_expansion(folder, entry, known)
    return problems


def scaffold(folder: Union[str, Path], extension_id: str, name: str = '', author: str = '') -> Path:
    """Create a minimal data extension (manifest + one example card) to start from."""
    from wonders.ruleset import RULES_VERSION

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {
        'id': extension_id,
        'name': name or extension_id,
        'version': '0.1.0',
        'author': author,
        'license': 'CC-BY-4.0',
        'description': 'A custom expansion.',
        'rules_version': RULES_VERSION,
        'expansions': [{'id': extension_id, 'name': name or extension_id, 'file': 'cards.json'}],
    }
    cards = {
        'cards': [
            {
                'name': f'{name or extension_id} Workshop',
                'color': 'blue',
                'age': 1,
                'at_players': [3, 3, 3, 4, 5, 6, 7],  # one copy per player: every Age deck still deals evenly
                'cost': {'resources': {'wood': 1}},
                'effects': [{'type': 'victory_points', 'points': 2}],
            }
        ]
    }
    (folder / MANIFEST).write_text(json.dumps(manifest, indent=1) + '\n', encoding='utf-8')
    (folder / 'cards.json').write_text(json.dumps(cards, indent=1) + '\n', encoding='utf-8')
    return folder
