"""Extensions: custom expansions, card effects, agents, models and translations.

Everything the game can be extended with goes through one :class:`Registry` (:data:`REGISTRY`):

* **expansions** — cards and wonder boards in the same JSON format as the bundled ones;
* **card effects** — new effect types for those JSON files (code);
* **agents** — scripted agents for the CLI, the benchmark and the GUI (code);
* **models** — trained network checkpoints, usable as ``neural:<name>``;
* **locales** — translated strings for the upcoming translation support.

Extensions come from two sources, discovered once and lazily (:func:`ensure_loaded`):

* **Folders** (data only) — every sub-folder of the extensions home (``$WONDERS_EXTENSIONS``, default
  ``~/.wonders/extensions``) holding a ``manifest.json``. They may bring expansions, models and
  locales, never code, so they are safe to load automatically.
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
import warnings
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

ENTRY_POINT_GROUP = 'wonders.extensions'
MANIFEST = 'manifest.json'
HOME_VARIABLE = 'WONDERS_EXTENSIONS'
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
    expansions: List[Dict[str, Any]] = field(default_factory=list)  # {id, name, file, starting_coins}
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
    source: str  # 'builtin' | 'folder' | 'plugin'
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
    def register_effect(self, type_name: str, builder: Callable[[Dict[str, Any]], Any]) -> None:
        """A new effect type for card / wonder JSON: ``builder(spec) -> Effect``."""
        from wonders.cards.content import EFFECT_BUILDERS

        if type_name in EFFECT_BUILDERS:
            raise ValueError(f'effect type {type_name!r} already exists')
        EFFECT_BUILDERS[type_name] = builder
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
        self, expansion_id: str, name: str, data: Union[str, Path, Dict[str, Any]], starting_coins: int = 3
    ) -> None:
        """An expansion: cards and wonders as a JSON file (path) or already-parsed data."""
        from wonders.ruleset import EXPANSIONS, Expansion

        if expansion_id in EXPANSIONS:
            raise ValueError(f'expansion id {expansion_id!r} is taken')
        EXPANSIONS[expansion_id] = Expansion(expansion_id, name, starting_coins=starting_coins)
        self.expansion_data[expansion_id] = Path(data) if isinstance(data, (str, Path)) else data
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


def extensions_home() -> Path:
    """Where folder extensions live: ``$WONDERS_EXTENSIONS`` or ``~/.wonders/extensions``."""
    return Path(os.environ.get(HOME_VARIABLE) or Path.home() / '.wonders' / 'extensions')


def _digest(folder: Path, manifest: Manifest) -> str:
    h = hashlib.sha256((folder / MANIFEST).read_bytes())
    for entry in manifest.expansions + manifest.models + manifest.locales:
        path = folder / entry['file']
        if path.is_file():
            h.update(path.read_bytes())
    return h.hexdigest()[:12]


def load_folder(folder: Union[str, Path], registry: Optional[Registry] = None) -> Extension:
    """Load one data-only extension folder (``manifest.json`` + its files) into the registry."""
    registry = registry or REGISTRY
    folder = Path(folder)
    manifest = Manifest.from_dict(json.loads((folder / MANIFEST).read_text(encoding='utf-8')))
    if manifest.id in registry.extensions:
        raise ValueError(f'extension {manifest.id!r} is already loaded')
    ext = Extension(manifest, 'folder', str(folder), _digest(folder, manifest))
    registry._current = ext
    try:
        for e in manifest.expansions:
            registry.register_expansion(e['id'], e.get('name', e['id']), folder / e['file'], e.get('starting_coins', 3))
        for m in manifest.models:
            registry.register_model(m['name'], folder / m['file'], m.get('description', ''))
        for loc in manifest.locales:
            registry.register_locale(loc['lang'], json.loads((folder / loc['file']).read_text(encoding='utf-8')))
    finally:
        registry._current = None
    registry.extensions[manifest.id] = ext
    return ext


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


def _load_entry_points(registry: Registry) -> None:
    for ep in metadata.entry_points(group=ENTRY_POINT_GROUP):
        try:
            target = ep.load()
            info = getattr(target, 'MANIFEST', None) or {'id': ep.name, 'name': ep.name}
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
    home = extensions_home()
    if home.is_dir():
        for folder in sorted(p for p in home.iterdir() if (p / MANIFEST).is_file()):
            try:
                load_folder(folder)
            except Exception as exc:  # a broken folder must not break the game
                warnings.warn(f'extension folder {folder} failed to load: {exc}', stacklevel=2)
    _load_entry_points(REGISTRY)
    return REGISTRY


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
    problems: List[str] = []
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


def scaffold(folder: Union[str, Path], extension_id: str, name: str = '') -> Path:
    """Create a minimal data extension (manifest + one example card) to start from."""
    from wonders.ruleset import RULES_VERSION

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {
        'id': extension_id,
        'name': name or extension_id,
        'version': '0.1.0',
        'author': '',
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
                'at_players': [3],
                'cost': {'resources': {'wood': 1}},
                'effects': [{'type': 'victory_points', 'points': 2}],
            }
        ]
    }
    (folder / MANIFEST).write_text(json.dumps(manifest, indent=1) + '\n', encoding='utf-8')
    (folder / 'cards.json').write_text(json.dumps(cards, indent=1) + '\n', encoding='utf-8')
    return folder
