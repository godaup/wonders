"""Card and board strength in the expansion builder (:mod:`wonders.agents.strength` behind the GUI).

Three levels, each in VP of score margin (a card relative to discarding it, a board relative to an
average official side):

* **while editing** — :func:`rate` values an unsaved card or board with the rate card (milliseconds);
* **after saving** — a background job (:class:`Jobs`, mode ``'items'``) simulates every card and board
  whose definition changed since its last simulation, one at a time in base-game decks;
* **on request** — a background job (mode ``'expansion'``) simulates the whole expansion at once and
  recalibrates the rate card to it (:func:`~wonders.agents.strength.recalibrate`).

Results live next to the saved games, in ``$WONDERS_HOME/builder/<id>.strength.json`` (outside the
extension folder, which stays a pure data extension), keyed by a hash of each item's definition, so a
changed card is shown as changed until it is simulated again. :func:`overview` gathers the lot for the
builder's overview page.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import statistics
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from wonders import storage
from wonders.agents import strength as st
from wonders.cards.content import build_card, build_wonder, load_content
from wonders.extensions import ensure_loaded
from wonders.ruleset import EXPANSIONS, rules_of

FOLDER = 'builder'
CARD_SAMPLES = 40  # per card after saving: about ±1 VP for Age III cards, ±1.5 for Age I
WONDER_SAMPLES = 30
MAX_EXPANSION_GAMES = 2000


def _path(project_id: str) -> Path:
    return storage.home() / FOLDER / f'{project_id}.strength.json'


def load(project_id: str) -> Dict[str, Any]:
    """The stored simulation results of a project (empty when there are none)."""
    path = _path(project_id)
    if not path.is_file():
        return {'cards': {}, 'wonders': {}, 'expansion': None}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {'cards': {}, 'wonders': {}, 'expansion': None}


def _save(project_id: str, data: Dict[str, Any]) -> None:
    storage.write_json(_path(project_id), data)


def forget(project_id: str) -> None:
    """Drop a project's results (when it is deleted)."""
    _path(project_id).unlink(missing_ok=True)


def spec_hash(spec: Dict[str, Any], colors: Sequence[Dict[str, Any]] = (), rules: Any = None) -> str:
    """A short hash of an item's definition (and what its value depends on: colours and meta rules)."""
    text = json.dumps([spec, list(colors), rules], sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]


def _reference(project_id: str = '') -> st.Reference:
    """The base-game reference positions, with the project's meta rules once it is loaded."""
    ensure_loaded()
    ref = st.reference()
    if project_id and project_id in EXPANSIONS:
        rules = rules_of(('base', project_id))
        if rules != ref.rules:
            ref = dataclasses.replace(ref, rules=rules)
    return ref


def _official_range(values: List[float], value: float) -> Optional[Dict[str, Any]]:
    if not values:
        return None
    below = sum(v < value for v in values)
    return {
        'min': round(values[0], 2),
        'median': round(statistics.median(values), 2),
        'max': round(values[-1], 2),
        'count': len(values),
        'share_below': round(below / len(values), 2),
    }


def _corrected(value: float, correction: Optional[Sequence[float]]) -> Optional[float]:
    return None if not correction else correction[0] + correction[1] * value


def rate(
    kind: str, spec: Dict[str, Any], colors: Sequence[Dict[str, Any]] = (), project_id: str = ''
) -> Dict[str, Any]:
    """The rate card for an unsaved card or board side (raises ``ValueError`` when it does not build)."""
    from wonders.play.builder import clean_card, clean_color, clean_wonder, define_colors

    define_colors([clean_color(c) for c in colors if isinstance(c, dict)])
    cal = st.default_calibration()
    ref = _reference(project_id)
    correction = (load(project_id).get('correction') or {}) if project_id else {}
    if kind == 'card':
        card_spec = clean_card(spec)
        card_spec.setdefault('name', '?')
        card_spec.setdefault('color', 'blue')
        card = build_card(card_spec, 'preview')
        rating = st.rate_card(card, ref, cal)
        rating.corrected = _corrected(rating.value, correction.get('cards'))
        best = rating.corrected if rating.corrected is not None else rating.value
        return {
            'kind': 'card',
            'rating': rating.as_dict(),
            'official': _official_range(cal.official(card.age, card.color.value), best)
            or _official_range(cal.official(card.age, None), best),
            'same_colour': bool(cal.official(card.age, card.color.value)),
            'uncertainty': cal.error.get('cards_loo_rmse'),
        }
    if kind == 'wonder':
        wonder_spec = clean_wonder(spec)
        wonder_spec.setdefault('name', '?')
        wonder = build_wonder(wonder_spec)
        rating = st.rate_wonder(wonder, ref, cal)
        rating.corrected = _corrected(rating.value, correction.get('wonders'))
        best = rating.corrected if rating.corrected is not None else rating.value
        return {
            'kind': 'wonder',
            'rating': rating.as_dict(),
            'stages': [r.as_dict() for r in st.rate_stages(wonder, ref, cal)],
            'official': _official_range(sorted(v['value'] for v in cal.boards.values()), best),
            'uncertainty': cal.error.get('boards_rmse'),
        }
    raise ValueError(f'unknown kind {kind!r}')


def _item_specs(project: Dict[str, Any]) -> Dict[str, List[Any]]:
    cards = [(c.get('name', '?'), c) for c in project['cards']]
    wonders = [(f"{w.get('name', '?')} {w.get('side', 'A')}", w) for w in project['wonders']]
    return {'cards': cards, 'wonders': wonders}


def overview(project_id: str, jobs: Optional['Jobs'] = None) -> Dict[str, Any]:
    """Every card and board side of a saved project with its rate card, simulations and best estimate."""
    from wonders.play.builder import load_project

    project = load_project(project_id)
    stored = load(project_id)
    expansion = stored.get('expansion') or {}
    rules = project.get('rules')
    out: Dict[str, Any] = {'cards': [], 'wonders': []}
    for section, items in _item_specs(project).items():
        kind = 'card' if section == 'cards' else 'wonder'
        for key, spec in items:
            h = spec_hash(spec, project['colors'], rules)
            row: Dict[str, Any] = {'name': key, 'hash': h}
            try:
                rated = rate(kind, spec, project['colors'], project_id)
                row.update(rated)
            except Exception as exc:  # a broken item still gets a row
                row['problem'] = str(exc)
            item = stored.get(section, {}).get(key)
            if item:
                row['simulated'] = {**item['estimate'], 'stale': item['hash'] != h}
            sim = (expansion.get(section) or {}).get(key)
            if sim:
                row['expansion'] = {**sim['estimate'], 'stale': sim['hash'] != h}
            row['estimate'] = _best(row)
            if spec.get('age'):
                row['age'] = spec.get('age')
            out[section].append(row)
    out['expansion'] = {k: expansion[k] for k in ('games', 'players', 'correction') if k in expansion} or None
    out['job'] = jobs.state(project_id) if jobs is not None else None
    return out


def _best(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The best available estimate: a fresh expansion simulation, a fresh item simulation, the rate card."""
    for key, source in (('expansion', 'expansion'), ('simulated', 'item')):
        sim = row.get(key)
        if sim and not sim['stale'] and sim.get('samples'):
            return {'value': sim['value'], 'se': sim['se'], 'source': source}
    rating = row.get('rating')
    if rating:
        value = rating.get('corrected', rating['value'])
        return {'value': value, 'se': row.get('uncertainty'), 'source': 'rate'}
    return None


class Jobs:
    """Background simulations for the builder: one at a time, with progress for polling."""

    def __init__(self, processes: Optional[int] = None) -> None:
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.current: Dict[str, Any] = {}
        self.processes = processes if processes is not None else max(1, (os.cpu_count() or 2) - 1)

    def state(self, project_id: str = '') -> Optional[Dict[str, Any]]:
        with self.lock:
            job = dict(self.current)
        if not job or (project_id and job.get('project') != project_id):
            return None
        return job

    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, project_id: str, mode: str = 'items', games: int = 200, wait: bool = False) -> Dict[str, Any]:
        """Start simulating ``project_id`` (``'items'``: what changed; ``'expansion'``: everything together)."""
        if mode not in ('items', 'expansion'):
            raise ValueError(f'unknown mode {mode!r}')
        ensure_loaded()  # after a restart, saved projects are found on first use
        if project_id not in EXPANSIONS:
            raise ValueError(f'save {project_id!r} without errors first')
        with self.lock:
            if self.busy():
                return dict(self.current)
            self.current = {'project': project_id, 'mode': mode, 'progress': 0.0, 'running': True, 'label': ''}
            target = self._items if mode == 'items' else self._expansion
            self.thread = threading.Thread(target=self._guard, args=(target, project_id, games), daemon=True)
            self.thread.start()
        if wait:
            self.thread.join()
        return self.state(project_id) or {}

    def _guard(self, target: Any, project_id: str, games: int) -> None:
        try:
            target(project_id, games)
            self._set(running=False, progress=1.0, label='')
        except Exception as exc:  # reported to the page
            self._set(running=False, error=str(exc))

    def _set(self, **values: Any) -> None:
        with self.lock:
            self.current.update(values)

    def _items(self, project_id: str, _games: int) -> None:
        from wonders.play.builder import load_project

        project = load_project(project_id)
        content = load_content(project_id)
        stored = load(project_id)
        rules = project.get('rules')
        todo = []
        for section, items in _item_specs(project).items():
            for key, spec in items:
                h = spec_hash(spec, project['colors'], rules)
                if (stored.get(section, {}).get(key) or {}).get('hash') != h:
                    todo.append((section, key, spec, h))
        for k, (section, key, spec, h) in enumerate(todo):
            self._set(progress=k / max(1, len(todo)), label=key)
            seed = int(h, 16) % 2**31
            if section == 'cards':
                card = content.card(spec['name'])
                if card.color.value == 'white':
                    continue  # leaders are drafted, not dealt: no paired build in an Age
                estimate = st.simulate_card(card, CARD_SAMPLES, seed, players=(3, 4, 5, 6, 7))
            else:
                wonder = content.wonder(spec['name'], spec.get('side', 'A'))
                expansions = ('base', *wonder.requires)
                estimate = st.simulate_wonder(wonder, WONDER_SAMPLES, seed, expansions=expansions)
            stored = load(project_id)  # keep what another job may have written meanwhile
            stored.setdefault(section, {})[key] = {'hash': h, 'estimate': estimate.as_dict()}
            _save(project_id, stored)

    def _expansion(self, project_id: str, games: int) -> None:
        from wonders.play.builder import load_project

        games = max(10, min(int(games), MAX_EXPANSION_GAMES))
        project = load_project(project_id)
        result = st.simulate_expansion(
            project_id, games, processes=self.processes, progress=lambda p: self._set(progress=p), context='spawn'
        )
        rules = project.get('rules')
        hashes = {
            section: {key: spec_hash(spec, project['colors'], rules) for key, spec in items}
            for section, items in _item_specs(project).items()
        }
        stored = load(project_id)
        expansion: Dict[str, Any] = {'games': games, 'cards': {}, 'wonders': {}}
        correction: Dict[str, List[float]] = {}
        for section, kind in (('cards', 'card'), ('wonders', 'wonder')):
            rates, values, errors = [], [], []
            for key, estimate in result[section].items():
                if not estimate.samples or key not in hashes[section]:
                    continue
                expansion[section][key] = {'hash': hashes[section][key], 'estimate': estimate.as_dict()}
                spec = dict(_item_specs(project)[section])[key]
                try:
                    rated = rate(kind, spec, project['colors'])  # uncorrected
                except Exception:
                    continue
                if estimate.samples > 1 and math.isfinite(estimate.se):
                    rates.append(rated['rating']['value'])
                    values.append(estimate.value)
                    errors.append(estimate.se)
            if rates:
                correction[section] = [round(x, 4) for x in st.recalibrate(rates, values, errors)]
        expansion['correction'] = correction
        stored['expansion'] = expansion
        stored['correction'] = correction
        _save(project_id, stored)


__all__ = ['Jobs', 'forget', 'load', 'overview', 'rate', 'spec_hash']
