"""A dependency-free local web server for :class:`~wonders.play.session.HumanSession`.

Serves the single-page UI (``static/index.html``) and a small JSON API:

- ``GET /api/options`` — the game menu (expansions, agents, wonders, seat limits)
- ``POST /api/new`` — start a game from a :class:`~wonders.play.session.GameConfig` dict
- ``GET /api/state`` — the human's current view
- ``POST /api/act`` — ``{"index": i}`` plays the i-th legal action (``"payment": [left, right]``
  optionally picks one of its listed splits of the purchase between the neighbours, ``"stage": k`` one of
  its listed wonder stages)
- ``GET /api/record`` — the replayable GameRecord JSON
- ``POST /api/analyse`` / ``GET /api/analysis`` — start / poll the post-game analysis of the human's moves
- ``GET /api/card.svg?name=…`` / ``GET /api/wonder.svg?name=…&side=A[&built=N]`` — renderings
  (``&edition=2``: the second edition's card of that name)
- ``GET /favicon.svg`` — the project logo (``static/logo.svg``, a copy of ``docs/images/logo.svg``)
- ``GET /api/locale?lang=de`` — the translation catalog the page uses for its own texts
- ``GET /api/games`` — saved games (newest first) with the rating after each, and the rating history
- ``POST /api/games/open`` / ``POST /api/games/delete`` — ``{"id": …}``: resume or view / delete a saved game
- ``GET /api/games/record?id=…`` — a saved game's GameRecord JSON
- ``GET /api/achievements`` — every achievement and the games that earned it
- ``GET /api/catalog?q=…`` — the community extensions shipped with the game (search: every word matches)
- ``POST /api/catalog/activate`` — ``{"id": …, "active": true|false}``: switch one on or off
- ``GET /api/builder`` — the expansion builder's forms (effect schemas, meta rules, …) and the projects
- ``GET /api/builder/project?id=…`` / ``GET /api/builder/export?id=…`` — one project / its folder as a zip
- ``POST /api/builder/check`` / ``save`` / ``delete`` / ``preview`` / ``balance`` — see
  :mod:`wonders.play.builder`; these need a JSON body and a local ``Host`` (no cross-site writes)
- ``POST /api/builder/rate`` — the rate card of an unsaved card or board side (:mod:`wonders.play.strength`)
- ``GET /api/builder/strength?id=…`` / ``POST /api/builder/strength`` — every item's strength and the
  simulation job; ``{"id", "mode": "items" | "expansion", "games"}`` starts one

Games are saved after every move to a :class:`~wonders.storage.Store` (``~/.wonders/games``), unless
the server runs without one (``wonders gui --no-save``).

Every route takes ``?lang=<code>`` (default English): names, card texts, menu entries and the
analysis come back translated (see :mod:`wonders.i18n`).

It binds to localhost only and plays one game at a time (one human, all local).
"""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse, urlsplit

from wonders.cards.content import Content, load_content
from wonders.i18n import DEFAULT, ENGLISH, Translator, catalog
from wonders.play import builder, strength
from wonders.play.achievements import achievements, earned
from wonders.play.rating import AGENT_ELO, DEFAULT_AGENT_ELO, START, agent_elo, player_history, rated_game
from wonders.play.session import ConfigError, GameConfig, HumanSession, menu_options
from wonders.ruleset import DEFAULT_EDITION, EDITIONS, EXPANSIONS
from wonders.storage import StorageError, Store
from wonders.viz.render import card_svg, wonder_svg


@lru_cache(maxsize=4)
def _all_content(edition: int = DEFAULT_EDITION) -> Content:
    """Every card and board of one edition (the editions share card names, e.g. Altar, with other values)."""
    from wonders.extensions import ensure_loaded

    ensure_loaded()  # cards of extension expansions render too
    return load_content(*(e.id for e in EXPANSIONS.values() if e.implemented and e.edition in (None, edition)))


def _edition(query: Dict[str, Any]) -> int:
    try:
        edition = int(query.get('edition') or DEFAULT_EDITION)
    except ValueError:
        return DEFAULT_EDITION
    return edition if edition in EDITIONS else DEFAULT_EDITION


@lru_cache(maxsize=1)
def _index_html() -> bytes:
    return resources.files('wonders.play').joinpath('static', 'index.html').read_bytes()


@lru_cache(maxsize=1)
def _favicon() -> bytes:
    return resources.files('wonders.play').joinpath('static', 'logo.svg').read_bytes()


@lru_cache(maxsize=16)
def _translator(lang: str) -> Translator:
    return Translator(lang)  # an unknown language has an empty catalog: English


def _t(query: Dict[str, Any]) -> Translator:
    return _translator(str(query.get('lang') or DEFAULT)[:16])


@lru_cache(maxsize=1024)
def _card_svg(name: str, lang: str = DEFAULT, edition: int = DEFAULT_EDITION) -> Optional[str]:
    card = next((c for c in _all_content(edition).cards if c.name == name), None)
    return None if card is None else card_svg(card, _translator(lang))


@lru_cache(maxsize=128)
def _wonder_svg(
    name: str, side: str, built: str = '0', lang: str = DEFAULT, edition: int = DEFAULT_EDITION
) -> Optional[str]:
    """``built`` is a count (the first stages) or ``s1,s3``: 1-based stages built in any order (Great Wall)."""
    try:
        wonder = _all_content(edition).wonder(name, side)
        if built.startswith('s'):
            stages = [int(x.strip('s')) - 1 for x in built.split(',') if x.strip('s')]
            return wonder_svg(wonder, built=stages, t=_translator(lang))
        return wonder_svg(wonder, built=max(0, int(built or 0)), t=_translator(lang))
    except (KeyError, ValueError):
        return None


class PlayServer:
    """The one game this server hosts, guarded by a lock; ``store`` keeps every game (``None``: none)."""

    def __init__(self, checkpoints: Sequence[str] = (), store: Optional[Store] = None):
        self.checkpoints = tuple(checkpoints)
        self.store = store
        self.session: Optional[HumanSession] = None
        self.lock = threading.Lock()
        self.builder_lock = threading.Lock()  # builder writes change the expansion registry
        self.strength_jobs = strength.Jobs()  # card / board simulations, one at a time
        self._options: Dict[str, Dict[str, Any]] = {}  # per language
        self.analysis: Dict[str, Any] = {'status': 'idle'}  # post-game analysis of the current game

    def options(self, t: Translator = ENGLISH) -> Dict[str, Any]:
        if t.lang not in self._options:
            options = menu_options(self.checkpoints, t)
            for agent in options['agents']:
                agent['elo'] = agent_elo(agent['id'])
            self._options[t.lang] = {**options, 'saving': self.store is not None}
        return self._options[t.lang]

    def _save(self) -> None:
        """Save the current game (called with the lock held)."""
        if self.store is not None and self.session is not None:
            self.store.save_game(self.session.saved())

    def new_game(self, data: Dict[str, Any], t: Translator = ENGLISH) -> Dict[str, Any]:
        config = GameConfig.from_dict(data)
        with self.lock:
            self.session = HumanSession(config, self.checkpoints, t=t)
            self.analysis = {'status': 'idle'}
            self._save()
            return self.session.state(t)

    def state(self, t: Translator = ENGLISH) -> Optional[Dict[str, Any]]:
        with self.lock:
            return None if self.session is None else self.session.state(t)

    def act(
        self, index: int, payment: Optional[Sequence[int]] = None, t: Translator = ENGLISH, stage: Optional[int] = None
    ) -> Dict[str, Any]:
        with self.lock:
            if self.session is None:
                raise ConfigError('no game in progress')
            self.session.act(index, payment, stage)
            self._save()
            return self.session.state(t)

    def record(self) -> Optional[str]:
        with self.lock:
            return None if self.session is None else self.session.record_json()

    # -- saved games -------------------------------------------------------------------------
    def _store(self) -> Store:
        if self.store is None:
            raise ConfigError('games are not saved (the server runs with --no-save)')
        return self.store

    def games(self, t: Translator = ENGLISH) -> Dict[str, Any]:
        """The saved games, newest first, and the human's rating history over the finished ones."""
        saved = self.store.list_games() if self.store is not None else []
        finished = sorted((g for g in saved if rated_game(g)), key=_finished_at)
        history = player_history(r for r in (rated_game(g) for g in finished) if r is not None)
        points = {p.id: p for p in history}
        with self.lock:
            current = self.session.id if self.session is not None else None
        games = []
        for game in reversed(saved):
            entry = {
                **game,
                'current': game['id'] == current,
                'players': [
                    {
                        **p,
                        'wonder_title': t.name(p['wonder']),
                        'elo': None if p['agent'] == 'human' else agent_elo(p['agent']),
                    }
                    for p in game['players']
                ],
                'achievements': earned(game),
            }
            point = points.get(game['id'])
            if point is not None:
                entry['rating'] = {'before': round(point.before), 'after': round(point.after)}
            games.append(entry)
        by_id = {g['id']: g for g in finished}
        return {
            'saving': self.store is not None,
            'games': games,
            'rating': {
                'start': START,
                'current': round(history[-1].after) if history else START,
                'games': len(history),
                'history': [
                    {
                        'id': p.id,
                        'rating': round(p.after, 1),
                        'at': _finished_at(by_id[p.id]),
                        'place': by_id[p.id]['result']['places'][by_id[p.id]['human']],
                        'players': len(by_id[p.id]['players']),
                    }
                    for p in history
                ],
                'agents': dict(AGENT_ELO),
                'default_agent': DEFAULT_AGENT_ELO,
            },
        }

    def open_game(self, game_id: str, t: Translator = ENGLISH) -> Dict[str, Any]:
        """Make a saved game the current one: an unfinished game continues, a finished one shows its result."""
        session = HumanSession.restore(self._store().load_game(game_id), self.checkpoints, t=t)
        with self.lock:
            self.session = session
            self.analysis = {'status': 'idle'}
            return session.state(t)

    def delete_game(self, game_id: str) -> Dict[str, Any]:
        with self.lock:
            self._store().delete_game(game_id)
            if self.session is not None and self.session.id == game_id:
                self.session = None
                self.analysis = {'status': 'idle'}
        return {'deleted': game_id}

    def saved_record(self, game_id: str) -> str:
        record = self._store().load_game(game_id).get('record')
        if record is None:
            raise StorageError(f'saved game {game_id!r} has no record')
        return json.dumps(record, indent=1)

    def achievements(self, t: Translator = ENGLISH) -> Dict[str, Any]:
        """Every achievement (translated) with its progress over the finished games."""
        saved = self.store.list_games() if self.store is not None else []
        finished = sorted((g for g in saved if rated_game(g)), key=_finished_at)
        ratings = [p.after for p in player_history(r for r in (rated_game(g) for g in finished) if r is not None)]
        items = achievements(finished, ratings)
        for item in items:
            item.update(name=t(item['name']), description=t(item['description']))
        return {
            'saving': self.store is not None,
            'achievements': items,
            'games': len(finished),
            'points': sum(item['points'] for item in items if item['count']),
            'max_points': sum(item['points'] for item in items),
        }

    # -- catalog of community extensions -----------------------------------------------------
    def catalog(self, query: str = '') -> Dict[str, Any]:
        from wonders.extensions import catalog

        with self.builder_lock:
            return {'extensions': [e.to_dict() for e in catalog(query)]}

    def catalog_activate(self, body: Dict[str, Any]) -> Dict[str, Any]:
        from wonders.extensions import activate

        with self.builder_lock:
            entry = activate(str(body.get('id', '')), bool(body.get('active', True)))
            self._content_changed()
            return {'extension': entry.to_dict()}

    # -- expansion builder -------------------------------------------------------------------
    def builder_state(self, t: Translator = ENGLISH) -> Dict[str, Any]:
        return {'info': builder.builder_info(t), 'projects': builder.projects()}

    def builder_save(self, body: Dict[str, Any]) -> Dict[str, Any]:
        with self.builder_lock:
            result = builder.save_project(body.get('project') or {}, str(body.get('previous_id') or ''))
            self._content_changed()
            return {**result, 'projects': builder.projects()}

    def builder_delete(self, project_id: str) -> Dict[str, Any]:
        with self.builder_lock:
            builder.delete_project(project_id)
            strength.forget(project_id)
            self._content_changed()
            return {'deleted': project_id, 'projects': builder.projects()}

    def builder_preview(self, body: Dict[str, Any], t: Translator = ENGLISH) -> Dict[str, Any]:
        with self.builder_lock:
            try:
                return {
                    'svg': builder.preview(str(body.get('kind')), body.get('spec') or {}, body.get('colors') or [], t)
                }
            except Exception as exc:  # an unfinished card is normal while editing
                return {'error': builder._reason(exc)}

    def builder_rate(self, body: Dict[str, Any]) -> Dict[str, Any]:
        with self.builder_lock:
            try:
                return strength.rate(
                    str(body.get('kind')), body.get('spec') or {}, body.get('colors') or [], str(body.get('id') or '')
                )
            except Exception as exc:  # an unfinished card is normal while editing
                return {'error': builder._reason(exc)}

    def builder_strength(self, project_id: str) -> Dict[str, Any]:
        with self.builder_lock:
            return strength.overview(project_id, self.strength_jobs)

    def builder_simulate(self, body: Dict[str, Any]) -> Dict[str, Any]:
        project_id = str(body.get('id') or '')
        job = self.strength_jobs.start(project_id, str(body.get('mode') or 'items'), int(body.get('games') or 200))
        return {'job': job}

    def _content_changed(self) -> None:
        """An expansion changed on disk: menus, renderings and image versions are rebuilt."""
        self._options.clear()
        for cache in (_all_content, _card_svg, _wonder_svg):
            cache.cache_clear()

    # -- post-game analysis ------------------------------------------------------------------
    def start_analysis(self) -> Dict[str, Any]:
        """Rate the human's moves of the finished game in a background thread (needs PyTorch)."""
        from wonders.engine.record import GameRecord

        with self.lock:
            if self.session is None or not self.session.done:
                raise ConfigError('finish the game first')
            if self.analysis['status'] in ('running', 'done'):
                return dict(self.analysis)
            record = GameRecord.from_json(self.session.record_json())
            seat = self.session.human
            self.analysis = {'status': 'running', 'done': 0, 'total': 0}
            game = self.session
        threading.Thread(target=self._analyse, args=(record, seat, game), daemon=True).start()
        return dict(self.analysis)

    def _analyse(self, record: Any, seat: int, game: HumanSession) -> None:
        def progress(done: int, total: int) -> None:
            with self.lock:
                if self.session is game:
                    self.analysis.update(done=done, total=total)

        try:
            from wonders.ml.analysis import UnknownContent, analyse
        except ImportError:
            result: Dict[str, Any] = {'status': 'unavailable', 'error': 'post-game analysis needs PyTorch'}
        else:
            try:
                processes = max(1, min(6, (os.cpu_count() or 2) - 1))
                report = analyse(record, seat, processes=processes, progress=progress)
                result = {'status': 'done', 'report': report}
            except UnknownContent as exc:  # e.g. a game with an extension's cards: say so in the page's language
                result = {'status': 'unavailable', 'error': str(exc), 'reason': exc.message}
            except Exception as exc:  # pragma: no cover - surfaced to the page
                result = {'status': 'error', 'error': str(exc)}
        with self.lock:
            if self.session is game:  # a new game started meanwhile: drop the stale result
                self.analysis = result

    def analysis_state(self, t: Translator = ENGLISH) -> Dict[str, Any]:
        with self.lock:
            state = dict(self.analysis)
        if state.get('reason') is not None:
            state['error'] = str(t.render(state.pop('reason')))
        if state.get('status') == 'done' and t.lang != DEFAULT:
            from wonders.ml.analysis import localize

            state['report'] = localize(state['report'], t)
        return state


def _finished_at(game: Dict[str, Any]) -> str:
    return str(game.get('finished_at') or game.get('updated') or '')


Response = Tuple[int, bytes, str, Tuple[Tuple[str, str], ...]]
NOT_FOUND = 'not found'


def _json(payload: Any, status: int = HTTPStatus.OK) -> Response:
    return status, json.dumps(payload).encode('utf-8'), 'application/json', ()


def _error(message: str, status: int = HTTPStatus.BAD_REQUEST) -> Response:
    return _json({'error': message}, status)


def _svg(svg: Optional[str]) -> Response:
    if svg is None:
        return _error(NOT_FOUND, HTTPStatus.NOT_FOUND)
    return HTTPStatus.OK, svg.encode('utf-8'), 'image/svg+xml', (('Cache-Control', 'max-age=86400'),)


def _record(play: PlayServer) -> Response:
    record = play.record()
    if record is None:
        return _error('no game in progress', HTTPStatus.NOT_FOUND)
    disposition = ('Content-Disposition', 'attachment; filename="wonders-game.json"')
    return HTTPStatus.OK, record.encode('utf-8'), 'application/json', (disposition,)


def _guarded(handler: Callable[[], Any]) -> Response:
    try:
        return _json(handler())
    except (builder.BuilderError, ValueError, OSError) as exc:
        return _error(str(exc))


def _export(project_id: str) -> Response:
    try:
        data = builder.export_project(project_id)
    except (builder.BuilderError, OSError) as exc:
        return _error(str(exc), HTTPStatus.NOT_FOUND)
    disposition = ('Content-Disposition', f'attachment; filename="{project_id}.zip"')
    return HTTPStatus.OK, data, 'application/zip', (disposition,)


def _saved_record(play: PlayServer, game_id: str) -> Response:
    try:
        record = play.saved_record(game_id)
    except (StorageError, ConfigError) as exc:
        return _error(str(exc), HTTPStatus.NOT_FOUND)
    disposition = ('Content-Disposition', f'attachment; filename="wonders-{game_id}.json"')
    return HTTPStatus.OK, record.encode('utf-8'), 'application/json', (disposition,)


GET_ROUTES: Dict[str, Callable[[PlayServer, Dict[str, str]], Response]] = {
    '/': lambda play, q: (HTTPStatus.OK, _index_html(), 'text/html; charset=utf-8', ()),
    '/favicon.svg': lambda play, q: (HTTPStatus.OK, _favicon(), 'image/svg+xml', (('Cache-Control', 'max-age=86400'),)),
    '/api/options': lambda play, q: _json(play.options(_t(q))),
    '/api/locale': lambda play, q: _json(catalog(_t(q).lang) if _t(q).lang != DEFAULT else {}),
    '/api/state': lambda play, q: _json({'state': play.state(_t(q))}),
    '/api/record': lambda play, q: _record(play),
    '/api/analysis': lambda play, q: _json(play.analysis_state(_t(q))),
    '/api/games': lambda play, q: _json(play.games(_t(q))),
    '/api/games/record': lambda play, q: _saved_record(play, q.get('id', '')),
    '/api/achievements': lambda play, q: _json(play.achievements(_t(q))),
    '/api/card.svg': lambda play, q: _svg(_card_svg(q.get('name', ''), _t(q).lang, _edition(q))),
    '/api/wonder.svg': lambda play, q: _svg(
        _wonder_svg(q.get('name', ''), q.get('side', 'A'), q.get('built', '0'), _t(q).lang, _edition(q))
    ),
    '/api/catalog': lambda play, q: _json(play.catalog(q.get('q', ''))),
    '/api/builder': lambda play, q: _json(play.builder_state(_t(q))),
    '/api/builder/project': lambda play, q: _guarded(lambda: {'project': builder.load_project(q.get('id', ''))}),
    '/api/builder/export': lambda play, q: _export(q.get('id', '')),
    '/api/builder/strength': lambda play, q: _guarded(lambda: play.builder_strength(q.get('id', ''))),
}
# POST routes get the JSON body and the query string (for ``lang``).
POST_ROUTES: Dict[str, Callable[[PlayServer, Dict[str, Any], Dict[str, str]], Response]] = {
    '/api/new': lambda play, body, q: _json({'state': play.new_game(body, _t(q))}),
    '/api/analyse': lambda play, body, q: _json(play.start_analysis()),
    '/api/games/open': lambda play, body, q: _json({'state': play.open_game(str(body.get('id', '')), _t(q))}),
    '/api/games/delete': lambda play, body, q: _json(play.delete_game(str(body.get('id', '')))),
    '/api/act': lambda play, body, q: _json(
        {'state': play.act(int(body.get('index', -1)), body.get('payment'), _t(q), body.get('stage'))}
    ),
    '/api/catalog/activate': lambda play, body, q: _guarded(lambda: play.catalog_activate(body)),
    '/api/builder/check': lambda play, body, q: _json(builder.check_project(body.get('project') or {})),
    '/api/builder/save': lambda play, body, q: _json(play.builder_save(body)),
    '/api/builder/delete': lambda play, body, q: _json(play.builder_delete(str(body.get('id', '')))),
    '/api/builder/preview': lambda play, body, q: _json(play.builder_preview(body, _t(q))),
    '/api/builder/rate': lambda play, body, q: _json(play.builder_rate(body)),
    '/api/builder/strength': lambda play, body, q: _guarded(lambda: play.builder_simulate(body)),
    '/api/builder/balance': lambda play, body, q: _json(
        builder.balance(str(body.get('id', '')), int(body.get('games', 100)), int(body.get('players', 4)))
    ),
}
LOCAL_HOSTS = ('127.0.0.1', 'localhost', '::1')
LOCAL_ONLY = ('/api/builder', '/api/catalog/')  # writes that change the expansions on this machine


class Handler(BaseHTTPRequestHandler):
    """Routes requests to the :class:`PlayServer` set as ``play`` (see :func:`make_server`)."""

    server_version = 'wonders-play'
    play: PlayServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - quiet by default
        pass

    def _send(self, response: Response) -> None:
        status, body, content_type, extra = response
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        for key, value in extra:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _local(self) -> bool:
        """Whether the request names this machine (a page on another site cannot rebind to it)."""
        try:
            return urlsplit('//' + (self.headers.get('Host') or '')).hostname in LOCAL_HOSTS
        except ValueError:
            return False

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        url = urlparse(self.path)
        route = GET_ROUTES.get('/' if url.path == '/index.html' else url.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path.startswith('/api/builder') and not self._local():
            self._send(_error('the builder only answers local requests', HTTPStatus.FORBIDDEN))
            return
        self._send(route(self.play, query) if route else _error(NOT_FOUND, HTTPStatus.NOT_FOUND))

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        url = urlparse(self.path)
        route = POST_ROUTES.get(url.path)
        if route is None:
            self._send(_error(NOT_FOUND, HTTPStatus.NOT_FOUND))
            return
        if url.path.startswith(LOCAL_ONLY) and (
            not self._local() or not (self.headers.get('Content-Type') or '').startswith('application/json')
        ):  # a JSON content type needs a CORS preflight, which this server never grants: no cross-site writes
            self._send(_error('only local JSON requests may change the expansions', HTTPStatus.FORBIDDEN))
            return
        try:
            length = int(self.headers.get('Content-Length') or 0)
            body = json.loads(self.rfile.read(length) or b'{}')
            if not isinstance(body, dict):
                raise ConfigError('expected a JSON object')
            self._send(route(self.play, body, {k: v[0] for k, v in parse_qs(url.query).items()}))
        except (ConfigError, StorageError, ValueError, TypeError, OSError) as exc:
            self._send(_error(str(exc)))


def make_server(
    host: str = '127.0.0.1', port: int = 8765, checkpoints: Sequence[str] = (), store: Optional[Store] = None
) -> ThreadingHTTPServer:
    """Build (but do not start) the server; ``port=0`` picks a free port; ``store`` saves the games."""
    handler = type('BoundHandler', (Handler,), {'play': PlayServer(checkpoints, store)})
    return ThreadingHTTPServer((host, port), handler)


def serve(
    host: str = '127.0.0.1',
    port: int = 8765,
    open_browser: bool = True,
    checkpoints: Sequence[str] = (),
    save: bool = True,
) -> None:
    """Run the UI until interrupted; ``save`` keeps every game under :func:`wonders.storage.home`."""
    store = Store() if save else None
    httpd = make_server(host, port, checkpoints, store)
    if store is not None:
        print(f'wonders: saving games to {store.games_dir}')
    url = f'http://{host}:{httpd.server_address[1]}/'
    print(f'wonders: playing at {url} (Ctrl+C to stop)')
    if open_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
