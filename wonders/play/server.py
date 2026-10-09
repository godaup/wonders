"""A dependency-free local web server for :class:`~wonders.play.session.HumanSession`.

Serves the single-page UI (``static/index.html``) and a small JSON API:

- ``GET /api/options`` — the game menu (expansions, agents, wonders, seat limits)
- ``POST /api/new`` — start a game from a :class:`~wonders.play.session.GameConfig` dict
- ``GET /api/state`` — the human's current view
- ``POST /api/act`` — ``{"index": i}`` plays the i-th legal action (``"payment": [left, right]``
  optionally picks one of its listed splits of the purchase between the neighbours)
- ``GET /api/record`` — the replayable GameRecord JSON
- ``POST /api/analyse`` / ``GET /api/analysis`` — start / poll the post-game analysis of the human's moves
- ``GET /api/card.svg?name=…`` / ``GET /api/wonder.svg?name=…&side=A[&built=N]`` — renderings
- ``GET /api/locale?lang=de`` — the translation catalog the page uses for its own texts

Every route takes ``?lang=<code>`` (default English): names, card texts, menu entries and the
analysis come back translated (see :mod:`wonders.i18n`).

It binds to localhost only and holds one game at a time (one human, all local).
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
from urllib.parse import parse_qs, urlparse

from wonders.cards.content import Content, load_content
from wonders.i18n import DEFAULT, ENGLISH, Translator, catalog
from wonders.play.session import ConfigError, GameConfig, HumanSession, menu_options
from wonders.ruleset import EXPANSIONS
from wonders.viz.render import card_svg, wonder_svg


@lru_cache(maxsize=1)
def _all_content() -> Content:
    from wonders.extensions import ensure_loaded

    ensure_loaded()  # cards of extension expansions render too
    return load_content(*(e.id for e in EXPANSIONS.values() if e.implemented))


@lru_cache(maxsize=1)
def _index_html() -> bytes:
    return resources.files('wonders.play').joinpath('static', 'index.html').read_bytes()


@lru_cache(maxsize=16)
def _translator(lang: str) -> Translator:
    return Translator(lang)  # an unknown language has an empty catalog: English


def _t(query: Dict[str, Any]) -> Translator:
    return _translator(str(query.get('lang') or DEFAULT)[:16])


@lru_cache(maxsize=1024)
def _card_svg(name: str, lang: str = DEFAULT) -> Optional[str]:
    card = next((c for c in _all_content().cards if c.name == name), None)
    return None if card is None else card_svg(card, _translator(lang))


@lru_cache(maxsize=128)
def _wonder_svg(name: str, side: str, built: str = '0', lang: str = DEFAULT) -> Optional[str]:
    try:
        return wonder_svg(_all_content().wonder(name, side), built=max(0, int(built or 0)), t=_translator(lang))
    except (KeyError, ValueError):
        return None


class PlayServer:
    """The one game this server hosts, guarded by a lock."""

    def __init__(self, checkpoints: Sequence[str] = ()):
        self.checkpoints = tuple(checkpoints)
        self.session: Optional[HumanSession] = None
        self.lock = threading.Lock()
        self._options: Dict[str, Dict[str, Any]] = {}  # per language
        self.analysis: Dict[str, Any] = {'status': 'idle'}  # post-game analysis of the current game

    def options(self, t: Translator = ENGLISH) -> Dict[str, Any]:
        if t.lang not in self._options:
            self._options[t.lang] = menu_options(self.checkpoints, t)
        return self._options[t.lang]

    def new_game(self, data: Dict[str, Any], t: Translator = ENGLISH) -> Dict[str, Any]:
        config = GameConfig.from_dict(data)
        with self.lock:
            self.session = HumanSession(config, self.checkpoints, t=t)
            self.analysis = {'status': 'idle'}
            return self.session.state(t)

    def state(self, t: Translator = ENGLISH) -> Optional[Dict[str, Any]]:
        with self.lock:
            return None if self.session is None else self.session.state(t)

    def act(self, index: int, payment: Optional[Sequence[int]] = None, t: Translator = ENGLISH) -> Dict[str, Any]:
        with self.lock:
            if self.session is None:
                raise ConfigError('no game in progress')
            self.session.act(index, payment)
            return self.session.state(t)

    def record(self) -> Optional[str]:
        with self.lock:
            return None if self.session is None else self.session.record_json()

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
            from wonders.ml.analysis import analyse
        except ImportError:
            result: Dict[str, Any] = {'status': 'unavailable', 'error': 'post-game analysis needs PyTorch'}
        else:
            try:
                processes = max(1, min(6, (os.cpu_count() or 2) - 1))
                report = analyse(record, seat, processes=processes, progress=progress)
                result = {'status': 'done', 'report': report}
            except Exception as exc:  # pragma: no cover - surfaced to the page
                result = {'status': 'error', 'error': str(exc)}
        with self.lock:
            if self.session is game:  # a new game started meanwhile: drop the stale result
                self.analysis = result

    def analysis_state(self, t: Translator = ENGLISH) -> Dict[str, Any]:
        with self.lock:
            state = dict(self.analysis)
        if state.get('status') == 'done' and t.lang != DEFAULT:
            from wonders.ml.analysis import localize

            state['report'] = localize(state['report'], t)
        return state


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


GET_ROUTES: Dict[str, Callable[[PlayServer, Dict[str, str]], Response]] = {
    '/': lambda play, q: (HTTPStatus.OK, _index_html(), 'text/html; charset=utf-8', ()),
    '/api/options': lambda play, q: _json(play.options(_t(q))),
    '/api/locale': lambda play, q: _json(catalog(_t(q).lang) if _t(q).lang != DEFAULT else {}),
    '/api/state': lambda play, q: _json({'state': play.state(_t(q))}),
    '/api/record': lambda play, q: _record(play),
    '/api/analysis': lambda play, q: _json(play.analysis_state(_t(q))),
    '/api/card.svg': lambda play, q: _svg(_card_svg(q.get('name', ''), _t(q).lang)),
    '/api/wonder.svg': lambda play, q: _svg(
        _wonder_svg(q.get('name', ''), q.get('side', 'A'), q.get('built', '0'), _t(q).lang)
    ),
}
# POST routes get the JSON body and the query string (for ``lang``).
POST_ROUTES: Dict[str, Callable[[PlayServer, Dict[str, Any], Dict[str, str]], Response]] = {
    '/api/new': lambda play, body, q: _json({'state': play.new_game(body, _t(q))}),
    '/api/analyse': lambda play, body, q: _json(play.start_analysis()),
    '/api/act': lambda play, body, q: _json(
        {'state': play.act(int(body.get('index', -1)), body.get('payment'), _t(q))}
    ),
}


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

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        url = urlparse(self.path)
        route = GET_ROUTES.get('/' if url.path == '/index.html' else url.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        self._send(route(self.play, query) if route else _error(NOT_FOUND, HTTPStatus.NOT_FOUND))

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        url = urlparse(self.path)
        route = POST_ROUTES.get(url.path)
        if route is None:
            self._send(_error(NOT_FOUND, HTTPStatus.NOT_FOUND))
            return
        try:
            length = int(self.headers.get('Content-Length') or 0)
            body = json.loads(self.rfile.read(length) or b'{}')
            if not isinstance(body, dict):
                raise ConfigError('expected a JSON object')
            self._send(route(self.play, body, {k: v[0] for k, v in parse_qs(url.query).items()}))
        except (ConfigError, ValueError, TypeError) as exc:
            self._send(_error(str(exc)))


def make_server(host: str = '127.0.0.1', port: int = 8765, checkpoints: Sequence[str] = ()) -> ThreadingHTTPServer:
    """Build (but do not start) the server; ``port=0`` picks a free port."""
    handler = type('BoundHandler', (Handler,), {'play': PlayServer(checkpoints)})
    return ThreadingHTTPServer((host, port), handler)


def serve(
    host: str = '127.0.0.1', port: int = 8765, open_browser: bool = True, checkpoints: Sequence[str] = ()
) -> None:
    """Run the UI until interrupted."""
    httpd = make_server(host, port, checkpoints)
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
