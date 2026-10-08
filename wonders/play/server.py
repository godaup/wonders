"""A dependency-free local web server for :class:`~wonders.play.session.HumanSession`.

Serves the single-page UI (``static/index.html``) and a small JSON API:

- ``GET /api/options`` — the game menu (expansions, agents, wonders, seat limits)
- ``POST /api/new`` — start a game from a :class:`~wonders.play.session.GameConfig` dict
- ``GET /api/state`` — the human's current view
- ``POST /api/act`` — ``{"index": i}`` plays the i-th legal action (``"payment": [left, right]``
  optionally picks one of its listed splits of the purchase between the neighbours)
- ``GET /api/record`` — the replayable GameRecord JSON
- ``GET /api/card.svg?name=…`` / ``GET /api/wonder.svg?name=…&side=A`` — renderings

It binds to localhost only and holds one game at a time (one human, all local).
"""

from __future__ import annotations

import json
import threading
import webbrowser
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

from wonders.cards.content import Content, load_content
from wonders.play.session import ConfigError, GameConfig, HumanSession, menu_options
from wonders.ruleset import EXPANSIONS
from wonders.viz.render import card_svg, wonder_svg


@lru_cache(maxsize=1)
def _all_content() -> Content:
    return load_content(*(e.id for e in EXPANSIONS.values() if e.implemented))


@lru_cache(maxsize=1)
def _index_html() -> bytes:
    return resources.files('wonders.play').joinpath('static', 'index.html').read_bytes()


@lru_cache(maxsize=512)
def _card_svg(name: str) -> Optional[str]:
    card = next((c for c in _all_content().cards if c.name == name), None)
    return None if card is None else card_svg(card)


@lru_cache(maxsize=64)
def _wonder_svg(name: str, side: str) -> Optional[str]:
    try:
        return wonder_svg(_all_content().wonder(name, side))
    except (KeyError, ValueError):
        return None


class PlayServer:
    """The one game this server hosts, guarded by a lock."""

    def __init__(self, checkpoints: Sequence[str] = ()):
        self.checkpoints = tuple(checkpoints)
        self.session: Optional[HumanSession] = None
        self.lock = threading.Lock()
        self._options: Optional[Dict[str, Any]] = None

    def options(self) -> Dict[str, Any]:
        if self._options is None:
            self._options = menu_options(self.checkpoints)
        return self._options

    def new_game(self, data: Dict[str, Any]) -> Dict[str, Any]:
        config = GameConfig.from_dict(data)
        with self.lock:
            self.session = HumanSession(config, self.checkpoints)
            return self.session.state()

    def state(self) -> Optional[Dict[str, Any]]:
        with self.lock:
            return None if self.session is None else self.session.state()

    def act(self, index: int, payment: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        with self.lock:
            if self.session is None:
                raise ConfigError('no game in progress')
            self.session.act(index, payment)
            return self.session.state()

    def record(self) -> Optional[str]:
        with self.lock:
            return None if self.session is None else self.session.record_json()


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
    '/api/options': lambda play, q: _json(play.options()),
    '/api/state': lambda play, q: _json({'state': play.state()}),
    '/api/record': lambda play, q: _record(play),
    '/api/card.svg': lambda play, q: _svg(_card_svg(q.get('name', ''))),
    '/api/wonder.svg': lambda play, q: _svg(_wonder_svg(q.get('name', ''), q.get('side', 'A'))),
}
POST_ROUTES: Dict[str, Callable[[PlayServer, Dict[str, Any]], Response]] = {
    '/api/new': lambda play, body: _json({'state': play.new_game(body)}),
    '/api/act': lambda play, body: _json({'state': play.act(int(body.get('index', -1)), body.get('payment'))}),
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
        route = POST_ROUTES.get(self.path)
        if route is None:
            self._send(_error(NOT_FOUND, HTTPStatus.NOT_FOUND))
            return
        try:
            length = int(self.headers.get('Content-Length') or 0)
            body = json.loads(self.rfile.read(length) or b'{}')
            if not isinstance(body, dict):
                raise ConfigError('expected a JSON object')
            self._send(route(self.play, body))
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
