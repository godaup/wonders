"""Local, permanent storage: saved games (finished and unfinished) under one home folder.

Everything lives under ``$WONDERS_HOME`` (default ``~/.wonders``)::

    games/<id>.json     one file per game: menu config + seed, the decision log and the GameRecord

A saved game is the source of truth for everything derived from it — the game list, the player's
Elo rating (:mod:`wonders.play.rating`) and the achievements (:mod:`wonders.play.achievements`) —
so deleting a derived view never loses anything, and a new achievement also counts past games.
Files are plain JSON (``schema`` versioned) and written atomically: a temporary file is replaced
into place, so a crash never leaves half a game behind.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Union

HOME_VARIABLE = 'WONDERS_HOME'
SCHEMA = 1
_ID = re.compile(r'^[0-9A-Za-z_-]{1,64}$')
# Fields of a game file that the game list does not need (they can be large).
HEAVY = ('log', 'record')


def home() -> Path:
    """The storage root: ``$WONDERS_HOME`` or ``~/.wonders``."""
    return Path(os.environ.get(HOME_VARIABLE) or Path.home() / '.wonders')


def now() -> str:
    """The current time as an ISO-8601 UTC timestamp (milliseconds, so games started together still sort)."""
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def new_id() -> str:
    """A fresh, sortable game id: UTC timestamp plus a random suffix."""
    return datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')[:-3] + '-' + secrets.token_hex(2)


class StorageError(ValueError):
    """A game id is malformed or unknown."""


def write_json(path: Path, data: Any) -> None:
    """Write ``data`` as JSON to ``path`` atomically (temporary file, then replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, separators=(',', ':'))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class Store:
    """Saved games under ``root`` (default :func:`home`), safe to share between threads.

    A game is a JSON object with at least ``id``; :meth:`save_game` stamps ``schema`` and
    ``updated``. :meth:`list_games` returns them without the heavy ``log`` / ``record`` fields,
    oldest first.
    """

    def __init__(self, root: Union[str, Path, None] = None):
        self.root = Path(root) if root is not None else home()
        self.games_dir = self.root / 'games'
        self._lock = threading.Lock()

    def _path(self, game_id: str) -> Path:
        if not _ID.match(game_id):
            raise StorageError(f'invalid game id {game_id!r}')
        return self.games_dir / f'{game_id}.json'

    def save_game(self, game: Dict[str, Any]) -> None:
        data = {**game, 'schema': SCHEMA, 'updated': now()}
        data.setdefault('created', data['updated'])
        with self._lock:
            write_json(self._path(str(game['id'])), data)

    def load_game(self, game_id: str) -> Dict[str, Any]:
        path = self._path(game_id)
        try:
            with self._lock:
                return json.loads(path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            raise StorageError(f'no saved game {game_id!r}') from None

    def delete_game(self, game_id: str) -> None:
        path = self._path(game_id)
        with self._lock:
            if not path.exists():
                raise StorageError(f'no saved game {game_id!r}')
            path.unlink()

    def list_games(self) -> List[Dict[str, Any]]:
        """Every saved game (without ``log`` / ``record``), oldest first; unreadable files are skipped."""
        out = []
        with self._lock:
            paths = sorted(self.games_dir.glob('*.json')) if self.games_dir.is_dir() else []
            for path in paths:
                try:
                    data = json.loads(path.read_text(encoding='utf-8'))
                except (OSError, ValueError):
                    continue
                if isinstance(data, dict) and data.get('schema') == SCHEMA:
                    out.append({k: v for k, v in data.items() if k not in HEAVY})
        out.sort(key=lambda g: (str(g.get('created', '')), str(g.get('id', ''))))
        return out


__all__ = ['HOME_VARIABLE', 'Store', 'StorageError', 'home', 'new_id', 'now', 'write_json']
