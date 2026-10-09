"""Achievements: feats a player earns in a finished game, derived from the saved games.

Each :class:`Achievement` is a predicate over one saved game (see
:meth:`~wonders.play.session.HumanSession.saved`) that reads only its ``result`` facts — scores per
category, card colours, discard-pile builds — and the human's seat. Nothing is stored separately:
:func:`achievements` checks every finished game, so a newly added achievement also counts the games
played before it existed. To add one, append to :data:`ACHIEVEMENTS` (and translate its texts).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

Game = Dict[str, Any]

BASE_CATEGORIES = ('military', 'treasury', 'wonders', 'civilian', 'commercial', 'guilds', 'science')


@dataclass(frozen=True)
class Achievement:
    """A feat: ``check(game, seat)`` decides whether ``seat`` earned it in the finished ``game``."""

    id: str
    name: str
    description: str
    icon: str
    check: Callable[[Game, int], bool]


def _won(game: Game, seat: int) -> bool:
    return game['result']['places'][seat] == 1


def _pacifist(game: Game, seat: int) -> bool:
    return _won(game, seat) and game['result']['colors'][seat].get('red', 0) == 0


def _renaissance(game: Game, seat: int) -> bool:
    scores = game['result']['scores'][seat]
    return all(scores.get(category, 0) >= 5 for category in BASE_CATEGORIES)


def _dumpster(game: Game, seat: int) -> bool:
    return game['result']['discard_builds'][seat] >= 3


ACHIEVEMENTS: Sequence[Achievement] = (
    Achievement(
        'pacifist',
        'Make Wonders, Not War',
        'Win a game without building a single military (red) card.',
        '🕊',
        _pacifist,
    ),
    Achievement(
        'renaissance',
        'Jack of All Trades, Master of Seven',
        'Score at least 5 points in each of military, treasury, wonders, civic, commercial, guilds and science '
        'in one game.',
        '🎭',
        _renaissance,
    ),
    Achievement(
        'dumpster',
        'Dumpster Diver',
        'Build three cards from the discard pile in one game.',
        '🗑',
        _dumpster,
    ),
)


def earned(game: Game) -> List[str]:
    """Ids of the achievements the human earned in a finished saved ``game``."""
    if not game.get('finished') or not game.get('result'):
        return []
    return [a.id for a in ACHIEVEMENTS if a.check(game, int(game['human']))]


def achievements(games: Sequence[Game]) -> List[Dict[str, Any]]:
    """Every achievement with the games that earned it (``games`` oldest first)."""
    by_id: Dict[str, List[Game]] = {a.id: [] for a in ACHIEVEMENTS}
    for game in games:
        for a_id in earned(game):
            by_id[a_id].append(game)
    return [
        {
            'id': a.id,
            'name': a.name,
            'description': a.description,
            'icon': a.icon,
            'count': len(by_id[a.id]),
            'first': by_id[a.id][0].get('finished_at') or by_id[a.id][0].get('updated') if by_id[a.id] else None,
            'games': [g['id'] for g in by_id[a.id]],
        }
        for a in ACHIEVEMENTS
    ]


__all__ = ['ACHIEVEMENTS', 'Achievement', 'achievements', 'earned']
