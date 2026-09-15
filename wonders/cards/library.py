"""Backwards-compatible convenience accessors over the JSON content.

The card and wonder data now lives declaratively in ``data/*.json`` and is parsed
by :mod:`wonders.cards.content`. These helpers just expose a few named handles used
by tests and demos; prefer ``content.load_content`` / ``content.build_age_deck`` for
real deck assembly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

from wonders.cards.base import Card
from wonders.cards.content import Content, load_content
from wonders.cards.wonder import Wonder


@lru_cache(maxsize=None)
def _base() -> Content:
    return load_content('base')


def wonder_rhodes_a() -> Wonder:
    return _base().wonder('The Colossus of Rhodes')


def wonder_gizah_a() -> Wonder:
    return _base().wonder('The Pyramids of Giza')


def wonder_ephesos_a() -> Wonder:
    return _base().wonder('The Temple of Artemis in Ephesus')


def base_cards() -> Dict[int, List[Card]]:
    """Base-game cards grouped by Age (1, 2, 3), one entry per distinct card."""
    cards = _base().cards
    return {age: [c for c in cards if c.age == age] for age in (1, 2, 3)}
