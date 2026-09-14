"""Public API for the wonders game engine.

Convenience re-exports so callers can ``from wonders.wonders import Game, Player``
without reaching into subpackages.
"""

from wonders.cards.base import Card, Color, Cost, Ressource, ScienceSymbol
from wonders.cards.wonder import Wonder, WonderStage
from wonders.engine.actions import Action, ActionKind
from wonders.engine.game import Game, ScoreBreakdown, science_score
from wonders.engine.phase import Phase
from wonders.engine.player import Player

__all__ = [
    'Action',
    'ActionKind',
    'Card',
    'Color',
    'Cost',
    'Game',
    'Phase',
    'Player',
    'Ressource',
    'ScienceSymbol',
    'ScoreBreakdown',
    'Wonder',
    'WonderStage',
    'science_score',
]
