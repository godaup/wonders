import enum
from typing import List, Optional

from wonders.effects.base import Effect


class Color(enum.Enum):
    BROWN = 'brown'
    RED = 'red'
    BLUE = 'blue'
    YELLOW = 'yellow'
    GREY = 'grey'
    GREEN = 'green'
    PURPLE = 'purple'
    WHITE = 'white'
    BLACK = 'black'


class Ressource(enum.Enum):
    WOOD = 'wood'
    STONE = 'stone'
    ORE = 'ore'
    BRICKS = 'bricks'
    GLASS = 'glass'
    LINEN = 'linen'
    PAPER = 'paper'
    GOLD = 'gold'


class Card:
    def __init__(
        self,
        name: str,
        color: Color,
        costs: List[Ressource],
        chain_requires: Optional[str],
        chain_enables: Optional[List[str]],
        effects: List[Effect],
    ) -> None:
        self.name = name
        self.color = color
        self.costs = costs
        self.chain_requires = chain_requires
        self.chain_enables = chain_enables
        self.effects = effects
