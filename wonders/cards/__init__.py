"""Static game content: colours, resources, costs, cards and wonder boards."""

from wonders.cards.base import MANUFACTURED_GOODS, RAW_MATERIALS, Card, Color, Cost, Ressource, ScienceSymbol
from wonders.cards.content import Content, build_age_deck, deal_hands, load_content
from wonders.cards.wonder import Wonder, WonderStage
