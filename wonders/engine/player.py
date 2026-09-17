"""A player's city: their wonder, tableau of built cards, coins and war tokens.

The player aggregates the game-relevant totals (production, shields, science, trade
prices) from the effects of every built card and wonder stage, so the engine can
query one object rather than walking cards itself.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterator, List

from wonders.cards.base import Card, Ressource
from wonders.cards.wonder import Wonder, WonderStage
from wonders.effects.base import Effect

STARTING_COINS = 3
DEFAULT_TRADE_PRICE = 2


class Player:
    def __init__(self, name: str, wonder: Wonder, coins: int = STARTING_COINS):
        self.name = name
        self.wonder = wonder
        self.coins = coins
        self.cards: List[Card] = []
        self._names: set = set()  # built structure names, for chaining & duplicate checks
        self.stages_built: List[WonderStage] = []
        self.tokens: List[int] = []  # military tokens (+1/+3/+5 / -1)
        self.debt = 0  # Cities: debt tokens, each -1 VP at game end
        self.diplomacy = 0  # Cities: diplomacy tokens, each skips one Age's conflict
        self.leaders_hand: List[Card] = []  # Leaders: drafted leaders not yet recruited

    # -- construction ------------------------------------------------------
    def has(self, name: str) -> bool:
        return name in self._names

    def add_card(self, card: Card) -> None:
        self.cards.append(card)
        self._names.add(card.name)

    def add_stage(self, stage: WonderStage) -> None:
        self.stages_built.append(stage)

    @property
    def wonder_stages_built(self) -> int:
        return len(self.stages_built)

    @property
    def next_stage(self):
        idx = len(self.stages_built)
        if idx < len(self.wonder.stages):
            return self.wonder.stages[idx]
        return None

    # -- effect aggregation ------------------------------------------------
    def iter_effects(self) -> Iterator[Effect]:
        for card in self.cards:
            yield from card.effects
        for stage in self.stages_built:
            yield from stage.effects

    def production_slots(self) -> List[frozenset]:
        """Every resource this city can produce, one choice-set per slot."""
        slots = [self.wonder.starting_production]
        for effect in self.iter_effects():
            slots.extend(effect.production())
        return slots

    def shields(self) -> int:
        return sum(effect.shields() for effect in self.iter_effects())

    def science(self) -> "Counter":
        counts: Counter = Counter()
        for effect in self.iter_effects():
            symbol = effect.science()
            if symbol is not None:
                counts[symbol] += 1
        return counts

    def science_wilds(self) -> int:
        return sum(effect.science_wild() for effect in self.iter_effects())

    def buy_price(self, side: str, resource: Ressource) -> int:
        """Coins to buy one ``resource`` from the neighbour on ``side``."""
        price = DEFAULT_TRADE_PRICE
        for effect in self.iter_effects():
            for rule in effect.trade_rules():
                if rule.applies(side, resource):
                    price = min(price, rule.price)
        return price
