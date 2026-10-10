"""A player's city: their wonder, tableau of built cards, coins and war tokens.

The player aggregates the game-relevant totals (production, shields, science, trade
prices) from the effects of every built card and wonder stage, so the engine can
query one object rather than walking cards itself.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterator, List, Optional, Tuple

from wonders.cards.base import Card, Color, Resource
from wonders.cards.wonder import Wonder, WonderStage
from wonders.effects.base import BuildEvent, Effect

STARTING_COINS = 3
DEFAULT_TRADE_PRICE = 2
SCIENCE_SET_VALUE = 7  # points per complete set of the three science symbols
BASIC_COLORS = frozenset(c for c in Color if c.tradeable)  # the official tradeable colours: brown / grey


class Player:
    def __init__(self, name: str, wonder: Wonder, coins: int = STARTING_COINS):
        self.name = name
        self.wonder = wonder
        self.coins = coins
        self.cards: List[Card] = []
        self._names: set = set()  # built structure names, for chaining & duplicate checks
        self.stages_built: List[WonderStage] = []
        # Each board stage as the stages it may be (one, or several when the Manneken Pis copies a Great
        # Wall stage); ``Game`` resolves copied stages once it knows the neighbours (``resolve_slots``).
        self.slots: List[Tuple[WonderStage, ...]] = [(s,) for s in wonder.stages]
        self.built_slots: List[int] = []  # board stage index of each built stage, in build order
        self.built_variants: List[int] = []  # which of the slot's stages each one is (Manneken Pis copies)
        self.stage_cards: List[Card] = []  # the card buried under each built stage, in build order
        self.entombed: List[Card] = []  # Abu Simbel: leaders placed face down (power lost, 2x cost in VP)
        # Aggregates derived from built cards/stages (production, trade prices, ...).
        # Only add_card/add_stage change those, and both clear this cache.
        self._cache: Dict[object, object] = {}
        self.version = 0  # bumped whenever a card or stage is added (cache key for the game)
        self.tokens: List[int] = []  # military tokens (+1/+3/+5 / -1)
        self.debt = 0  # Cities: debt tokens, each -1 VP at game end
        self.diplomacy = 0  # Cities: diplomacy tokens, each skips one Age's conflict
        self.leaders_hand: List[Card] = []  # Leaders: drafted leaders not yet recruited
        self.free_builds_used = 0  # free builds from hand spent this Age (reset each Age)
        self.pending_discard_builds = 0  # free builds from the discard pile still owed
        self.pending_extra_recruits = 0  # Rome B: extra leader recruits still owed
        self.pending_leader_copies = 0  # Courtesans' Guild: neighbour leaders still to copy
        self.pending_entombs = 0  # Abu Simbel: recruited leaders still to entomb
        self.discard_build_priorities: List[int] = []  # one per owed discard build (FAQ order)
        self.copied_leaders: List[Card] = []  # Courtesans' Guild: copied (not owned) leaders

    @classmethod
    def starting(cls, name: str, wonder: Wonder, coins: int = STARTING_COINS) -> 'Player':
        """A player at the start of a game: the treasury plus the board's extra coins (the Manneken Pis)."""
        return cls(name, wonder, coins + wonder.extra_coins)

    # -- construction ------------------------------------------------------
    def has(self, name: str) -> bool:
        return name in self._names

    def add_card(self, card: Card) -> None:
        self.cards.append(card)
        self._names.add(card.name)
        self._cache.clear()
        self.version += 1

    def add_stage(
        self, stage: WonderStage, slot: Optional[int] = None, card: Optional[Card] = None, variant: int = 0
    ) -> None:
        """Build ``stage`` on board stage ``slot`` (default: the next one) with ``card`` buried under it."""
        self.built_slots.append(len(self.stages_built) if slot is None else slot)
        self.built_variants.append(variant)
        self.stages_built.append(stage)
        if card is not None:
            self.stage_cards.append(card)
        self._cache.clear()
        self.version += 1

    def entomb(self, card: Card) -> None:
        """Abu Simbel: place a recruited leader face down — it stops working (and is no leader in the city)."""
        self.cards.remove(card)
        self.entombed.append(card)
        self._cache.clear()
        self.version += 1

    def resolve_slots(self, left: 'Player', right: 'Player') -> None:
        """Turn copied stages (Manneken Pis) into the neighbours' stages they copy."""
        slots: List[Tuple[WonderStage, ...]] = []
        for stage in self.wonder.stages:
            if stage.copy is None:
                slots.append((stage,))
            else:
                side, which = stage.copy
                slots.append((left if side == 'left' else right).wonder.stage_variants(which))
        self.slots = slots

    @property
    def wonder_stages_built(self) -> int:
        return len(self.stages_built)

    def open_slots(self) -> List[int]:
        """Board stages that may be built now: the next one, or every unbuilt one (the Great Wall)."""
        if self.wonder.any_order:
            return [k for k in range(len(self.slots)) if k not in self.built_slots and self.slots[k]]
        k = len(self.built_slots)
        return [k] if k < len(self.slots) and self.slots[k] else []

    def stage_options(self) -> List[Tuple[int, WonderStage]]:
        """Every (board stage, stage) a WONDER action may build now, in board order."""
        return [(k, stage) for k in self.open_slots() for stage in self.slots[k]]

    @property
    def next_stage(self):
        options = self.stage_options()
        return options[0][1] if options else None

    # -- effect aggregation ------------------------------------------------
    def add_copied_leader(self, card: Card) -> None:
        """Courtesans' Guild: gain a neighbour leader's powers without owning the card."""
        self.copied_leaders.append(card)
        self._cache.clear()
        self.version += 1

    def iter_effects(self) -> Iterator[Effect]:
        yield from self.wonder.effects
        for card in self.cards:
            yield from card.effects
        for stage in self.stages_built:
            yield from stage.effects
        for card in self.copied_leaders:
            yield from card.effects

    def production_slots(self) -> List[frozenset]:
        """Every resource this city can produce, one choice-set per slot (do not mutate)."""
        slots = self._cache.get('slots')
        if slots is None:
            slots = [self.wonder.starting_production] if self.wonder.starting_resource is not None else []
            for effect in self.iter_effects():
                slots.extend(effect.production())
            self._cache['slots'] = slots
        return slots  # type: ignore[return-value]

    def tradeable_slots(self) -> List[frozenset]:
        """The production neighbours may buy: the board's resource and brown / grey cards (do not mutate).

        Resources from commercial (yellow) cards such as Forum or Caravansery, wonder stages and
        leaders are for the owner only.
        """
        slots = self._cache.get('trade_slots')
        if slots is None:
            slots = [self.wonder.starting_production] if self.wonder.starting_resource is not None else []
            slots += [s for c in self.cards if c.color.tradeable for e in c.effects for s in e.production()]
            self._cache['trade_slots'] = slots
        return slots  # type: ignore[return-value]

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

    def plays_last_card(self) -> bool:
        return any(effect.plays_last_card() for effect in self.iter_effects())

    @property
    def free_builds_left(self) -> int:
        allowance = self._cache.get('free_builds')
        if allowance is None:
            allowance = sum(effect.free_builds_per_age() for effect in self.iter_effects())
            self._cache['free_builds'] = allowance
        return allowance - self.free_builds_used  # type: ignore[operator]

    def guild_copies(self) -> int:
        return sum(effect.guild_copies() for effect in self.iter_effects())

    # -- Leaders / Cities powers ---------------------------------------------
    def private_production_slots(self) -> List[frozenset]:
        """Production only this city may use (Black Market, Secret Warehouse; do not mutate)."""
        slots = self._cache.get('private_slots')
        if slots is None:
            # Black Market / Secret Warehouse look only at brown/grey cards and the board.
            basic = [self.wonder.starting_production]
            basic += [s for c in self.cards if c.color.tradeable for e in c.effects for s in e.production()]
            produced = frozenset().union(*basic)
            slots = [slot for effect in self.iter_effects() for slot in effect.private_production(produced)]
            self._cache['private_slots'] = slots
        return slots  # type: ignore[return-value]

    def resource_discount(self, target: str) -> int:
        """Resource units waived when paying for ``target`` (a card colour value or 'wonder')."""
        key = ('resource_discount', target)
        if key not in self._cache:
            self._cache[key] = sum(effect.resource_discount(target) for effect in self.iter_effects())
        return self._cache[key]  # type: ignore[return-value]

    def builds_free(self, card: Card) -> bool:
        key = ('builds_free', card.name)  # a card's identity is its name
        if key not in self._cache:
            self._cache[key] = any(effect.builds_free(card) for effect in self.iter_effects())
        return self._cache[key]  # type: ignore[return-value]

    def bank_purchases(self) -> int:
        if 'bank' not in self._cache:
            self._cache['bank'] = sum(effect.bank_purchases() for effect in self.iter_effects())
        return self._cache['bank']  # type: ignore[return-value]

    def science_set_bonus(self) -> int:
        """Extra points per complete science set (Aristotle)."""
        return sum(effect.science_set_bonus() for effect in self.iter_effects())

    def science_set_value(self) -> int:
        """Points per complete science set under the base rules: 7, plus bonuses (``Game`` uses its own rules)."""
        return SCIENCE_SET_VALUE + self.science_set_bonus()

    def science_copies(self) -> int:
        return sum(effect.science_copies() for effect in self.iter_effects())

    def coins_per_victory(self) -> int:
        return sum(effect.coins_per_victory() for effect in self.iter_effects())

    def redirects_defeats(self) -> bool:
        return any(effect.redirects_defeats() for effect in self.iter_effects())

    def coin_discount(self, card: Card) -> int:
        key = ('coin_discount', card.color)  # discounts depend on the card's colour only
        if key not in self._cache:
            self._cache[key] = sum(effect.coin_discount(card) for effect in self.iter_effects())
        return self._cache[key]  # type: ignore[return-value]

    def neighbour_coin_discount(self, card: Card) -> int:
        """Coins this city knocks off its *neighbours'* cost for ``card`` (Rome B)."""
        key = ('neighbour_coin_discount', card.color)
        if key not in self._cache:
            self._cache[key] = sum(effect.neighbour_coin_discount(card) for effect in self.iter_effects())
        return self._cache[key]  # type: ignore[return-value]

    def build_income(self, event: BuildEvent) -> int:
        return sum(effect.build_income(event) for effect in self.iter_effects())

    def buy_price(self, side: str, resource: Resource) -> int:
        """Coins to buy one ``resource`` from the neighbour on ``side``."""
        key = (side, resource)
        cached = self._cache.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        price = DEFAULT_TRADE_PRICE
        for effect in self.iter_effects():
            for rule in effect.trade_rules():
                if rule.applies(side, resource):
                    price = min(price, rule.price)
        price = max(0, price - sum(effect.trade_reduction(self, side, resource) for effect in self.iter_effects()))
        self._cache[key] = price
        return price
