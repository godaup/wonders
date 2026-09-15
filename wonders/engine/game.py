"""The game engine: turn resolution, military, scoring and the draft loop.

Seating is a list of players in clockwise order. For player ``i`` we define the
*left* (clockwise) neighbour as ``i+1`` and the *right* neighbour as ``i-1``.
Military and trading treat both neighbours symmetrically, so the labelling only
matters for the direction hands are passed and for side-specific trade discounts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from wonders.cards.base import Card, Cost, ScienceSymbol
from wonders.effects.base import Context
from wonders.engine.actions import Action, ActionKind
from wonders.engine.phase import Phase
from wonders.engine.player import Player
from wonders.engine.trade import Payment, cheapest_payment

DISCARD_COINS = 3
CONFLICT_VALUE = {1: 1, 2: 3, 3: 5}
DEFEAT_VALUE = -1

# A policy decides one player's action given the game, the player's seat and hand.
Policy = Callable[["Game", int, List[Card]], Action]


def science_score(counts: Counter, wilds: int = 0) -> int:
    """Σ (per-symbol count)² + 7 × (complete sets), maximising over wild symbols."""
    base = [
        counts.get(ScienceSymbol.TABLET, 0),
        counts.get(ScienceSymbol.GEAR, 0),
        counts.get(ScienceSymbol.COMPASS, 0),
    ]
    best = 0
    for t in range(wilds + 1):
        for g in range(wilds - t + 1):
            c = wilds - t - g
            vals = (base[0] + t, base[1] + g, base[2] + c)
            score = sum(v * v for v in vals) + 7 * min(vals)
            best = max(best, score)
    return best


@dataclass
class ScoreBreakdown:
    military: int = 0
    treasury: int = 0
    wonders: int = 0
    civilian: int = 0
    commercial: int = 0
    guilds: int = 0
    science: int = 0

    @property
    def total(self) -> int:
        return (
            self.military + self.treasury + self.wonders + self.civilian + self.commercial + self.guilds + self.science
        )

    def as_dict(self) -> Dict[str, int]:
        d = self.__dict__.copy()
        d['total'] = self.total
        return d


class IllegalAction(Exception):
    pass


class Game:
    def __init__(self, players: List[Player]):
        if len(players) < 2:
            raise ValueError('need at least 2 players')
        self.players = players
        self.n = len(players)
        self.phase = Phase.SETUP
        self.age = 0
        self.discard_pile: List[Card] = []

    # -- neighbours --------------------------------------------------------
    def left_of(self, i: int) -> Player:
        return self.players[(i + 1) % self.n]

    def right_of(self, i: int) -> Player:
        return self.players[(i - 1) % self.n]

    def context(self, i: int) -> Context:
        return Context(self.players[i], self.left_of(i), self.right_of(i))

    # -- payment -----------------------------------------------------------
    def payment_for(self, i: int, cost: Cost) -> Optional[Payment]:
        """Cheapest trade payment for ``cost``'s resources, or ``None`` if impossible.

        Only the resource part is considered here; ``cost.coins`` is added by callers.
        """
        player = self.players[i]
        return cheapest_payment(
            cost.resources,
            player.production_slots(),
            self.left_of(i).production_slots(),
            self.right_of(i).production_slots(),
            player.buy_price,
        )

    def can_build_card(self, i: int, card: Card) -> bool:
        return self._card_payment(i, card) is not None

    def _card_payment(self, i: int, card: Card):
        """Return (coins_to_bank, Payment) if buildable, else ``None``.

        Handles duplicates (illegal) and free chain builds.
        """
        player = self.players[i]
        if player.has(card.name):
            return None  # cannot own two copies of the same structure
        if card.chain_requires and player.has(card.chain_requires):
            return (0, Payment(0, 0, 0))  # free via chain
        pay = self.payment_for(i, card.cost)
        if pay is None:
            return None
        if card.cost.coins + pay.total > player.coins:
            return None
        return (card.cost.coins, pay)

    def _stage_payment(self, i: int):
        player = self.players[i]
        stage = player.next_stage
        if stage is None:
            return None
        pay = self.payment_for(i, stage.cost)
        if pay is None:
            return None
        if stage.cost.coins + pay.total > player.coins:
            return None
        return (stage, stage.cost.coins, pay)

    # -- actions -----------------------------------------------------------
    def apply(self, i: int, action: Action) -> None:
        if action.kind is ActionKind.BUILD:
            self._do_build(i, action.card)
        elif action.kind is ActionKind.WONDER:
            self._do_wonder(i, action.card)
        elif action.kind is ActionKind.DISCARD:
            self._do_discard(i, action.card)
        else:  # pragma: no cover - exhaustive
            raise IllegalAction(f'unknown action {action.kind}')

    def _settle(self, i: int, coins_to_bank: int, pay: Payment) -> None:
        player = self.players[i]
        player.coins -= coins_to_bank + pay.total
        self.left_of(i).coins += pay.left
        self.right_of(i).coins += pay.right

    def _do_build(self, i: int, card: Card) -> None:
        result = self._card_payment(i, card)
        if result is None:
            raise IllegalAction(f'{self.players[i].name} cannot build {card.name}')
        coins_to_bank, pay = result
        self._settle(i, coins_to_bank, pay)
        player = self.players[i]
        player.add_card(card)
        ctx = self.context(i)
        for effect in card.effects:
            player.coins += effect.coins_on_build(ctx)

    def _do_wonder(self, i: int, card: Card) -> None:
        result = self._stage_payment(i)
        if result is None:
            raise IllegalAction(f'{self.players[i].name} has no buildable wonder stage')
        stage, coins_to_bank, pay = result
        self._settle(i, coins_to_bank, pay)
        player = self.players[i]
        player.add_stage(stage)
        self.discard_pile.append(card)  # the card itself is buried, effect ignored
        ctx = self.context(i)
        for effect in stage.effects:
            player.coins += effect.coins_on_build(ctx)

    def _do_discard(self, i: int, card: Card) -> None:
        self.players[i].coins += DISCARD_COINS
        self.discard_pile.append(card)

    # -- legal actions & observation (DL-facing) ---------------------------
    def legal_actions(self, i: int, hand: List[Card]) -> List[Action]:
        actions: List[Action] = []
        stage_ok = self._stage_payment(i) is not None
        for card in hand:
            if self.can_build_card(i, card):
                actions.append(Action.build(card))
            if stage_ok:
                actions.append(Action.wonder(card))
            actions.append(Action.discard(card))
        return actions

    def observe(self, i: int) -> Dict[str, object]:
        """A flat, numeric-friendly snapshot of the game from player ``i``'s seat."""
        p = self.players[i]

        def city(player: Player) -> Dict[str, object]:
            colors = Counter(c.color.value for c in player.cards)
            sci = player.science()
            return {
                'coins': player.coins,
                'shields': player.shields(),
                'wonder_stages': player.wonder_stages_built,
                'tokens': sum(player.tokens),
                'cards_by_color': dict(colors),
                'science': {s.value: n for s, n in sci.items()},
            }

        return {
            'age': self.age,
            'phase': self.phase.value,
            'self': city(p),
            'left': city(self.left_of(i)),
            'right': city(self.right_of(i)),
            'score': self.score(i).as_dict(),
        }

    # -- military ----------------------------------------------------------
    def resolve_conflict(self, age: int) -> None:
        self.phase = Phase.CONFLICT
        value = CONFLICT_VALUE[age]
        shields = [p.shields() for p in self.players]
        for i, player in enumerate(self.players):
            for j in ((i + 1) % self.n, (i - 1) % self.n):
                if shields[i] > shields[j]:
                    player.tokens.append(value)
                elif shields[i] < shields[j]:
                    player.tokens.append(DEFEAT_VALUE)

    # -- scoring -----------------------------------------------------------
    def score(self, i: int) -> ScoreBreakdown:
        from wonders.cards.base import Color

        p = self.players[i]
        ctx = self.context(i)
        b = ScoreBreakdown()
        b.military = sum(p.tokens)
        b.treasury = p.coins // 3
        b.science = science_score(p.science(), p.science_wilds())

        # Wonder-stage victory points.
        for stage in p.stages_built:
            for effect in stage.effects:
                b.wonders += effect.victory_points(ctx)

        # Card victory points, bucketed by colour for a readable breakdown.
        for card in p.cards:
            vp = sum(effect.victory_points(ctx) for effect in card.effects)
            if not vp:
                continue
            if card.color is Color.BLUE:
                b.civilian += vp
            elif card.color is Color.PURPLE:
                b.guilds += vp
            else:
                b.commercial += vp
        return b

    def winner(self) -> int:
        """Seat index of the winner: most points, ties broken by coins."""

        def key(i: int):
            return (self.score(i).total, self.players[i].coins)

        return max(range(self.n), key=key)

    # -- draft loop --------------------------------------------------------
    def play_age(
        self,
        age: int,
        hands: List[List[Card]],
        policy: Policy,
        on_turn: Optional[Callable[[int, int, List[Action]], None]] = None,
    ) -> None:
        """Play one Age from ``hands`` (one hand per player) using ``policy``.

        Six-card ages in the real game leave one card that is discarded; this
        generalises: with hands of size H, players make H-1 plays and the final
        leftover card of each hand is discarded (no coins), per the base rules.

        ``on_turn(age, turn, actions)`` is called each turn with the chosen actions
        (in seat order) before they are applied — used to record a transcript.
        """
        self.age = age
        self.phase = Phase.for_age(age)
        pass_left = self.phase.pass_left
        plays = len(hands[0]) - 1
        for turn in range(plays):
            actions = [policy(self, i, hands[i]) for i in range(self.n)]
            if on_turn is not None:
                on_turn(age, turn, actions)
            for i, action in enumerate(actions):
                hands[i].remove(action.card)
                self.apply(i, action)
            hands = self._pass(hands, pass_left)
        for hand in hands:  # final card of each hand is discarded
            self.discard_pile.extend(hand)
        self.resolve_conflict(age)

    def _pass(self, hands: List[List[Card]], pass_left: bool) -> List[List[Card]]:
        new: List[Optional[List[Card]]] = [None] * self.n
        for i in range(self.n):
            dest = (i + 1) % self.n if pass_left else (i - 1) % self.n
            new[dest] = hands[i]
        return new  # type: ignore[return-value]

    def play(self, deal: Callable[[int], List[List[Card]]], policy: Policy) -> None:
        """Play all three Ages. ``deal(age)`` returns fresh hands for that Age."""
        for age in (1, 2, 3):
            self.play_age(age, deal(age), policy)
        self.phase = Phase.FINISHED
