"""The game engine: turn resolution, military, scoring and the draft loop.

Seating is a list of players in clockwise order. For player ``i`` we define the
*left* (clockwise) neighbour as ``i+1`` and the *right* neighbour as ``i-1``.
Military and trading treat both neighbours symmetrically, so the labelling only
matters for the direction hands are passed and for side-specific trade discounts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Generator, List, Optional, Tuple

from wonders.cards.base import Card, Color, Cost, ScienceSymbol
from wonders.effects.base import Context
from wonders.engine.actions import Action, ActionKind, DecisionKind, DecisionPoint
from wonders.engine.phase import Phase
from wonders.engine.player import Player
from wonders.engine.trade import Payment, cheapest_payment

DISCARD_COINS = 3
CONFLICT_VALUE = {1: 1, 2: 3, 3: 5}
DEFEAT_VALUE = -1

# A policy decides one player's action given the game, the player's seat and hand.
Policy = Callable[["Game", int, List[Card]], Action]

# One Age as a coroutine: yields decision points, receives {seat: action} for each.
AgeSteps = Generator[DecisionPoint, Dict[int, Action], None]


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
        self.turn = 0
        self.decision = DecisionKind.PLAY  # what the current decision point asks for
        self.discard_pile: List[Card] = []
        self._pending: List = []  # cross-player build effects queued until end of turn

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
        # Free via chaining: any owned structure that unlocks this one. Using the
        # owners' chain_enables (rather than this card's chain_requires) naturally
        # handles cards unlocked by more than one prerequisite (e.g. Forum).
        if any(card.name in owned.chain_enables for owned in player.cards):
            return (0, Payment(0, 0, 0))
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
        """Apply one action and immediately resolve its cross-player effects.

        For a whole simultaneous turn use ``_dispatch`` per seat then
        ``_flush_pending`` once, so end-of-turn effects (Cities plunder/grant/debt on
        neighbours) do not change a later seat's affordability mid-turn.
        """
        self._dispatch(i, action)
        self._flush_pending()

    def _dispatch(self, i: int, action: Action) -> None:
        if action.kind is ActionKind.BUILD:
            self._do_build(i, action.card)
        elif action.kind is ActionKind.WONDER:
            self._do_wonder(i, action.card)
        elif action.kind is ActionKind.DISCARD:
            self._do_discard(i, action.card)
        elif action.kind is ActionKind.FREE:
            self._do_free(i, action.card)
        else:  # pragma: no cover - exhaustive
            raise IllegalAction(f'unknown action {action.kind}')

    def _flush_pending(self) -> None:
        """Resolve queued cross-player build effects (end of turn: gains, then losses)."""
        pending, self._pending = self._pending, []
        for ctx, effect in pending:
            effect.on_build(ctx)

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
        self.players[i].add_card(card)
        self._activate(i, card.effects)

    def _activate(self, i: int, effects) -> None:
        """Trigger the build-time effects of a newly built card or wonder stage."""
        player = self.players[i]
        ctx = self.context(i)
        for effect in effects:
            player.coins += effect.coins_on_build(ctx)  # own coins: immediate
            player.pending_discard_builds += effect.discard_builds()  # resolved at end of turn
            self._pending.append((ctx, effect))  # cross-player: deferred to end of turn

    def _do_wonder(self, i: int, card: Card) -> None:
        result = self._stage_payment(i)
        if result is None:
            raise IllegalAction(f'{self.players[i].name} has no buildable wonder stage')
        stage, coins_to_bank, pay = result
        self._settle(i, coins_to_bank, pay)
        player = self.players[i]
        player.add_stage(stage)
        self.discard_pile.append(card)  # the card itself is buried, effect ignored
        self._activate(i, stage.effects)

    def _do_discard(self, i: int, card: Card) -> None:
        self.players[i].coins += DISCARD_COINS
        self.discard_pile.append(card)

    def _do_free(self, i: int, card: Card) -> None:
        """Build without paying: from the discard pile (Halicarnassus) or from hand (Olympia)."""
        player = self.players[i]
        if player.has(card.name):
            raise IllegalAction(f'{player.name} already owns {card.name}')
        if self.decision is DecisionKind.DISCARD_BUILD:
            if player.pending_discard_builds <= 0 or card not in self.discard_pile:
                raise IllegalAction(f'{player.name} cannot build {card.name} from the discard pile')
            self.discard_pile.remove(card)
            player.pending_discard_builds -= 1
        else:
            if not self._can_build_free(i, card):
                raise IllegalAction(f'{player.name} has no free build for {card.name}')
            player.free_builds_used += 1
        player.add_card(card)
        self._activate(i, card.effects)

    def _can_build_free(self, i: int, card: Card) -> bool:
        """Olympia's once-per-Age free build: offered only where it saves something."""
        player = self.players[i]
        if player.free_builds_left <= 0 or player.has(card.name) or card.color is Color.WHITE:
            return False  # leaders are recruited, not built
        payment = self._card_payment(i, card)
        return payment is None or payment[0] + payment[1].total > 0

    # -- legal actions & observation (DL-facing) ---------------------------
    def legal_actions(self, i: int, hand: List[Card]) -> List[Action]:
        """Legal actions for seat ``i`` choosing from ``hand`` at the current decision."""
        actions: List[Action] = []
        if self.decision is DecisionKind.DISCARD_BUILD:
            seen = set()
            for card in hand:
                if card.name not in seen and not self.players[i].has(card.name):
                    seen.add(card.name)
                    actions.append(Action.free(card))
            return actions
        stage_ok = self._stage_payment(i) is not None
        for card in hand:
            if self.can_build_card(i, card):
                actions.append(Action.build(card))
            if self._can_build_free(i, card):
                actions.append(Action.free(card))
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
        # Cities diplomacy: a player holding a diplomacy token sits this Age out and
        # spends the token. Remaining ("active") players are adjacent around a ring
        # with the diplomats removed, so their neighbours shift accordingly.
        active = [i for i in range(self.n) if self.players[i].diplomacy <= 0]
        for i in range(self.n):
            if self.players[i].diplomacy > 0:
                self.players[i].diplomacy -= 1
        m = len(active)
        for idx in range(m):
            i = active[idx]
            for j in {active[(idx + 1) % m], active[(idx - 1) % m]} - {i}:
                if shields[i] > shields[j]:
                    self.players[i].tokens.append(value)
                elif shields[i] < shields[j]:
                    self.players[i].tokens.append(DEFEAT_VALUE)

    # -- scoring -----------------------------------------------------------
    def score(self, i: int) -> ScoreBreakdown:
        p = self.players[i]
        ctx = self.context(i)
        b = ScoreBreakdown()
        b.military = sum(p.tokens)
        b.treasury = p.coins // 3 - p.debt  # Cities debt tokens are -1 VP each
        symbols, wilds = p.science(), p.science_wilds()
        b.science = science_score(symbols, wilds)
        if p.guild_copies():
            copied_vp, symbols, wilds = self._copy_guilds(i, symbols, wilds)
            b.guilds += copied_vp
            b.science = science_score(symbols, wilds)

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

    def _copy_guilds(self, i: int, symbols: Counter, wilds: int) -> Tuple[int, Counter, int]:
        """Olympia B: copy the best neighbour guild(s) as if built in this city.

        Each copy greedily takes the guild with the largest total gain — its victory
        points plus any science improvement (Scientists Guild). Returns the copied VP
        and the science symbols / wilds including the copies.
        """
        p = self.players[i]
        ctx = self.context(i)
        candidates = [
            card
            for nb in (self.left_of(i), self.right_of(i))
            for card in nb.cards
            if card.color is Color.PURPLE and not p.has(card.name)
        ]
        vp_total = 0
        for _ in range(p.guild_copies()):
            current = science_score(symbols, wilds)
            best = None
            for card in candidates:
                vp = sum(e.victory_points(ctx) for e in card.effects)
                sym = symbols + Counter(e.science() for e in card.effects if e.science() is not None)
                w = wilds + sum(e.science_wild() for e in card.effects)
                gain = vp + science_score(sym, w) - current
                if best is None or gain > best[0]:
                    best = (gain, vp, card, sym, w)
            if best is None:
                break
            _gain, vp, card, symbols, wilds = best
            candidates.remove(card)
            vp_total += vp
        return vp_total, symbols, wilds

    def winner(self) -> int:
        """Seat index of the winner: most points, ties broken by coins."""

        def key(i: int):
            return (self.score(i).total, self.players[i].coins)

        return max(range(self.n), key=key)

    # -- draft loop --------------------------------------------------------
    def age_steps(self, age: int, hands: List[List[Card]]) -> AgeSteps:
        """Play one Age as a coroutine that yields every decision point.

        Each yielded :class:`DecisionPoint` lists the seats that must choose and their
        options; send back ``{seat: action}`` to continue. This is the single source of
        truth for turn order — :meth:`play_age` (callback policies), record replay and
        the training environment all drive it.

        With hands of size H, players make H-1 simultaneous plays. On the last play a
        seat with ``plays_last_card`` (Babylon B) also plays its leftover card; every
        other leftover is discarded (no coins). After each turn, owed discard-pile
        builds (Halicarnassus) are resolved one seat at a time, so they see every card
        discarded that turn — including the end-of-Age leftovers.
        """
        self.age = age
        self.phase = Phase.for_age(age)
        for player in self.players:
            player.free_builds_used = 0
        pass_left = self.phase.pass_left
        plays = len(hands[0]) - 1
        for turn in range(plays):
            self.turn = turn
            actions = yield self._ask(DecisionKind.PLAY, {i: hands[i] for i in range(self.n)})
            self._resolve(actions, hands)
            if turn == plays - 1:
                last = {i: hands[i] for i in range(self.n) if hands[i] and self.players[i].plays_last_card()}
                if last:
                    self._resolve((yield self._ask(DecisionKind.LAST_CARD, last)), hands)
                for hand in hands:  # remaining leftovers are discarded
                    self.discard_pile.extend(hand)
                    hand.clear()
            yield from self._discard_builds()
            hands = self._pass(hands, pass_left)
        self.decision = DecisionKind.PLAY
        self.resolve_conflict(age)

    def _ask(self, kind: DecisionKind, options: Dict[int, List[Card]]) -> DecisionPoint:
        self.decision = kind
        return DecisionPoint(self.age, self.turn, kind, options)

    def _resolve(self, actions: Dict[int, Action], hands: Optional[List[List[Card]]] = None) -> None:
        """Apply simultaneous actions in seat order, then end-of-turn cross-player effects."""
        for i in sorted(actions):
            if hands is not None:
                hands[i].remove(actions[i].card)
            self._dispatch(i, actions[i])
        self._flush_pending()

    def _discard_builds(self) -> AgeSteps:
        """Let each seat owed a discard-pile build pick its card (Halicarnassus)."""
        for i in range(self.n):
            player = self.players[i]
            while player.pending_discard_builds > 0:
                options = [c for c in self.discard_pile if not player.has(c.name)]
                if not options:
                    player.pending_discard_builds = 0  # nothing to build: the power fizzles
                    break
                self._resolve((yield self._ask(DecisionKind.DISCARD_BUILD, {i: options})))
        self.decision = DecisionKind.PLAY

    def play_age(
        self,
        age: int,
        hands: List[List[Card]],
        policy: Policy,
        on_turn: Optional[Callable[[int, int, List[Action]], None]] = None,
        on_extra: Optional[Callable[[DecisionPoint, Dict[int, Action]], None]] = None,
    ) -> None:
        """Play one Age from ``hands`` (one hand per player) using ``policy``.

        ``on_turn(age, turn, actions)`` is called each regular turn with the chosen
        actions (in seat order) before they are applied — used to record a transcript.
        ``on_extra(point, actions)`` is called likewise for the extra decisions granted
        by wonder powers (Babylon's last card, Halicarnassus' discard build).
        """

        def choose(point: DecisionPoint) -> Dict[int, Action]:
            actions = {i: policy(self, i, options) for i, options in point.options.items()}
            if point.kind is DecisionKind.PLAY:
                if on_turn is not None:
                    on_turn(point.age, point.turn, [actions[i] for i in range(self.n)])
            elif on_extra is not None:
                on_extra(point, actions)
            return actions

        drive(self.age_steps(age, hands), choose)

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


def drive(steps: AgeSteps, choose: Callable[[DecisionPoint], Dict[int, Action]]) -> None:
    """Run an :meth:`Game.age_steps` coroutine to completion, answering with ``choose``."""
    try:
        point = next(steps)
        while True:
            point = steps.send(choose(point))
    except StopIteration:
        pass
