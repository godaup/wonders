"""The game engine: turn resolution, military, scoring and the draft loop.

Seating is a list of players in clockwise order. For player ``i`` we define the
*left* (clockwise) neighbour as ``i+1`` and the *right* neighbour as ``i-1``.
Military and trading treat both neighbours symmetrically, so the labelling only
matters for the direction hands are passed and for side-specific trade discounts.
"""

from __future__ import annotations

import itertools
import random
from collections import Counter
from dataclasses import dataclass, field, fields
from typing import Callable, Dict, Generator, Iterator, List, Optional, Sequence, Tuple

from wonders.cards.base import Card, Color, Cost, ScienceSymbol
from wonders.cards.wonder import TRACKS, Shipyard, WonderStage
from wonders.effects.base import BuildEvent, Context
from wonders.effects.library import Plunder, lose_coins
from wonders.engine.actions import Action, ActionKind, DecisionKind, DecisionPoint
from wonders.engine.phase import Phase
from wonders.engine.player import SCIENCE_SET_VALUE, Player
from wonders.engine.trade import Payment, cheapest_payment, payment_options
from wonders.ruleset import Rules, has_family, rules_of

# Where owed discard builds come from, in the order they resolve (Wonder Pack FAQ); a board's own
# stages use their effect's priority (Halicarnassus 0, the Great Wall 1).
OWN_STAGE, COPIED_STAGE, CARD, COPIED_LEADER = -1, 2, 3, 4
SHIPYARD, ISLAND = 5, 6  # Armada: effects of a fleet space reached, of an island coming into play

# The base game's values; a game's own come from its expansions' meta rules (``Game.rules``).
DISCARD_COINS = 3
CONFLICT_VALUE = {1: 1, 2: 3, 3: 5}
DEFEAT_VALUE = -1
# Armada naval conflicts, per Age: the weakest, the strongest and the second strongest.
NAVAL_DEFEAT = {1: -1, 2: -2, 3: -3}
NAVAL_FIRST = {1: 3, 2: 5, 3: 7}
NAVAL_SECOND = {1: 1, 2: 3, 3: 5}
EXPLORE_DRAW = 4  # islands a lone explorer draws

# A policy decides one player's action given the game, the player's seat and hand.
Policy = Callable[["Game", int, List[Card]], Action]

# One Age as a coroutine: yields decision points, receives {seat: action} for each.
AgeSteps = Generator[DecisionPoint, Dict[int, Action], None]

_NO_TRADE = Payment(0, 0, 0)


_SYMBOL_ORDER = (ScienceSymbol.TABLET, ScienceSymbol.GEAR, ScienceSymbol.COMPASS)


def science_score(
    counts: Counter,
    wilds: int = 0,
    set_value: int = SCIENCE_SET_VALUE,
    copies: Sequence[Sequence[ScienceSymbol]] = (),
    most: int = 0,
    replace: int = 0,
) -> int:
    """Σ (per-symbol count)² + ``set_value`` × (complete sets), maximised over wilds.

    ``wilds`` may each become any symbol. ``copies`` lists alternative extra symbol
    multisets (e.g. which neighbour green cards the Cities masks copy); the best one is
    used. Each wild/copy choice is maximised independently of the others' order.
    ``replace`` symbols may each be changed into any other (Aganice), and ``most`` extra
    symbols go to the kind held most, the best of tied kinds (Enheduanna), after the rest.
    """
    best = 0
    for extra in copies or ((),):
        counted = [counts.get(symbol, 0) for symbol in _SYMBOL_ORDER]
        for symbol in extra:
            counted[_SYMBOL_ORDER.index(symbol)] += 1
        for base in _replacements(counted, replace):
            spare = wilds + (sum(counted) - sum(base))  # a replaced symbol becomes a wild
            for t in range(spare + 1):
                for g in range(spare - t + 1):
                    vals = [base[0] + t, base[1] + g, base[2] + spare - t - g]
                    if most:
                        top = max(vals)
                        best = max(
                            best, max(_science_value(vals, k, most, set_value) for k in range(3) if vals[k] == top)
                        )
                    else:
                        best = max(best, sum(v * v for v in vals) + set_value * min(vals))
    return best


def _science_value(vals: List[int], k: int, extra: int, set_value: int) -> int:
    vals = list(vals)
    vals[k] += extra
    return sum(v * v for v in vals) + set_value * min(vals)


def _replacements(counted: List[int], replace: int) -> List[List[int]]:
    """Every way to take up to ``replace`` symbols out of ``counted`` (each then becomes a wild)."""
    options = [list(counted)]
    for _ in range(replace):
        options += [[*o[:k], o[k] - 1, *o[k + 1 :]] for o in options for k in range(3) if o[k] > 0]
    return [list(o) for o in dict.fromkeys(tuple(o) for o in options)]


@dataclass
class ScoreBreakdown:
    military: int = 0
    treasury: int = 0
    wonders: int = 0
    civilian: int = 0
    commercial: int = 0
    guilds: int = 0
    science: int = 0
    leaders: int = 0  # white cards (and leaders copied with the Courtesans' Guild)
    cities: int = 0  # black cards (the official scorecard adds these to the leaders row)
    naval: int = 0  # Armada: naval conflict tokens
    islands: int = 0  # Armada: islands
    # Armada is in play: ``as_dict`` lists the naval and island columns (other games keep their old keys).
    armada: bool = field(default=False, repr=False, compare=False)

    @property
    def total(self) -> int:
        return sum(getattr(self, f.name) for f in fields(self) if f.name != 'armada')

    def as_dict(self) -> Dict[str, int]:
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name != 'armada'}
        if not self.armada:
            del d['naval'], d['islands']
        d['total'] = self.total
        return d


class IllegalAction(Exception):
    pass


class Game:
    def __init__(self, players: List[Player], expansions: Sequence[str] = ('base',)):
        if len(players) < 2:
            raise ValueError('need at least 2 players')
        self.players = players
        self.expansions: Tuple[str, ...] = tuple(expansions)
        self.rules: Rules = rules_of(self.expansions)  # meta rules: discard coins, military tokens, ...
        self.n = len(players)
        self.phase = Phase.SETUP
        self.age = 0
        self.turn = 0
        self.last_turn = -1  # the current Age's last regular turn (set by age_steps)
        self.decision = DecisionKind.PLAY  # what the current decision point asks for
        self.discard_pile: List[Card] = []
        self._pending: List = []  # cross-player build effects queued until end of turn
        # Memoized payments. A result depends only on the seat's and its neighbours'
        # built structures (Player.version) and the seat's coins, which form the key,
        # so entries never go stale — even if callers mutate players directly.
        self._payments: Dict[tuple, tuple] = {}
        self.leader_box: List[Card] = []  # Leaders: undealt leaders, drawn from the end (Rome B)
        self.tick = 0  # bumped on every engine-driven state change (see ObservationBuilder)
        self._totals: Optional[Tuple[int, List[Dict[str, int]]]] = None  # (tick, breakdowns) cache
        # Armada (see ``setup_armada``): island decks by level, the fleets as cards, Taxes raised this turn and
        # every island draw ([age, turn, seat, level, names], what a record keeps to replay without the shuffle).
        self.armada = has_family(self.expansions, 'armada')
        self.island_decks: Dict[int, List[Card]] = {}
        self.fleet_cards: Dict[str, Card] = {}
        self.island_log: List[list] = []
        self._island_rng: Optional[random.Random] = None
        self._island_script: Optional[Iterator[list]] = None
        self._taxes: List[Tuple[int, int]] = []
        for i, player in enumerate(players):  # copied stages (Manneken Pis) become the neighbours' stages
            if any(stage.copy is not None for stage in player.wonder.stages):
                player.resolve_slots(self.left_of(i), self.right_of(i))

    # -- neighbours --------------------------------------------------------
    def left_of(self, i: int) -> Player:
        return self.players[(i + 1) % self.n]

    def right_of(self, i: int) -> Player:
        return self.players[(i - 1) % self.n]

    def setup_armada(
        self,
        shipyards: Sequence[Shipyard],
        islands: Sequence[Card],
        fleets: Sequence[Card],
        rng: Optional[random.Random] = None,
        script: Optional[Sequence[list]] = None,
    ) -> None:
        """Armada: each seat's Shipyard, the island decks (shuffled with ``rng``) and the fleet cards.

        ``script`` replays a record's island draws (``island_log``) instead of drawing from shuffled decks.
        """
        for player, shipyard in zip(self.players, shipyards):
            player.shipyard = shipyard
        decks: Dict[int, List[Card]] = {}
        for card in islands:
            decks.setdefault(card.level or 1, []).append(card)
        if rng is not None:
            for level in sorted(decks):
                rng.shuffle(decks[level])
        self.island_decks = decks
        self.fleet_cards = {card.color.value: card for card in fleets}
        self._island_rng = rng
        self._island_script = iter(script) if script is not None else None

    def context(self, i: int) -> Context:
        return Context(
            self.players[i],
            self.left_of(i),
            self.right_of(i),
            tuple(self.players),
            debt_tokens=has_family(self.expansions, 'cities'),
            final=self.phase is Phase.FINISHED,
            age=self.age,
            victory_value=self.rules.victory(self.age) if 1 <= self.age <= 3 else 0,
        )

    # -- payment -----------------------------------------------------------
    def payment_for(self, i: int, cost: Cost, waivers: int = 0) -> Optional[Payment]:
        """Cheapest trade payment for ``cost``'s resources, or ``None`` if impossible.

        Only the resource part is considered here; ``cost.coins`` is added by callers.
        ``waivers`` resource units are discounted away (Leaders cost discounts); the
        seat's private production and bank purchases (Cities / Bilkis) are included.
        """
        player = self.players[i]
        if not cost.resources:
            return _NO_TRADE
        # Many cards share a resource cost: memoize by what the payment depends on (the
        # seat's and neighbours' built structures, the cost and the discount) — not coins.
        key: tuple = (
            'pay',
            i,
            player.version,
            self.left_of(i).version,
            self.right_of(i).version,
            frozenset(cost.resources.items()),
            waivers,
        )
        far = self._far_sources(i)
        if far:
            key += tuple(self.players[(i + d) % self.n].version for d in (2, -2))
        if key in self._payments:
            return self._payments[key]  # type: ignore[return-value]
        private = player.private_production_slots()
        pay = cheapest_payment(
            cost.resources,
            player.production_slots() + private if private else player.production_slots(),
            self.left_of(i).tradeable_slots(),
            self.right_of(i).tradeable_slots(),
            player.buy_price,
            waivers=waivers,
            bank=player.bank_purchases(),
            far=far,
        )
        self._payments[key] = pay  # type: ignore[assignment]
        return pay

    def _far_sources(self, i: int) -> list:
        """Armada's Emporiums: the far players' production seat ``i`` may buy from (none at 3 players)."""
        if not self.armada or self.n < 4:
            return []
        far = self.players[i].far_trade()
        if not far:
            return []
        return [
            (side, self.players[(i + d) % self.n].tradeable_slots(), far.get(d, 0))
            for side, d in (('far_left', 2), ('far_right', -2))
            if far.get(d, 0)
        ]

    def can_build_card(self, i: int, card: Card) -> bool:
        return self._card_payment(i, card) is not None

    def _state_key(self, i: int) -> tuple:
        player = self.players[i]
        return (i, player.version, self.left_of(i).version, self.right_of(i).version, player.coins)

    def _card_payment(self, i: int, card: Card, naval: bool = False):
        """Return (coins_to_bank, Payment) if buildable, else ``None``.

        Handles duplicates (illegal) and free chain builds. ``naval`` adds the Naval Construction of the
        card's fleet (Armada), paid together with the card (alone when the card itself is free).
        """
        key = (id(card), self.age if card.age_cost else 0) + self._state_key(i)
        if naval:
            key += ('naval',)
        hit = self._payments.get(key)
        if hit is not None and hit[0] is card:  # holding the card keeps its id from being reused
            return hit[1]
        result = self._compute_card_payment(i, card, naval)
        self._payments[key] = (card, result)
        return result

    def _compute_card_payment(self, i: int, card: Card, naval: bool = False):
        player = self.players[i]
        if player.has(card.name):
            return None  # cannot own two copies of the same structure
        extra = player.naval_cost(card.color.value) if naval else None
        if naval and extra is None:
            return None
        # Free via chaining: any owned structure that unlocks this one. Using the
        # owners' chain_enables (rather than this card's chain_requires) naturally
        # handles cards unlocked by more than one prerequisite (e.g. Forum).
        if self._chained(i, card) or player.builds_free(card):  # Ramses / Maecenas build free
            if extra is None:
                return (0, Payment(0, 0, 0))
            return self._afford(i, extra.coins, extra, player.resource_discount('naval'))
        waivers = player.resource_discount(card.color.value)
        coins = self._coin_cost(i, card)
        cost = card.cost
        if extra is not None:
            cost = _plus(cost, extra)
            coins += extra.coins
            waivers += player.resource_discount('naval')
        return self._afford(i, coins, cost, waivers)

    def _afford(self, i: int, coins: int, cost: Cost, waivers: int):
        """``(coins, cheapest payment)`` for ``coins`` to the bank plus ``cost``'s resources, if affordable."""
        pay = self.payment_for(i, cost, waivers=waivers)
        if pay is None:
            return None
        if coins + pay.total > self.players[i].coins:
            return None
        return (coins, pay)

    def _free_naval_payment(self, i: int, card: Card):
        """The Naval Construction that goes with a free build of ``card`` from hand (Olympia), if affordable."""
        player = self.players[i]
        extra = player.naval_cost(card.color.value)
        if extra is None:
            return None
        return self._afford(i, extra.coins, extra, player.resource_discount('naval'))

    def naval_track(self, i: int, action: Action) -> Optional[str]:
        """The fleet ``action`` would advance with a Naval Construction (``None``: none may)."""
        if action.kind is ActionKind.DISCARD:
            return 'yellow'
        if action.kind is ActionKind.WONDER:
            return self.players[i].wonder_track()
        color = action.card.color.value
        return color if color in TRACKS else None

    def _coin_cost(self, i: int, card: Card) -> int:
        """The card's coin cost after discounts from the seat and its neighbours (Rome B)."""
        discount = self.players[i].coin_discount(card)
        discount += self.left_of(i).neighbour_coin_discount(card) + self.right_of(i).neighbour_coin_discount(card)
        return max(0, card.coin_cost(self.age) - discount)

    def _chained(self, i: int, card: Card) -> bool:
        """Whether an owned structure unlocks ``card`` for free (chaining)."""
        return any(card.name in owned.chain_enables for owned in self.players[i].cards)

    def stage_options(self, i: int) -> List[Tuple[int, WonderStage]]:
        """The (board stage, stage) choices a WONDER action of seat ``i`` has (``Action.staged`` picks one).

        Usually the next stage only; every unbuilt stage on the Great Wall; every Great Wall stage
        for a Manneken Pis stage that copies it. Affordability is not checked here.
        """
        return self.players[i].stage_options()

    def _stage_payment(self, i: int, option: Optional[int] = None, naval: bool = False):
        """``(slot, stage, coins to the bank, cheapest payment)`` for stage option ``option`` of seat ``i``.

        ``None`` takes the first affordable option; ``None`` is returned when it cannot be paid. ``naval``
        adds the Naval Construction of the wonder's fleet (Armada).
        """
        key: tuple = ('stage', option) + self._state_key(i)
        if naval:
            key += ('naval',)
        hit = self._payments.get(key)
        if hit is None:
            hit = (None, self._compute_stage_payment(i, option, naval))
            self._payments[key] = hit
        return hit[1]

    def _compute_stage_payment(self, i: int, option: Optional[int], naval: bool = False):
        player = self.players[i]
        options = player.stage_options()
        if option is not None:
            if not 0 <= option < len(options):
                return None
            options = [options[option]]
        extra = None
        if naval:
            track = player.wonder_track()
            extra = player.naval_cost(track) if track is not None else None
            if extra is None:
                return None
        for slot, stage in options:
            cost, waivers = stage.cost, player.resource_discount('wonder')
            if extra is not None:
                cost, waivers = _plus(cost, extra), waivers + player.resource_discount('naval')
            pay = self.payment_for(i, cost, waivers=waivers)
            if pay is not None and cost.coins + pay.total <= player.coins:
                return (slot, stage, cost.coins, pay)
        return None

    def payment_options(self, i: int, action: Action) -> List[Payment]:
        """Every affordable way to split ``action``'s trade between the neighbours (cheapest first).

        Only BUILD and WONDER actions trade; chained / free builds and actions without a
        resource purchase have the single option of paying nobody. The player may pick any of
        them via :meth:`Action.paying` (the cheapest is played when none is chosen).
        """
        player = self.players[i]
        naval = action.naval and action.kind is not ActionKind.DISCARD
        track = self.naval_track(i, action) if naval else None
        extra = player.naval_cost(track) if track is not None else None
        if action.kind is ActionKind.BUILD:
            result = self._card_payment(i, action.card, naval)
            if result is None:
                return []
            coins, cheapest = result
            free = self._chained(i, action.card) or player.builds_free(action.card)
            if cheapest.total == 0 or (free and extra is None):
                return [cheapest]
            if free and extra is not None:
                cost, waivers = extra, player.resource_discount('naval')
            else:
                cost, waivers = action.card.cost, player.resource_discount(action.card.color.value)
                if extra is not None:
                    cost, waivers = _plus(cost, extra), waivers + player.resource_discount('naval')
        elif action.kind is ActionKind.WONDER:
            stage_result = self._stage_payment(i, action.stage, naval)
            if stage_result is None:
                return []
            _slot, stage, coins, cheapest = stage_result
            if cheapest.total == 0:
                return [cheapest]
            cost, waivers = stage.cost, player.resource_discount('wonder')
            if extra is not None:
                cost, waivers = _plus(cost, extra), waivers + player.resource_discount('naval')
        elif action.kind is ActionKind.FREE and extra is not None:
            free_result = self._free_naval_payment(i, action.card)
            if free_result is None:
                return []
            coins, cheapest = free_result
            if cheapest.total == 0:
                return [cheapest]
            cost, waivers = extra, player.resource_discount('naval')
        else:
            return [_NO_TRADE]
        private = player.private_production_slots()
        options = payment_options(
            cost.resources,
            player.production_slots() + private,
            self.left_of(i).tradeable_slots(),
            self.right_of(i).tradeable_slots(),
            player.buy_price,
            waivers=waivers,
            bank=player.bank_purchases(),
            far=self._far_sources(i),
        )
        return [p for p in options if coins + p.total <= player.coins]

    def _chosen_payment(self, i: int, action: Action, split: Tuple[int, int]) -> Payment:
        for option in self.payment_options(i, action):
            if (option.left, option.right) == tuple(split):
                return option
        raise IllegalAction(f'{self.players[i].name} cannot pay {split} (left, right) for {action.card.name}')

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
        if action.naval and not self.armada:
            raise IllegalAction('a Naval Construction needs Armada')
        if action.kind is ActionKind.BUILD:
            self._do_build(i, action.card, action.payment, action.naval)
        elif action.kind is ActionKind.WONDER:
            self._do_wonder(i, action.card, action.payment, action.stage, action.naval)
        elif action.kind is ActionKind.DISCARD:
            self._do_discard(i, action.card, action.naval)
        elif action.kind is ActionKind.FREE:
            self._do_free(i, action.card, action.naval, action.payment)
        else:  # pragma: no cover - exhaustive
            raise IllegalAction(f'unknown action {action.kind}')

    def _flush_pending(self) -> None:
        """Resolve queued cross-player build effects (end of turn: gains, then losses)."""
        pending, self._pending = self._pending, []
        for ctx, effect in pending:
            effect.on_build(ctx)
        if self._taxes:
            self._collect_taxes()
        self._reserve_bonuses()
        self.tick += 1

    def _collect_taxes(self) -> None:
        """Armada: only the highest Tax raised this turn applies, to everyone but the players who raised it."""
        top = max(amount for _seat, amount in self._taxes)
        raisers = {seat for seat, amount in self._taxes if amount == top}
        self._taxes = []
        debt = has_family(self.expansions, 'cities')
        for j, player in enumerate(self.players):
            if j not in raisers and not player.coin_loss_immune():
                lose_coins(player, top - player.commercial_level(), debt)

    def _reserve_bonuses(self) -> None:
        """Berenice: once per turn, a seat that gained coins from the reserve takes its bonus."""
        for player in self.players:
            if player.reserve_gained:
                player.coins += player.reserve_bonus()
                player.reserve_gained = False

    def _settle(self, i: int, coins_to_bank: int, pay: Payment) -> None:
        player = self.players[i]
        player.coins -= coins_to_bank + pay.total
        self.left_of(i).coins += pay.left
        self.right_of(i).coins += pay.right
        if pay.far_left or pay.far_right:  # Armada's Emporiums
            self.players[(i + 2) % self.n].coins += pay.far_left
            self.players[(i - 2) % self.n].coins += pay.far_right

    def _do_build(self, i: int, card: Card, split: Optional[Tuple[int, int]] = None, naval: bool = False) -> None:
        if naval and card.color.value not in TRACKS:
            raise IllegalAction(f'{card.name} has no fleet to advance')
        result = self._card_payment(i, card, naval)
        if result is None:
            raise IllegalAction(f'{self.players[i].name} cannot build {card.name}')
        coins_to_bank, pay = result
        if split is not None:
            pay = self._chosen_payment(i, Action(ActionKind.BUILD, card, naval=naval), split)
        via_chain = self._chained(i, card)
        self._settle(i, coins_to_bank, pay)
        self.players[i].add_card(card)
        self._activate(i, card.effects, CARD)
        self._income(i, BuildEvent(card.color.value, via_chain, pay.traded, _sides(pay)))
        if naval:  # the card's effects first, then the Naval Construction's
            self._advance_fleet(i, card.color.value)

    def _income(self, i: int, event: BuildEvent) -> None:
        """Coins the owner's effects pay out for a build (Xenophon, Vitruvius, Hatshepsut)."""
        self.players[i].gain(self.players[i].build_income(event))

    def _activate(self, i: int, effects, source: int = OWN_STAGE) -> None:
        """Trigger the build-time effects of a newly built card or wonder stage.

        ``source`` places owed discard builds in the Wonder Pack FAQ's order: a board's own stages
        (Halicarnassus, then the Great Wall by their ``priority``), stages the Manneken Pis copies,
        cards (Solomon), then copied leaders (the Courtesans' Guild).
        """
        player = self.players[i]
        ctx = self.context(i)
        for effect in effects:
            player.gain(effect.coins_on_build(ctx))  # own coins: immediate
            owed = effect.discard_builds()  # resolved at end of turn
            if owed:
                priority = effect.discard_build_priority() if source == OWN_STAGE else source
                player.pending_discard_builds += owed
                player.discard_build_priorities += [priority] * owed
            player.pending_entombs += effect.leader_entombs()  # resolved at end of turn
            player.pending_extra_recruits += effect.extra_recruits()  # resolved at end of turn
            player.pending_leader_copies += effect.leader_copies()  # resolved at end of turn
            for _ in range(effect.leader_draws()):  # Rome B: draw leaders from the box now
                if self.leader_box:
                    player.leaders_hand.append(self.leader_box.pop())
            self._pending.append((ctx, effect))  # cross-player: deferred to end of turn
        if self.armada:
            self._activate_naval(i, effects)

    def _activate_naval(self, i: int, effects) -> None:
        """Armada: Taxes, explorations, Boardings, islands drawn and fleets advanced by newly active effects."""
        player = self.players[i]
        draws: List[int] = []
        moves: List[str] = []
        for effect in effects:
            tax = effect.tax()
            if tax:
                self._taxes.append((i, tax))
            player.pending_explores += effect.explores()  # at the end of the turn
            player.pending_fleet_choices += effect.fleet_choices()  # at the end of the turn
            for offset in effect.boards():
                self.players[(i + offset) % self.n].boarded_by.append(i)
            draws += effect.island_draws()
            moves += effect.fleet_advances()
        for level in draws:
            drawn = self._draw_islands(i, level, 1)
            if drawn:
                self._add_island(i, drawn[0])
        for track in moves:
            self._advance_fleet(i, track)

    def _advance_fleet(self, i: int, track: str) -> None:
        """Move seat ``i``'s ``track`` fleet one space and apply the space reached (nothing at the end)."""
        player = self.players[i]
        if player.shipyard is None or player.naval_cost(track) is None:
            return
        player.advance_fleet(track)
        player.gain(player.fleet_income())  # Wonderous Oasis
        self._activate(i, player.shipyard.space(track, player.fleets[track]), SHIPYARD)

    def _add_island(self, i: int, card: Card) -> None:
        self.players[i].add_island(card)
        self._activate(i, card.effects, ISLAND)

    def _draw_islands(self, seat: int, level: int, count: int) -> List[Card]:
        """Take ``count`` islands of ``level`` for ``seat``: from the shuffled deck, or as a record says."""
        deck = self.island_decks.setdefault(level, [])
        if self._island_script is not None:
            entry = next(self._island_script, None)
            if entry is None or list(entry[2:4]) != [seat, level]:
                raise ValueError(f'the record has no island draw of level {level} for seat {seat} here')
            cards = []
            for name in entry[4]:
                card = next((c for c in deck if c.name == name), None)
                if card is None:
                    raise ValueError(f'recorded island {name!r} is not in the level {level} deck')
                deck.remove(card)
                cards.append(card)
        else:
            cards = [deck.pop() for _ in range(min(count, len(deck)))]
        self.island_log.append([self.age, self.turn, seat, level, [c.name for c in cards]])
        return cards

    def _do_wonder(
        self,
        i: int,
        card: Card,
        split: Optional[Tuple[int, int]] = None,
        option: Optional[int] = None,
        naval: bool = False,
    ) -> None:
        result = self._stage_payment(i, option, naval)
        if result is None:
            raise IllegalAction(f'{self.players[i].name} has no buildable wonder stage')
        slot, stage, coins_to_bank, pay = result
        track = self.players[i].wonder_track() if naval else None
        if split is not None:
            pay = self._chosen_payment(i, Action(ActionKind.WONDER, card, stage=option, naval=naval), split)
        self._settle(i, coins_to_bank, pay)
        player = self.players[i]
        variant = next(k for k, s in enumerate(player.slots[slot]) if s is stage)
        effects = tuple(effect.buried(card) for effect in stage.effects)  # Stonehenge B notes the card
        if effects != stage.effects:
            stage = WonderStage(stage.cost, effects)
        player.add_stage(stage, slot, card, variant)
        # The card is tucked under the board face down (effect ignored): it does not go to the discard
        # pile, so Halicarnassus / Solomon cannot build it later.
        copied = player.wonder.stages[slot].copy is not None
        self._activate(i, stage.effects, COPIED_STAGE if copied else OWN_STAGE)
        self._income(i, BuildEvent(None, traded=pay.traded, neighbours_traded=_sides(pay)))
        if player.stage_plunder():  # Octavia: every other player loses coins, at the end of the turn
            self._pending.append((self.context(i), Plunder(player.stage_plunder())))
        if track is not None:
            self._advance_fleet(i, track)
        elif self.armada and player.shipyard is not None and player.wonder_track() is None:
            player.pending_fleet_choices.append(1)  # Lost Island: a fleet of the owner's choice, for free

    def _do_discard(self, i: int, card: Card, naval: bool = False) -> None:
        if naval:  # Armada: sell to advance the yellow fleet for free instead of taking coins
            if self.players[i].naval_cost('yellow') is None:
                raise IllegalAction(f'{self.players[i].name} cannot advance the yellow fleet')
            self._advance_fleet(i, 'yellow')
        else:
            self.players[i].gain(self.rules.discard_coins)
        if card.color is not Color.WHITE:  # discarded leaders return to the box (Leaders rulebook), not the pile
            self.discard_pile.append(card)

    def _do_free(self, i: int, card: Card, naval: bool = False, split: Optional[Tuple[int, int]] = None) -> None:
        """Build without paying: from the discard pile (Halicarnassus) or from hand (Olympia).

        A free build from hand may come with a paid Naval Construction (Armada); one from the discard pile never.
        """
        player = self.players[i]
        if player.has(card.name):
            raise IllegalAction(f'{player.name} already owns {card.name}')
        naval_pay = None
        if self.decision is DecisionKind.DISCARD_BUILD:
            if naval:
                raise IllegalAction('a card built from the discard pile allows no Naval Construction')
            if player.pending_discard_builds <= 0 or card not in self.discard_pile:
                raise IllegalAction(f'{player.name} cannot build {card.name} from the discard pile')
            self.discard_pile.remove(card)
            player.pending_discard_builds -= 1
        else:
            if not self._can_build_free(i, card):
                raise IllegalAction(f'{player.name} has no free build for {card.name}')
            if naval:
                naval_pay = self._free_naval_payment(i, card) if card.color.value in TRACKS else None
                if naval_pay is None:
                    raise IllegalAction(f'{player.name} cannot pay the Naval Construction for {card.name}')
                if split is not None:
                    chosen = self._chosen_payment(i, Action(ActionKind.FREE, card, naval=True), split)
                    naval_pay = (naval_pay[0], chosen)
            if not self._free_turn(i):  # Olympia night's first / last card costs no per-Age allowance
                player.free_builds_used += 1
        if naval_pay is not None:
            self._settle(i, *naval_pay)
        player.add_card(card)
        self._activate(i, card.effects, CARD)
        self._income(i, BuildEvent(card.color.value))
        if naval_pay is not None:
            self._advance_fleet(i, card.color.value)

    def _can_build_free(self, i: int, card: Card) -> bool:
        """Olympia's once-per-Age free build: offered only where it saves something."""
        player = self.players[i]
        if player.has(card.name) or card.color is Color.WHITE:
            return False  # leaders are recruited, not built
        if player.free_builds_for(card) <= 0 and not self._free_turn(i):
            return False
        payment = self._card_payment(i, card)
        return payment is None or payment[0] + payment[1].total > 0

    def _free_turn(self, i: int) -> bool:
        """Olympia night (2nd edition): the card of the Age's first / last regular turn is free for seat ``i``."""
        turns = self.players[i].free_build_turns()
        if not turns or self.decision is not DecisionKind.PLAY:
            return False
        return ('first' in turns and self.turn == 0) or ('last' in turns and self.turn == self.last_turn)

    # -- legal actions & observation (DL-facing) ---------------------------
    def legal_actions(self, i: int, hand: List[Card]) -> List[Action]:
        """Legal actions for seat ``i`` choosing from ``hand`` at the current decision."""
        actions: List[Action] = []
        if self.decision in _PICKS:  # pick one of the options
            return [Action.build(card) for card in hand]
        if self.decision is DecisionKind.LEADER_BUILD:  # Rome B: only recruiting is allowed
            return [Action.build(card) for card in hand if self.can_build_card(i, card)]
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
        if self.armada:
            return self._with_naval(i, actions)
        return actions

    def _with_naval(self, i: int, actions: List[Action]) -> List[Action]:
        """Armada: each action followed by its Naval Construction variant when that is possible."""
        out: List[Action] = []
        for action in actions:
            out.append(action)
            if self._naval_legal(i, action):
                out.append(action.with_naval())
        return out

    def _naval_legal(self, i: int, action: Action) -> bool:
        player = self.players[i]
        track = self.naval_track(i, action)
        if track is None or player.naval_cost(track) is None:
            return False
        if action.kind is ActionKind.DISCARD:
            return True
        if action.kind is ActionKind.BUILD:
            return self._card_payment(i, action.card, naval=True) is not None
        if action.kind is ActionKind.WONDER:
            return self._stage_payment(i, action.stage, naval=True) is not None
        return self._free_naval_payment(i, action.card) is not None

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
        self.tick += 1
        value = self.rules.victory(age)
        shields = [p.shields() for p in self.players]
        # Cities diplomacy: a player holding a diplomacy token sits this Age out and
        # spends the token. Remaining ("active") players are adjacent around a ring
        # with the diplomats removed, so their neighbours shift accordingly.
        active = [i for i in range(self.n) if self.players[i].diplomacy <= 0]
        for i in range(self.n):
            if self.players[i].diplomacy > 0:
                self.players[i].diplomacy -= 1
        m = len(active)
        fought = set()
        for idx in range(m):
            i = active[idx]
            for j in sorted({active[(idx + 1) % m], active[(idx - 1) % m]} - {i}):
                self._battle(i, j, value, shields)
                fought.add(frozenset((i, j)))
        if self.armada:
            self._boardings(value, shields, fought)
            self._resolve_naval(age)
        self._reserve_bonuses()  # Nero's coins come from the reserve too (Berenice)

    def _battle(self, i: int, j: int, value: int, shields: List[int]) -> None:
        """Seat ``i``'s side of a military conflict with seat ``j``."""
        if shields[i] > shields[j]:
            self.players[i].tokens.append(value)
            self.players[i].gain(self.players[i].coins_per_victory())  # Nero
        elif shields[i] < shields[j]:
            # Tomyris: the loser's defeat token goes to the neighbour who won.
            loser = self.players[j] if self.players[i].redirects_defeats() else self.players[i]
            loser.tokens.append(self.rules.defeat_points)

    def _boardings(self, value: int, shields: List[int], fought: set) -> None:
        """Armada: each Boarding token is one more military conflict, diplomacy or not.

        With ``rules.boarding_once`` (the first edition's FAQ) a pair of players fights at most once per Age.
        """
        for target, player in enumerate(self.players):
            for attacker in player.boarded_by:
                pair = frozenset((attacker, target))
                if attacker == target or (self.rules.boarding_once and pair in fought):
                    continue
                fought.add(pair)
                self._battle(attacker, target, value, shields)
                self._battle(target, attacker, value, shields)
            player.boarded_by = []

    def _resolve_naval(self, age: int) -> None:
        """Armada: the naval conflict of the Age, across the whole table.

        The weakest take a defeat token (all of them on a tie). Of the others, the strongest take the first
        token and the next strongest the second; players tied at a rank take the next rank's token and push
        the ranks below down one (a tie for second takes nothing). The first edition adds a third rank in Age
        III (``rules.naval_third_points``). Nobody takes a token when all are equally strong. A player who
        may sit out (Forgotten Oasis) does so when they would take the defeat token.
        """
        strength = [p.naval_strength() for p in self.players]
        seats = list(range(self.n))
        while True:
            low = min(strength[i] for i in seats)
            out = [i for i in seats if strength[i] == low and self.players[i].naval_opt_out()]
            if not out or len(out) == len(seats):
                break
            seats = [i for i in seats if i not in out]
        self.tick += 1
        if len({strength[i] for i in seats}) <= 1:
            return
        low = min(strength[i] for i in seats)
        for i in seats:
            if strength[i] == low:
                self.players[i].naval_tokens.append(NAVAL_DEFEAT[age])
        rewards = [NAVAL_FIRST[age], NAVAL_SECOND[age]]
        if age == 3 and self.rules.naval_third_points:
            rewards.append(self.rules.naval_third_points)
        rank = 0
        for level in sorted({strength[i] for i in seats if strength[i] > low}, reverse=True):
            group = [i for i in seats if strength[i] == level]
            if len(group) > 1:
                rank += 1  # tied: the next rank's token
            if rank < len(rewards):
                for i in group:
                    self.players[i].naval_tokens.append(rewards[rank])
            rank += 1

    # -- scoring -----------------------------------------------------------
    def score(self, i: int) -> ScoreBreakdown:
        p = self.players[i]
        ctx = self.context(i)
        b = ScoreBreakdown(armada=self.armada)
        cap = self.rules.guild_vp_cap
        b.military = sum(p.tokens)
        b.treasury = p.coins // self.rules.coins_per_point - p.debt  # Cities debt tokens are -1 VP each
        symbols, wilds = p.science(), p.science_wilds()
        b.science = self._science(i, symbols, wilds)
        if p.guild_copies():
            copied_vp, symbols, wilds = self._copy_guilds(i, symbols, wilds)
            b.guilds += copied_vp
            b.science = self._science(i, symbols, wilds)

        # Wonder-stage victory points.
        for stage in p.stages_built:
            for effect in stage.effects:
                b.wonders += effect.victory_points(ctx)

        # Card victory points, bucketed by colour for a readable breakdown (each colour names its column).
        for card in p.cards:
            vp = sum(effect.victory_points(ctx) for effect in card.effects)
            if cap and card.color is Color.PURPLE:  # Armada: a guild scores at most 10
                vp = min(vp, cap)
            if vp:
                column = card.color.score
                setattr(b, column, getattr(b, column) + vp)
        for card in p.copied_leaders:  # Courtesans' Guild copies score as leaders
            b.leaders += sum(effect.victory_points(ctx) for effect in card.effects)
        # Abu Simbel: twice the leader's cost (6 for a leader whose cost is the Age)
        b.wonders += sum(2 * card.coin_cost(3) for card in p.entombed)
        if self.armada:
            b.naval = sum(p.naval_tokens)
            b.islands = sum(effect.victory_points(ctx) for card in p.islands for effect in card.effects)
            b.civilian += sum(effect.victory_points(ctx) for effect in p.fleet_effects())  # the blue fleet
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
            current = self._science(i, symbols, wilds)
            best = None
            for card in candidates:
                vp = sum(e.victory_points(ctx) for e in card.effects)
                if self.rules.guild_vp_cap:
                    vp = min(vp, self.rules.guild_vp_cap)
                sym = symbols + Counter(e.science() for e in card.effects if e.science() is not None)
                w = wilds + sum(e.science_wild() for e in card.effects)
                gain = vp + self._science(i, sym, w) - current
                if best is None or gain > best[0]:
                    best = (gain, vp, card, sym, w)
            if best is None:
                break
            _gain, vp, card, symbols, wilds = best
            candidates.remove(card)
            vp_total += vp
        return vp_total, symbols, wilds

    def _science(self, i: int, symbols: Counter, wilds: int) -> int:
        """Science points for seat ``i``, with its set bonus and neighbour-symbol copies."""
        p = self.players[i]
        options: List[Tuple[ScienceSymbol, ...]] = []
        if p.science_copies():
            # Each mask copies the symbol of a *different* green card in a neighbouring city.
            pool = [
                symbol
                for nb in (self.left_of(i), self.right_of(i))
                for card in nb.cards
                if card.color is Color.GREEN
                for symbol in (e.science() for e in card.effects)
                if symbol is not None
            ]
            pool.sort(key=_SYMBOL_ORDER.index)
            # De-duplicate (keeping order) the possible multisets of copied symbols.
            options = list(dict.fromkeys(itertools.combinations(pool, min(len(pool), p.science_copies()))))
        set_value = self.rules.science_set_points + p.science_set_bonus()
        return science_score(symbols, wilds, set_value, options, most=p.science_most(), replace=p.science_replaces())

    def cached_totals(self) -> List[int]:
        """Every seat's score total, cached per engine state (``tick``).

        Only valid while the game is advanced by the engine (as in the env and
        :class:`~wonders.engine.observation.ObservationBuilder`): mutating players directly
        does not bump ``tick``. Use :meth:`score` otherwise.
        """
        return [b['total'] for b in self.cached_breakdowns()]

    def cached_breakdowns(self) -> List[Dict[str, int]]:
        """Every seat's score breakdown (``ScoreBreakdown.as_dict``), cached per ``tick``.

        Same validity caveat as :meth:`cached_totals`.
        """
        if self._totals is None or self._totals[0] != self.tick:
            self._totals = (self.tick, [self.score(i).as_dict() for i in range(self.n)])
        return self._totals[1]

    def winner(self) -> int:
        """Seat index of the winner: most points, ties broken by coins."""

        def key(i: int):
            return (self.score(i).total, self.players[i].coins)

        return max(range(self.n), key=key)

    # -- draft loop --------------------------------------------------------
    def age_steps(self, age: int, hands: List[List[Card]], first_turn: int = 0) -> AgeSteps:
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

        ``first_turn`` resumes an Age already in progress (``hands`` are then the hands held
        at that turn, and free builds already spent this Age are kept) — used by search,
        which rebuilds positions mid-Age.
        """
        self.age = age
        self.phase = Phase.for_age(age)
        if first_turn == 0:
            for player in self.players:
                player.free_builds_used = 0
        pass_left = self.phase.pass_left
        last_turn = first_turn + len(hands[0]) - 2
        self.last_turn = last_turn
        for turn in range(first_turn, last_turn + 1):
            self.turn = turn
            actions = yield self._ask(DecisionKind.PLAY, {i: hands[i] for i in range(self.n)})
            self._resolve(actions, hands)
            if turn == last_turn:
                last = {i: hands[i] for i in range(self.n) if hands[i] and self.players[i].plays_last_card()}
                if last:
                    self._resolve((yield self._ask(DecisionKind.LAST_CARD, last)), hands)
                for hand in hands:  # remaining leftovers are discarded
                    self.discard_pile.extend(hand)
                    hand.clear()
            yield from self._power_steps()
            hands = self._pass(hands, pass_left)
        self.decision = DecisionKind.PLAY
        self.resolve_conflict(age)

    def draft_steps(self, hands: List[List[Card]]) -> AgeSteps:
        """Leaders draft as a coroutine: keep one leader per round, pass the hands on.

        ``hands`` are the dealt leader hands (one per seat, equal size). Each round every
        seat keeps one card (``Action.build`` of it) and passes the rest to the left
        (``rules.leader_draft_left``; the second edition passes right);
        the kept cards become each player's ``leaders_hand``.
        """
        kept: List[List[Card]] = [[] for _ in range(self.n)]
        hands = [list(h) for h in hands]
        for round_ in range(len(hands[0])):
            self.turn = round_
            actions = yield self._ask(DecisionKind.DRAFT, {i: hands[i] for i in range(self.n)})
            for i in range(self.n):
                hands[i].remove(actions[i].card)
                kept[i].append(actions[i].card)
            if self.rules.leader_draft_left:  # first edition: pass left, seat i receives from i-1
                hands = [hands[(i - 1) % self.n] for i in range(self.n)]
            else:  # second edition: pass right
                hands = [hands[(i + 1) % self.n] for i in range(self.n)]
        for player, cards in zip(self.players, kept):
            player.leaders_hand = cards
        self.decision = DecisionKind.PLAY

    def recruit_steps(self, age: int) -> AgeSteps:
        """Leaders recruitment before ``age``: each seat holding leaders plays one, simultaneously."""
        self.age, self.turn = age, 0
        seats = {i: p.leaders_hand for i, p in enumerate(self.players) if p.leaders_hand}
        if seats:
            actions = yield self._ask(DecisionKind.RECRUIT, seats)
            for i in sorted(actions):
                self.players[i].leaders_hand.remove(actions[i].card)
            self._resolve(actions)
            yield from self._power_steps()  # e.g. Solomon / a Rome B stage built with a leader
        if age == 3 and self.rules.return_last_leaders:
            for player in self.players:  # the unchosen leaders go back to the box, except on Roma's night side
                keeps = any(e.extra_recruits() for stage in player.wonder.stages for e in stage.effects)
                if not keeps and player.leaders_hand:
                    self.leader_box.extend(player.leaders_hand)
                    player.leaders_hand = []
                    self.tick += 1
        self.decision = DecisionKind.PLAY

    def _ask(self, kind: DecisionKind, options: Dict[int, List[Card]]) -> DecisionPoint:
        self.decision = kind
        self.tick += 1
        return DecisionPoint(self.age, self.turn, kind, options)

    def _resolve(self, actions: Dict[int, Action], hands: Optional[List[List[Card]]] = None) -> None:
        """Apply simultaneous actions in seat order, then end-of-turn cross-player effects."""
        for i in sorted(actions):
            if hands is not None:
                hands[i].remove(actions[i].card)
            self._dispatch(i, actions[i])
        self._flush_pending()

    def _power_steps(self) -> AgeSteps:
        """Resolve owed power decisions one seat at a time, until none are left.

        Halicarnassus / Solomon: build a discarded card for free. Courtesans' Guild: copy a
        neighbour's leader. Rome B: recruit an extra leader. A power with no possible
        choice fizzles. Resolving one power can grant another (e.g. copying Solomon), so
        this loops until nothing is owed.
        """
        progress = True
        while progress:
            progress = False
            if self.armada:  # shipyard effects (choices of fleet, explorations) come before discard builds
                progress = (yield from self._naval_steps()) or progress
            # Discard builds across the table in the Wonder Pack FAQ's order, then by seat.
            while any(p.pending_discard_builds > 0 for p in self.players):
                i = min(
                    (min(p.discard_build_priorities, default=0), k)
                    for k, p in enumerate(self.players)
                    if p.pending_discard_builds > 0
                )[1]
                player = self.players[i]
                options = [c for c in self.discard_pile if not player.has(c.name)]
                if not options:  # nothing to build: the power fizzles
                    player.pending_discard_builds = 0
                    player.discard_build_priorities.clear()
                    continue
                player.discard_build_priorities.remove(min(player.discard_build_priorities, default=0))
                self._resolve((yield self._ask(DecisionKind.DISCARD_BUILD, {i: options})))
                progress = True
            for i in range(self.n):
                player = self.players[i]
                while player.pending_entombs > 0:
                    player.pending_entombs -= 1
                    options = [c for c in player.cards if c.color is Color.WHITE]
                    if options:  # no recruited leader: the power fizzles
                        card = (yield self._ask(DecisionKind.ENTOMB_LEADER, {i: options}))[i].card
                        player.entomb(card)
                        self.tick += 1
                        progress = True
                while player.pending_leader_copies > 0:
                    player.pending_leader_copies -= 1
                    options = [
                        c for nb in (self.left_of(i), self.right_of(i)) for c in nb.cards if c.color is Color.WHITE
                    ]
                    if options:
                        card = (yield self._ask(DecisionKind.COPY_LEADER, {i: options}))[i].card
                        player.add_copied_leader(card)
                        self._activate(i, card.effects, COPIED_LEADER)
                        self._flush_pending()
                        progress = True
                while player.pending_extra_recruits > 0:
                    player.pending_extra_recruits -= 1
                    self.decision = DecisionKind.LEADER_BUILD
                    if any(self.can_build_card(i, c) for c in player.leaders_hand):
                        actions = yield self._ask(DecisionKind.LEADER_BUILD, {i: player.leaders_hand})
                        player.leaders_hand.remove(actions[i].card)
                        self._resolve(actions)
                        progress = True
        self.decision = DecisionKind.PLAY

    def _naval_steps(self) -> Generator[DecisionPoint, Dict[int, Action], bool]:
        """Armada: owed free advances of fleets of choice, then explorations; whether anything was resolved."""
        resolved = False
        while True:
            seat = next((k for k, p in enumerate(self.players) if p.pending_fleet_choices), None)
            if seat is not None:
                player = self.players[seat]
                chosen: List[str] = []
                for _ in range(player.pending_fleet_choices.pop(0)):  # k different fleets, one at a time
                    options = [
                        self.fleet_cards[t]
                        for t in TRACKS
                        if t not in chosen and t in self.fleet_cards and player.naval_cost(t) is not None
                    ]
                    if not options:
                        break
                    track = (yield self._ask(DecisionKind.ADVANCE_FLEET, {seat: options}))[seat].card.color.value
                    if track in chosen or player.naval_cost(track) is None:
                        raise IllegalAction(f'{player.name} cannot advance the {track} fleet')
                    chosen.append(track)
                    self._advance_fleet(seat, track)
                    self._flush_pending()
                resolved = True
                continue
            levels = sorted({level for p in self.players for level in p.pending_explores})
            if not levels:
                return resolved
            yield from self._explore(levels[0])
            resolved = True

    def _explore(self, level: int) -> AgeSteps:
        """Every seat exploring ``level`` now: a lone explorer draws 4 islands, several share the whole deck.

        Each keeps one; the others go back and the deck is shuffled.
        """
        seats = [k for k, p in enumerate(self.players) if level in p.pending_explores]
        for k in seats:
            self.players[k].pending_explores.remove(level)
        deck = self.island_decks.setdefault(level, [])
        share = min(EXPLORE_DRAW, len(deck)) if len(seats) == 1 else len(deck) // len(seats)
        offers = {k: self._draw_islands(k, level, share) for k in seats} if share else {}
        offers = {k: cards for k, cards in offers.items() if cards}
        if not offers:
            return
        actions = yield self._ask(DecisionKind.EXPLORE, offers)
        kept = {}
        for k, cards in offers.items():
            card = actions[k].card
            if card not in cards:
                raise IllegalAction(f'{self.players[k].name} was not offered {card.name}')
            rest = list(cards)
            rest.remove(card)
            deck.extend(rest)
            kept[k] = card
        if self._island_rng is not None:
            self._island_rng.shuffle(deck)
        for k in sorted(kept):
            self._add_island(k, kept[k])
        self._flush_pending()

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


_PICKS = (
    DecisionKind.DRAFT,
    DecisionKind.COPY_LEADER,
    DecisionKind.ENTOMB_LEADER,
    DecisionKind.EXPLORE,
    DecisionKind.ADVANCE_FLEET,
)


def _plus(a: Cost, b: Cost) -> Cost:
    """Two costs paid together (a card and its Naval Construction)."""
    return Cost(coins=a.coins + b.coins, resources=a.resources + b.resources)


def _sides(pay: Payment) -> int:
    """How many neighbours a payment bought from (Hatshepsut pays per neighbour)."""
    return int(pay.left > 0) + int(pay.right > 0)


def drive(steps: AgeSteps, choose: Callable[[DecisionPoint], Dict[int, Action]]) -> None:
    """Run an :meth:`Game.age_steps` coroutine to completion, answering with ``choose``."""
    try:
        point = next(steps)
        while True:
            point = steps.send(choose(point))
    except StopIteration:
        pass
