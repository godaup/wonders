"""Resource-payment solver.

Given a resource cost, the player's own production and both neighbours' production,
find the cheapest way to cover the cost. Own production is free; a neighbour's
production is bought at a per-resource price (default 2 coins, less with trade
discounts). Production is reusable turn to turn, but within a single build each
production slot yields at most one unit — matching the physical game, where a
neighbour's single "1 wood" card can sell you only one wood.

This is a small min-cost assignment problem (resource units -> production slots).
Costs are tiny (a handful of units, a dozen-ish slots), so an exhaustive
best-first backtracking search with pruning is both correct and fast.

Two Leaders/Cities extensions: ``waivers`` units of the cost may be skipped outright
(cost discounts; the search waives whichever units are dearest), and ``bank`` extra
slots sell any tradeable resource from the bank at ``bank_price`` (Bilkis). Coins paid
to the bank are part of ``Payment.total`` but go to neither neighbour.

Armada's Emporiums add ``far`` sources: the production of the player two seats away (``'far_left'`` /
``'far_right'``), at most so many units per build at ``far_price``; those coins are ``Payment.far_left`` /
``far_right``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from wonders.cards.base import MANUFACTURED_GOODS, RAW_MATERIALS, Resource

BANK_OPTIONS = frozenset(RAW_MATERIALS | MANUFACTURED_GOODS)


@dataclass(frozen=True)
class Payment:
    """A feasible way to pay a resource cost by trading."""

    total: int  # coins paid for resources (neighbours + bank + far players)
    left: int  # of which, to the left neighbour
    right: int  # of which, to the right neighbour
    far_left: int = 0  # of which, to the player left of the left neighbour (Armada's Western Emporium)
    far_right: int = 0  # of which, to the player right of the right neighbour (Eastern Emporium)

    @property
    def traded(self) -> int:
        """Coins paid to neighbours (excludes bank purchases)."""
        return self.left + self.right


# A production slot is (side, options): side in {'self', 'left', 'right', 'bank', 'far_left', 'far_right'}.
Slot = Tuple[str, frozenset]

PriceFn = Callable[[str, Resource], int]


# Far sources: (side 'far_left' / 'far_right', that player's tradeable production, units per build).
Far = Sequence[Tuple[str, Sequence[frozenset], int]]


def _slots(side: str, production: Sequence[frozenset]) -> List[Slot]:
    return [(side, options) for options in production]


def _far_slots(far: Far) -> Tuple[List[Slot], Dict[str, int]]:
    slots: List[Slot] = []
    limits: Dict[str, int] = {}
    for side, production, count in far:
        if count > 0 and production:
            slots += _slots(side, production)
            limits[side] = limits.get(side, 0) + count
    return slots, limits


def cheapest_payment(
    required: Counter,
    own: Sequence[frozenset],
    left: Sequence[frozenset],
    right: Sequence[frozenset],
    price: PriceFn,
    waivers: int = 0,
    bank: int = 0,
    bank_price: int = 1,
    far: Far = (),
    far_price: int = 1,
) -> Optional[Payment]:
    """Return the minimum-coin :class:`Payment`, or ``None`` if unaffordable.

    ``price(side, resource)`` gives the coins to buy one ``resource`` from the
    neighbour on ``side`` (``'left'`` / ``'right'``). ``waivers`` units need not be
    paid at all; ``bank`` slots each buy one resource from the bank at ``bank_price``.
    """
    units: List[Resource] = list(required.elements())
    if len(units) <= waivers:
        return Payment(0, 0, 0)

    slots: List[Slot] = (
        _slots('self', own) + _slots('left', left) + _slots('right', right) + _slots('bank', [BANK_OPTIONS] * bank)
    )
    if far:
        return _cheapest_far(units, slots, price, waivers, bank_price, far, far_price)

    # Most-constrained unit first (fewest slots that can supply it) → strong pruning.
    def supply(resource: Resource) -> int:
        return sum(1 for _side, opts in slots if resource in opts)

    units.sort(key=supply)
    # Early exit if more units cannot be supplied at all than may be waived.
    if sum(1 for u in units if supply(u) == 0) > waivers:
        return None

    prices: Dict[Tuple[str, Resource], int] = {}

    def price_of(side: str, resource: Resource) -> int:
        key = (side, resource)
        if key not in prices:
            prices[key] = price(side, resource)
        return prices[key]

    n_slots = len(slots)
    used = [False] * n_slots
    best: List[Optional[Tuple[int, int, int]]] = [None]  # (total, left, right)

    def rec(i: int, total: int, l_paid: int, r_paid: int, waived: int) -> None:
        if best[0] is not None and total >= best[0][0]:
            return
        if i == len(units):
            best[0] = (total, l_paid, r_paid)
            return
        resource = units[i]
        if waived < waivers:  # skip paying for this unit (cost discount)
            rec(i + 1, total, l_paid, r_paid, waived + 1)
        tried = set()  # dedupe symmetric slots at this level
        for j in range(n_slots):
            if used[j]:
                continue
            side, opts = slots[j]
            if resource not in opts:
                continue
            sig = (side, opts)
            if sig in tried:
                continue
            tried.add(sig)
            if side == 'self':
                add = 0
                nl, nr = 0, 0
            elif side == 'bank':
                add = bank_price
                nl, nr = 0, 0
            else:
                add = price_of(side, resource)
                nl = add if side == 'left' else 0
                nr = add if side == 'right' else 0
            used[j] = True
            rec(i + 1, total + add, l_paid + nl, r_paid + nr, waived)
            used[j] = False

    rec(0, 0, 0, 0, 0)
    if best[0] is None:
        return None
    total, l_paid, r_paid = best[0]
    return Payment(total, l_paid, r_paid)


def _cheapest_far(
    units: List[Resource],
    slots: List[Slot],
    price: PriceFn,
    waivers: int,
    bank_price: int,
    far: Far,
    far_price: int,
) -> Optional[Payment]:
    """:func:`cheapest_payment` with far sources (kept apart so the common path stays as fast as it was)."""
    options = _enumerate(units, slots, price, waivers, bank_price, far, far_price)
    if not options:
        return None
    return min(options, key=lambda p: (p.total, p.left + p.right, p.left, p.far_left))


def _enumerate(
    units: List[Resource],
    slots: List[Slot],
    price: PriceFn,
    waivers: int,
    bank_price: int,
    far: Far,
    far_price: int,
) -> List[Payment]:
    """The cheapest payment per distinct split (left, right, far left, far right)."""
    extra, limits = _far_slots(far)
    slots = slots + extra
    best: Dict[Tuple[int, int, int, int], int] = {}
    used = [False] * len(slots)
    taken: Dict[str, int] = {side: 0 for side in limits}

    def rec(i: int, total: int, paid: Tuple[int, int, int, int], waived: int) -> None:
        if i == len(units):
            if paid not in best or total < best[paid]:
                best[paid] = total
            return
        resource = units[i]
        if waived < waivers:
            rec(i + 1, total, paid, waived + 1)
        tried = set()
        for j, (side, opts) in enumerate(slots):
            if used[j] or resource not in opts or (side, opts) in tried:
                continue
            if side in limits and taken[side] >= limits[side]:
                continue
            tried.add((side, opts))
            if side == 'self':
                add, slot = 0, -1
            elif side == 'bank':
                add, slot = bank_price, -1
            elif side in limits:
                add, slot = far_price, 2 if side == 'far_left' else 3
            else:
                add, slot = price(side, resource), 0 if side == 'left' else 1
            new = list(paid)
            if slot >= 0:
                new[slot] += add
            used[j] = True
            if side in limits:
                taken[side] += 1
            rec(i + 1, total + add, (new[0], new[1], new[2], new[3]), waived)
            if side in limits:
                taken[side] -= 1
            used[j] = False

    rec(0, 0, (0, 0, 0, 0), 0)
    return [Payment(total, *paid) for paid, total in best.items()]


def payment_options(
    required: Counter,
    own: Sequence[frozenset],
    left: Sequence[frozenset],
    right: Sequence[frozenset],
    price: PriceFn,
    waivers: int = 0,
    bank: int = 0,
    bank_price: int = 1,
    far: Far = (),
    far_price: int = 1,
) -> List[Payment]:
    """Every distinct way to split a purchase between the neighbours (cheapest first).

    The rules let a player buy from either neighbour; :func:`cheapest_payment` only finds the
    cheapest. This lists one :class:`Payment` per distinct ``(left, right)`` split — the
    cheapest total for that split — so a human can choose whom to pay. Same arguments as
    :func:`cheapest_payment`; empty if the cost cannot be covered.
    """
    units: List[Resource] = list(required.elements())
    slots: List[Slot] = (
        _slots('self', own) + _slots('left', left) + _slots('right', right) + _slots('bank', [BANK_OPTIONS] * bank)
    )
    if far:  # Armada's Emporiums: every split with the far players too, cheapest first
        found = _enumerate(units, slots, price, waivers, bank_price, far, far_price)
        return sorted(
            (p for p in found if not any(q != p and _dominates(q, p) for q in found)),
            key=lambda p: (p.total, p.left, p.right, p.far_left, p.far_right),
        )
    best: Dict[Tuple[int, int], int] = {}
    used = [False] * len(slots)

    def rec(i: int, total: int, l_paid: int, r_paid: int, waived: int) -> None:
        if i == len(units):
            key = (l_paid, r_paid)
            if key not in best or total < best[key]:
                best[key] = total
            return
        resource = units[i]
        if waived < waivers:
            rec(i + 1, total, l_paid, r_paid, waived + 1)
        tried = set()
        for j, (side, opts) in enumerate(slots):
            if used[j] or resource not in opts or (side, opts) in tried:
                continue
            tried.add((side, opts))
            add = 0 if side == 'self' else bank_price if side == 'bank' else price(side, resource)
            used[j] = True
            rec(
                i + 1,
                total + add,
                l_paid + (add if side == 'left' else 0),
                r_paid + (add if side == 'right' else 0),
                waived,
            )
            used[j] = False

    rec(0, 0, 0, 0, 0)
    options = [Payment(total, l_paid, r_paid) for (l_paid, r_paid), total in best.items()]

    def dominated(p: Payment) -> bool:  # another split pays nobody more, and less in total or to someone
        return any(q != p and q.left <= p.left and q.right <= p.right and q.total <= p.total for q in options)

    return sorted((p for p in options if not dominated(p)), key=lambda p: (p.total, p.left, p.right))


def _dominates(q: Payment, p: Payment) -> bool:
    """``q`` pays nobody more than ``p`` and no more in total."""
    return (
        q.left <= p.left
        and q.right <= p.right
        and q.far_left <= p.far_left
        and q.far_right <= p.far_right
        and q.total <= p.total
    )
