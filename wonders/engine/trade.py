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
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from wonders.cards.base import Ressource


@dataclass(frozen=True)
class Payment:
    """A feasible way to pay a resource cost by trading."""

    total: int  # coins paid to neighbours
    left: int  # of which, to the left neighbour
    right: int  # of which, to the right neighbour


# A production slot is (side, options): side in {'self', 'left', 'right'}.
Slot = Tuple[str, frozenset]

PriceFn = Callable[[str, Ressource], int]


def _slots(side: str, production: Sequence[frozenset]) -> List[Slot]:
    return [(side, options) for options in production]


def cheapest_payment(
    required: Counter,
    own: Sequence[frozenset],
    left: Sequence[frozenset],
    right: Sequence[frozenset],
    price: PriceFn,
) -> Optional[Payment]:
    """Return the minimum-coin :class:`Payment`, or ``None`` if unaffordable.

    ``price(side, resource)`` gives the coins to buy one ``resource`` from the
    neighbour on ``side`` (``'left'`` / ``'right'``).
    """

    units: List[Ressource] = list(required.elements())
    if not units:
        return Payment(0, 0, 0)

    slots: List[Slot] = _slots('self', own) + _slots('left', left) + _slots('right', right)

    # Most-constrained unit first (fewest slots that can supply it) → strong pruning.
    def supply(resource: Ressource) -> int:
        return sum(1 for _side, opts in slots if resource in opts)

    units.sort(key=supply)
    # Early exit if some unit cannot be supplied at all.
    if any(supply(u) == 0 for u in units):
        return None

    n_slots = len(slots)
    used = [False] * n_slots
    best: List[Optional[Tuple[int, int, int]]] = [None]  # (total, left, right)

    def rec(i: int, total: int, l_paid: int, r_paid: int) -> None:
        if best[0] is not None and total >= best[0][0]:
            return
        if i == len(units):
            best[0] = (total, l_paid, r_paid)
            return
        resource = units[i]
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
            else:
                add = price(side, resource)
                nl = add if side == 'left' else 0
                nr = add if side == 'right' else 0
            used[j] = True
            rec(i + 1, total + add, l_paid + nl, r_paid + nr)
            used[j] = False

    rec(0, 0, 0, 0)
    if best[0] is None:
        return None
    total, l_paid, r_paid = best[0]
    return Payment(total, l_paid, r_paid)
