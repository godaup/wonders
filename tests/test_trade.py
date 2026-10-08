"""Tests for the resource-payment / trading solver."""

from collections import Counter

from wonders.cards.base import Ressource as R
from wonders.engine.trade import Payment, cheapest_payment, payment_options


def price_two(side, resource):
    return 2


def fs(*rs):
    return frozenset(rs)


def test_free_when_self_produces_everything():
    pay = cheapest_payment(
        Counter([R.WOOD, R.STONE]), own=[fs(R.WOOD), fs(R.STONE)], left=[], right=[], price=price_two
    )
    assert pay is not None and pay.total == 0


def test_buy_missing_from_neighbour():
    pay = cheapest_payment(Counter([R.WOOD, R.WOOD]), own=[fs(R.WOOD)], left=[fs(R.WOOD)], right=[], price=price_two)
    assert pay is not None
    assert pay.total == 2 and pay.left == 2 and pay.right == 0


def test_prefers_cheaper_neighbour():
    def price(side, resource):
        return 1 if side == 'right' else 2

    pay = cheapest_payment(Counter([R.ORE]), own=[], left=[fs(R.ORE)], right=[fs(R.ORE)], price=price)
    assert pay.total == 1 and pay.right == 1 and pay.left == 0


def test_choice_slot_can_only_cover_one_unit():
    # A single choice slot produces one unit per turn, so a second wood must be bought.
    pay = cheapest_payment(
        Counter([R.WOOD, R.WOOD]), own=[fs(R.WOOD, R.STONE)], left=[fs(R.WOOD)], right=[], price=price_two
    )
    assert pay.total == 2


def test_unaffordable_returns_none():
    pay = cheapest_payment(Counter([R.GLASS]), own=[fs(R.WOOD)], left=[fs(R.STONE)], right=[], price=price_two)
    assert pay is None


def test_minimises_across_split_sources():
    # Need 2 wood + 1 ore; self has one wood; buy the rest as cheaply as possible.
    def price(side, resource):
        return 1 if side == 'left' else 2

    pay = cheapest_payment(
        Counter([R.WOOD, R.WOOD, R.ORE]),
        own=[fs(R.WOOD)],
        left=[fs(R.WOOD), fs(R.ORE)],
        right=[fs(R.WOOD)],
        price=price,
    )
    # cheapest: self wood (0) + left wood (1) + left ore (1) = 2
    assert pay.total == 2 and pay.left == 2


def test_payment_options_list_every_useful_split():
    # Two wood, both neighbours sell it; the left one at 1 coin, the right one at 2.
    def price(side, resource):
        return 1 if side == 'left' else 2

    wood = Counter([R.WOOD, R.WOOD])
    sellers = [fs(R.WOOD), fs(R.WOOD)]
    options = payment_options(wood, own=[], left=sellers, right=sellers, price=price)
    assert options == [Payment(2, 2, 0), Payment(3, 1, 2), Payment(4, 0, 4)]
    assert options[0] == cheapest_payment(wood, own=[], left=sellers, right=sellers, price=price)


def test_payment_options_drop_dominated_splits_and_unaffordable_costs():
    options = payment_options(Counter([R.ORE]), own=[fs(R.ORE)], left=[fs(R.ORE)], right=[], price=price_two)
    assert options == [Payment(0, 0, 0)]  # paying a neighbour for what you produce is never useful
    assert payment_options(Counter([R.GLASS]), own=[], left=[], right=[], price=price_two) == []
