"""Tests for the mechanized Leaders / Cities powers (formerly inert ``Special`` effects)."""

from collections import Counter

import pytest

from wonders.cards.base import Card, Color, Cost
from wonders.cards.base import Resource as R
from wonders.cards.base import ScienceSymbol as S
from wonders.cards.content import load_content
from wonders.cards.library import wonder_ephesos_a, wonder_gizah_a, wonder_rhodes_a
from wonders.effects.library import Production, Science, Special
from wonders.engine.actions import Action
from wonders.engine.game import Game, science_score
from wonders.engine.player import Player

CONTENT = load_content('base', 'leaders', 'cities')


def game(n=3):
    wonders = [wonder_gizah_a, wonder_rhodes_a, wonder_ephesos_a]
    players = [Player(f'P{i}', wonders[i % 3]()) for i in range(n)]
    return Game(players, expansions=('base', 'leaders', 'cities'))  # the powers tested here come from both


def own(g, seat, name):
    """Give a seat a real card from the content (without paying for it)."""
    g.players[seat].add_card(CONTENT.card(name))


def card(name, color=Color.BLUE, cost=Cost(), **kw):
    return Card(name, color, cost, **kw)


def test_no_special_placeholders_remain():
    for c in CONTENT.cards:
        assert not any(isinstance(e, Special) for e in c.effects), c.name
    for w in CONTENT.wonders:
        assert not any(isinstance(e, Special) for st in w.stages for e in st.effects), w.name


# -- scoring ------------------------------------------------------------------


def test_aristotle_raises_the_science_set_value():
    g = game()
    for sym in (S.TABLET, S.GEAR, S.COMPASS):
        g.players[0].add_card(card(sym.value, Color.GREEN, effects=(Science(sym),)))
    assert g.score(0).science == 3 + 7
    own(g, 0, 'Aristotle')
    assert g.score(0).science == 3 + 10


def test_justinian_and_plato_score_colour_sets():
    g = game()
    for i, color in enumerate((Color.BLUE, Color.RED, Color.GREEN, Color.BLUE)):
        g.players[0].add_card(card(f'c{i}', color))
    own(g, 0, 'Justinian')
    assert g.score(0).leaders == 3  # one blue+red+green set, scored in the leaders column
    own(g, 0, 'Plato')
    before = g.score(0).leaders
    for i, color in enumerate((Color.BROWN, Color.GREY, Color.YELLOW, Color.PURPLE)):
        g.players[0].add_card(card(f'd{i}', color))
    assert g.score(0).leaders - before == 7  # one full set of seven colours (Justinian unchanged)


def test_science_copy_takes_a_neighbour_symbol():
    g = game()
    g.players[0].add_card(card('t', Color.GREEN, effects=(Science(S.TABLET),)))
    g.players[0].add_card(card('c', Color.GREEN, effects=(Science(S.COMPASS),)))
    g.players[1].add_card(card('g', Color.GREEN, effects=(Science(S.GEAR),)))
    assert g.score(0).science == 2
    own(g, 0, 'Pigeonhole')
    assert g.score(0).science == 3 + 7  # copied the neighbour's gear -> a complete set


def test_science_score_copies_ignore_empty_options():
    assert science_score(Counter({S.TABLET: 1}), copies=[frozenset()]) == 1


# -- build income ----------------------------------------------------------------


def test_xenophon_vitruvius_hatshepsut_income():
    g = game()
    own(g, 0, 'Xenophon')
    g.players[0].coins = 0
    g.apply(0, Action.build(card('Shop', Color.YELLOW)))
    assert g.players[0].coins == 2
    own(g, 0, 'Vitruvius')
    g.players[0].add_card(card('Baths', chain_enables=('Aqueduct',)))
    g.apply(0, Action.build(card('Aqueduct', cost=Cost.of(R.STONE, R.STONE, R.STONE))))
    assert g.players[0].coins == 4  # chain build pays Vitruvius
    own(g, 0, 'Hatshepsut')
    g.players[1].add_card(card('Mine', Color.BROWN, effects=(Production(R.WOOD),)))
    g.apply(0, Action.build(card('Hut', cost=Cost.of(R.WOOD))))
    assert g.players[0].coins == 4 - 2 + 1  # paid 2 for wood, Hatshepsut refunds 1


def test_hatshepsut_pays_per_neighbour_bought_from():
    g = game()
    own(g, 0, 'Hatshepsut')
    g.players[0].coins = 4
    g.players[1].add_card(card('Mine', Color.BROWN, effects=(Production(R.WOOD),)))
    g.players[2].add_card(card('Pit', Color.BROWN, effects=(Production(R.GLASS),)))
    g.apply(0, Action.build(card('Hall', cost=Cost.of(R.WOOD, R.GLASS))))
    assert g.players[0].coins == 4 - 4 + 2  # bought from both neighbours: 1 coin each


def test_nero_pays_for_raider_tokens_too():
    g = game()
    own(g, 0, 'Nero')
    coins = g.players[0].coins
    g.apply(0, Action.build(card('Raid', Color.BLACK, effects=CONTENT.card('Raider Camp').effects)))
    assert g.players[0].coins == coins + 2 and any(t > 0 for t in g.players[0].tokens)


def test_masks_copy_distinct_neighbour_green_cards():
    g = game()
    own(g, 0, 'Pigeonhole')
    own(g, 0, 'Band of Spies')
    # One gear card in the neighbourhood: only one mask can copy it.
    g.players[1].add_card(card('g', Color.GREEN, effects=(Science(S.GEAR),)))
    assert g.score(0).science == 1
    # Several different choices: two masks over tablet/gear/compass cards complete a set.
    g.players[0].add_card(card('t0', Color.GREEN, effects=(Science(S.TABLET),)))
    g.players[2].add_card(card('c', Color.GREEN, effects=(Science(S.COMPASS),)))
    g.players[2].add_card(card('t', Color.GREEN, effects=(Science(S.TABLET),)))
    assert g.score(0).science == 3 + 7  # own tablet + copied gear + copied compass
    # A gear from a leader (not a green card) cannot be copied.
    h = game()
    own(h, 0, 'Pigeonhole')
    own(h, 1, 'Ptolemy')
    assert h.score(0).science == 0


def test_black_market_ignores_yellow_production():
    g = game()
    own(g, 0, 'Black Market')
    forum = card('Forum', Color.YELLOW, effects=(Production(choice=[R.GLASS, R.LINEN, R.PAPER]),))
    g.players[0].add_card(forum)
    # Glass is still "not produced" (yellow cards don't count), so the market may make it.
    assert frozenset({R.GLASS}) <= g.players[0].private_production_slots()[0]


# -- military ---------------------------------------------------------------------


def test_nero_and_tomyris():
    g = game()
    own(g, 0, 'Nero')
    g.players[0].add_card(card('Wall', Color.RED, effects=(CONTENT.card('Stockade').effects)))
    coins = g.players[0].coins
    g.resolve_conflict(1)
    assert g.players[0].tokens == [1, 1] and g.players[0].coins == coins + 4
    t = game()
    own(t, 1, 'Tomyris')
    t.players[0].add_card(card('Wall', Color.RED, effects=(CONTENT.card('Stockade').effects)))
    t.resolve_conflict(1)
    assert -1 not in t.players[1].tokens  # Tomyris's defeat went to the winner
    assert t.players[0].tokens.count(-1) == 1 and t.players[0].tokens.count(1) == 2


def test_memorial_cashes_in_defeats():
    g = game()
    g.players[0].tokens = [-1, -1, 3]
    coins = g.players[0].coins
    g.apply(0, Action.build(card('Memorial', Color.BLACK, effects=CONTENT.card('Memorial').effects)))
    assert g.players[0].tokens == [3] and g.players[0].coins == coins + 4


# -- costs ------------------------------------------------------------------------


def test_cost_discounts_waive_resources():
    g = game()
    g.players[0].coins = 0
    temple = card('Temple', cost=Cost.of(R.GLASS))
    assert not g.can_build_card(0, temple)
    own(g, 0, 'Hammurabi')  # blue cards cost 1 fewer resource
    assert g.can_build_card(0, temple)
    assert not g.can_build_card(0, card('Fort', Color.RED, cost=Cost.of(R.GLASS)))
    assert not g.can_build_card(0, card('Palace', cost=Cost.of(R.GLASS, R.GLASS)))


def test_imhotep_and_architect_firm_discount_wonder_stages():
    g = game()
    g.players[0].coins = 0
    assert g._stage_payment(0) is None  # Giza A stage 1 needs 2 stone; only 1 produced
    own(g, 0, 'Imhotep')
    assert g._stage_payment(0) is not None
    a = game()
    a.players[0].coins = 0
    own(a, 0, 'Architect Firm')
    a.players[0].add_stage(a.players[0].wonder.stages[0])
    assert a._stage_payment(0) is not None  # any stage needs no resources


def test_ramses_and_maecenas_build_free():
    g = game()
    g.players[0].coins = 0
    guild = CONTENT.card('Workers Guild')
    assert not g.can_build_card(0, guild)
    own(g, 0, 'Ramses')
    assert g._card_payment(0, guild) == (0, g._card_payment(0, guild)[1]) and g.can_build_card(0, guild)
    leader = CONTENT.card('Plato')
    assert not g.can_build_card(0, leader)
    own(g, 0, 'Maecenas')
    assert g.can_build_card(0, leader)


def test_bilkis_buys_from_the_bank():
    g = game()
    g.players[0].coins = 1
    glass = card('Lab', cost=Cost.of(R.GLASS))
    assert not g.can_build_card(0, glass)
    own(g, 0, 'Bilkis')
    left, right = g.players[1].coins, g.players[2].coins
    g.apply(0, Action.build(glass))
    assert g.players[0].coins == 0 and (g.players[1].coins, g.players[2].coins) == (left, right)


def test_black_market_and_secret_warehouse_are_private_production():
    g = game()
    g.players[0].coins = 0
    linen = card('Loom', cost=Cost.of(R.LINEN))
    own(g, 0, 'Black Market')
    assert g.can_build_card(0, linen)
    # The neighbour cannot buy from the Black Market.
    g.players[1].coins = 10
    assert not g.can_build_card(1, card('Sail', cost=Cost.of(R.LINEN)))
    w = game()
    w.players[0].coins = 0
    two_stone = card('Wall', cost=Cost.of(R.STONE, R.STONE))  # Giza produces 1 stone
    assert not w.can_build_card(0, two_stone)
    own(w, 0, 'Secret Warehouse')
    assert w.can_build_card(0, two_stone)


def test_smugglers_cache_cheapens_the_starting_resource():
    g = game()
    assert g.players[0].buy_price('left', R.STONE) == 2
    own(g, 0, 'Smugglers Cache')  # Giza's starting resource is stone
    assert g.players[0].buy_price('left', R.STONE) == 1
    assert g.players[0].buy_price('left', R.WOOD) == 2


# -- plunder ----------------------------------------------------------------------


def test_plunder_per_hits_every_other_player():
    g = game(4)  # seat 2 is not a neighbour of seat 0
    for p in g.players:
        p.coins = 3
    g.players[1].tokens = [1, -1]
    g.players[2].tokens = [3, 3, 5, 5]
    g.apply(0, Action.build(card('Sepulcher', Color.BLACK, effects=CONTENT.card('Sepulcher').effects)))
    # Only *victory* tokens count (Cities rulebook): seat 1 holds one, seat 2 four.
    assert g.players[1].coins == 2 and g.players[2].coins == 0 and g.players[2].debt == 1
    assert g.players[0].coins == 3 and g.players[3].coins == 3


def test_cities_plunder_and_grant_reach_non_neighbours():
    g = game(4)
    plunder = next(c for c in CONTENT.cards if any(type(e).__name__ == 'Plunder' for e in c.effects))
    before = g.players[2].coins
    g.apply(0, Action.build(card('P', Color.BLACK, effects=plunder.effects)))
    assert g.players[2].coins < before


@pytest.mark.parametrize('name', ['Aristotle', 'Bilkis', 'Black Market', 'Sepulcher', 'Pigeonhole'])
def test_converted_cards_keep_a_render_label(name):
    assert all(getattr(e, 'label', '') for e in CONTENT.card(name).effects if type(e).__name__ != 'VictoryPoints')
