"""Tests for science, military and end-game scoring."""

from collections import Counter

from wonders.cards.base import Card, Color
from wonders.cards.base import ScienceSymbol as S
from wonders.cards.library import wonder_gizah_a, wonder_rhodes_a
from wonders.effects.library import Military, Science, Tally, VictoryPoints
from wonders.engine.game import Game, science_score
from wonders.engine.player import Player


def make_game(*wonders):
    return Game([Player(f'P{i}', w) for i, w in enumerate(wonders)])


def test_science_formula_squares_plus_sets():
    # 2 tablets, 2 gears, 1 compass -> 4 + 4 + 1 + 7*1 = 16
    counts = Counter({S.TABLET: 2, S.GEAR: 2, S.COMPASS: 1})
    assert science_score(counts) == 16


def test_science_single_symbol():
    assert science_score(Counter({S.GEAR: 3})) == 9  # 3^2, no set bonus


def test_science_wild_is_optimised():
    # 1 of each + a wild: best is to make a symbol count 2 -> 1+1+4 + 7 = 13
    counts = Counter({S.TABLET: 1, S.GEAR: 1, S.COMPASS: 1})
    assert science_score(counts, wilds=1) == 13


def test_military_resolution_values_and_defeats():
    g = make_game(wonder_gizah_a(), wonder_gizah_a(), wonder_gizah_a())
    g.players[0].add_card(Card('W', Color.RED, effects=(Military(3),)))
    g.resolve_conflict(age=3)
    # P0 beats both neighbours (+5, +5); P1 and P2 each lose once to P0, tie once.
    assert sum(g.players[0].tokens) == 10
    assert sum(g.players[1].tokens) == -1
    assert sum(g.players[2].tokens) == -1


def _ring(shields):
    g = make_game(*[wonder_gizah_a() for _ in shields])
    for p, n in zip(g.players, shields):
        if n:
            p.add_card(Card(f'Fort {n}', Color.RED, effects=(Military(n),)))
    return g


def test_military_compares_every_pair_of_neighbours_once():
    # Five cities in a ring: P0-P1, P1-P2, P2-P3, P3-P4 and the wrap-around P4-P0 fight; nobody else does.
    g = _ring([5, 1, 3, 0, 4])
    g.resolve_conflict(age=2)  # win +3, defeat -1
    assert [sorted(p.tokens) for p in g.players] == [
        [3, 3],  # beats P1 (1) and P4 (4)
        [-1, -1],  # loses to P0 and P2
        [3, 3],  # beats P1 and P3
        [-1, -1],  # loses to P2 and P4
        [-1, 3],  # beats P3, loses to P0
    ]
    # Every conflict hands out exactly one victory and one defeat: as many of each as there are cities.
    tokens = [t for p in g.players for t in p.tokens]
    assert tokens.count(3) == tokens.count(-1) == 5


def test_a_diplomat_leaves_the_ring_and_its_neighbours_fight_each_other():
    g = _ring([5, 1, 3, 0, 4])
    g.players[1].diplomacy = 1
    g.resolve_conflict(age=1)
    # P1 sits out: P0 and P2 become neighbours (5 vs 3), the other pairs are unchanged.
    assert [sorted(p.tokens) for p in g.players] == [[1, 1], [], [-1, 1], [-1, -1], [-1, 1]]
    g.resolve_conflict(age=1)  # the token was spent: P1 fights again
    assert sorted(g.players[1].tokens) == [-1, -1]


def test_three_cities_with_a_diplomat_fight_once():
    g = _ring([2, 0, 1])
    g.players[1].diplomacy = 1
    g.resolve_conflict(age=3)
    assert g.players[0].tokens == [5] and g.players[2].tokens == [-1]  # one conflict, not two


def test_treasury_is_coins_over_three():
    g = make_game(wonder_gizah_a(), wonder_gizah_a())
    g.players[0].coins = 7
    assert g.score(0).treasury == 2


def test_guild_scores_from_neighbours():
    g = make_game(wonder_gizah_a(), wonder_gizah_a(), wonder_gizah_a())
    # neighbours of P0 each hold 2 brown cards -> Workers Guild = 4 VP.
    for nb in (1, 2):
        g.players[nb].add_card(Card('B1', Color.BROWN))
        g.players[nb].add_card(Card('B2', Color.BROWN))
    g.players[0].add_card(
        Card('Workers Guild', Color.PURPLE, effects=(Tally('colors', 'neighbours', {Color.BROWN}, vp_each=1),))
    )
    assert g.score(0).guilds == 4


def test_wonder_stage_victory_points_counted():
    g = make_game(wonder_rhodes_a(), wonder_rhodes_a())
    p = g.players[0]
    p.add_stage(p.wonder.stages[0])  # +3 VP
    p.add_stage(p.wonder.stages[1])  # military, 0 VP
    assert g.score(0).wonders == 3


def test_full_breakdown_total():
    g = make_game(wonder_gizah_a(), wonder_gizah_a())
    p = g.players[0]
    p.coins = 6  # 2 treasury
    p.add_card(Card('Altar', Color.BLUE, effects=(VictoryPoints(2),)))
    p.add_card(Card('Lab', Color.GREEN, effects=(Science(S.GEAR),)))
    b = g.score(0)
    assert b.civilian == 2
    assert b.science == 1
    assert b.treasury == 2
    assert b.total == 5
