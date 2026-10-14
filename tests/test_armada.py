"""Armada (``armada2``, second edition; ``armada``, first edition): shipyards, fleets, islands and naval conflicts."""

import random
from collections import Counter

import pytest

from wonders.agents.bench import AGENTS
from wonders.agents.match import play_match
from wonders.cards.base import Card, Color, Cost, Resource
from wonders.cards.content import build_age_deck, load_content
from wonders.cards.decks import deck_size
from wonders.cards.library import wonder_gizah_a, wonder_rhodes_a
from wonders.effects.library import (
    Boarding,
    CoinLossImmunity,
    FarTrade,
    Military,
    NavalOptOut,
    NavalStrength,
    Pirates,
    Production,
    VictoryPoints,
)
from wonders.engine.actions import Action, DecisionKind
from wonders.engine.game import Game, IllegalAction, ScoreBreakdown, drive
from wonders.engine.player import Player
from wonders.engine.record import GameRecord, PlayerSetup
from wonders.env.environment import WondersEnv
from wonders.ruleset import Ruleset, rules_of

SECOND = ('base2', 'armada2')
FIRST = ('base', 'armada')
STONE, BRICKS, WOOD = Resource.STONE, Resource.BRICKS, Resource.WOOD


def _game(n=3, expansions=SECOND, content=None):
    content = content or load_content(*expansions)
    boards = [wonder_gizah_a(), wonder_rhodes_a()] * 4
    game = Game([Player(f'P{i}', boards[i]) for i in range(n)], expansions)
    game.setup_armada([content.shipyards[0]] * n, content.islands, content.fleets, random.Random(0))
    for p in game.players:
        p.coins = 0
    return game


def _give(game, seat, *effects, color=Color.BROWN, name=None):
    game.players[seat].add_card(Card(name or f'Gift {seat} {len(game.players[seat].cards)}', color, effects=effects))


# -- content --------------------------------------------------------------------------


@pytest.mark.parametrize('expansions, boards', [(SECOND, 7), (FIRST, 8)])
def test_content(expansions, boards):
    content = load_content(*expansions)
    armada = [c for c in content.cards if c.deck == 'draw']
    assert len(armada) == 24 and Counter(c.age for c in armada) == {1: 8, 2: 8, 3: 8}
    assert sum(1 for c in armada if c.at_players == (4,)) == 5  # removed at 3 players
    assert len(content.islands) == 27 and Counter(c.level for c in content.islands) == {1: 9, 2: 9, 3: 9}
    assert len(content.shipyards) == boards and [y.official for y in content.shipyards].count(True) == 1
    assert [c.name for c in content.fleets] == ['Red Fleet', 'Yellow Fleet', 'Blue Fleet', 'Green Fleet']
    assert {w.name for w in content.wonders} >= {'Siracusa'}
    assert {y.wonder_track for y in content.shipyards} == {'red', 'yellow', 'blue', 'green'}


def test_rules_of_the_editions():
    assert (rules_of(SECOND).guild_vp_cap, rules_of(SECOND).naval_third_points, rules_of(SECOND).boarding_once) == (
        10,
        0,
        0,
    )
    assert (rules_of(FIRST).naval_third_points, rules_of(FIRST).boarding_once) == (3, 1)
    with pytest.raises(ValueError):
        Ruleset.of('base2', 'armada')  # the first-edition Armada needs the first-edition base game


def test_armada_adds_one_card_per_player_and_age():
    for combo, hand in ((SECOND, 8), (SECOND + ('cities2',), 9), (FIRST, 8)):
        cards = load_content(*combo).cards
        for n in range(3, 8):
            for age in (1, 2, 3):
                size = deck_size(cards, age, n)
                assert (size.hand, size.leftover) == (hand, 0), (combo, n, age)
    deck = build_age_deck(load_content(*SECOND), 1, 3, random.Random(1))
    assert sum(1 for c in deck if c.deck == 'draw') == 3 and all(c.at_players != (4,) for c in deck)


# -- Naval Construction ------------------------------------------------------------------


def test_a_naval_construction_is_paid_with_the_card_and_moves_its_fleet():
    g = _game()
    baths = Card('Baths', Color.BLUE, Cost.of(STONE), (VictoryPoints(3),), age=1)
    _give(g, 0, Production(STONE))
    assert Action.build(baths).with_naval() not in g.legal_actions(0, [baths])  # the blue space costs bricks too
    _give(g, 0, Production(BRICKS))
    legal = g.legal_actions(0, [baths])
    assert Action.build(baths) in legal and Action.build(baths).with_naval() in legal
    g.apply(0, Action.build(baths).with_naval())
    p = g.players[0]
    assert p.fleets['blue'] == 1 and p.has('Baths')
    assert g.score(0).civilian == 3 + 1  # the card, then the blue fleet's first space


def test_a_chained_card_pays_only_its_naval_construction():
    g = _game()
    first = Card('First', Color.BLUE, chain_enables=('Second',), age=1)
    second = Card('Second', Color.BLUE, Cost.of(STONE, STONE, STONE), age=2)
    g.players[0].add_card(first)
    _give(g, 0, Production(BRICKS))  # the blue fleet's first space costs bricks
    assert g._card_payment(0, second, naval=True)[1].total == 0
    g.apply(0, Action.build(second).with_naval())
    assert g.players[0].fleets['blue'] == 1


def test_selling_can_advance_the_yellow_fleet_instead_of_taking_coins():
    g = _game()
    card = Card('X', Color.RED, Cost.of(STONE), age=1)
    g.apply(0, Action.discard(card).with_naval())
    assert g.players[0].fleets['yellow'] == 1 and g.players[0].coins == 0 and card in g.discard_pile
    g.apply(1, Action.discard(card))
    assert g.players[1].coins == 3 and g.players[1].fleets['yellow'] == 0


def test_a_card_built_from_the_discard_pile_has_no_naval_construction():
    g = _game()
    card = Card('X', Color.RED, age=1)
    g.discard_pile.append(card)
    g.players[0].pending_discard_builds = 1
    g.decision = DecisionKind.DISCARD_BUILD
    assert g.legal_actions(0, [card]) == [Action.free(card)]
    with pytest.raises(IllegalAction):
        g.apply(0, Action.free(card).with_naval())


def test_naval_actions_need_armada():
    g = Game([Player(f'P{i}', wonder_gizah_a()) for i in range(3)])
    card = Card('X', Color.RED, age=1)
    assert all(not a.naval for a in g.legal_actions(0, [card]))
    with pytest.raises(IllegalAction):
        g.apply(0, Action.discard(card).with_naval())
    assert 'naval' not in g.score(0).as_dict() and 'naval' in ScoreBreakdown(armada=True).as_dict()


def test_the_emporium_buys_from_the_player_two_seats_away():
    g = _game(n=4)
    _give(g, 0, FarTrade('left2', 1), color=Color.YELLOW)
    _give(g, 2, Production(WOOD))
    g.players[0].coins = 1
    card = Card('Needs wood', Color.BLUE, Cost.of(WOOD), age=1)
    coins, pay = g._card_payment(0, card)
    assert (coins, pay.total, pay.far_left) == (0, 1, 1)
    g.apply(0, Action.build(card))
    assert g.players[0].coins == 0 and g.players[2].coins == 1


# -- coins: Taxes and Pirates ---------------------------------------------------------------


def test_only_the_highest_tax_applies_and_not_to_those_who_raised_it():
    g = _game()
    p0, p1, p2 = g.players
    p0.advance_fleet('yellow')  # space 1: the next one raises a Tax of 1
    for _ in range(3):
        p1.advance_fleet('yellow')  # space 3: the next one raises a Tax of 2
    for p in g.players:
        p.coins = 5
    g._advance_fleet(0, 'yellow')
    g._advance_fleet(1, 'yellow')
    g._flush_pending()
    # P0: +2 coins (space 2), pays the Tax of 2 minus its commercial level 1; P1 raised it; P2 (level 0) pays 2.
    assert [p.coins for p in g.players] == [5 + 2 - 1, 5 + 4, 3]


def test_pirates_take_the_commercial_level_and_foggy_island_protects():
    g = _game()
    for p in g.players:
        p.coins = 5
        p.advance_fleet('yellow')
        p.advance_fleet('yellow')
        p.advance_fleet('yellow')  # commercial level 2
    _give(g, 2, CoinLossImmunity(), color=Color.YELLOW)
    Pirates().on_build(g.context(0))
    assert [p.coins for p in g.players] == [5, 3, 5]


def test_with_cities_an_unpaid_tax_becomes_debt():
    g = _game(expansions=SECOND + ('cities2',))
    g.players[0].advance_fleet('yellow')
    g._advance_fleet(0, 'yellow')
    g._flush_pending()
    assert [p.debt for p in g.players] == [0, 1, 1]


# -- conflicts ---------------------------------------------------------------------------------


def _naval(g, strengths):
    for i, s in enumerate(strengths):
        if s:
            _give(g, i, NavalStrength(s), color=Color.RED)


@pytest.mark.parametrize(
    'expansions, age, strengths, tokens',
    [
        (SECOND, 1, [5, 3, 1], [[3], [1], [-1]]),
        (SECOND, 1, [5, 5, 1], [[1], [1], [-1]]),  # tied for first: the second token each
        (SECOND, 1, [5, 3, 3], [[3], [-1], [-1]]),  # tied for last: a defeat each
        (SECOND, 1, [2, 2, 2], [[], [], []]),  # all equal: nothing
        (SECOND, 2, [5, 4, 4, 1], [[5], [], [], [-2]]),  # tied for second: nothing
        (FIRST, 3, [9, 9, 5, 1], [[5], [5], [3], [-3]]),  # first edition: the third rank in Age III
        (FIRST, 3, [9, 5, 5, 1], [[7], [3], [3], [-3]]),
        (FIRST, 3, [9, 5, 3, 3], [[7], [5], [-3], [-3]]),
        (FIRST, 2, [9, 5, 3, 1], [[5], [3], [], [-2]]),
    ],
)
def test_naval_conflicts(expansions, age, strengths, tokens):
    g = _game(n=len(strengths), expansions=expansions)
    _naval(g, strengths)
    g._resolve_naval(age)
    assert [p.naval_tokens for p in g.players] == tokens


def test_a_player_may_sit_out_rather_than_lose():
    g = _game()
    _naval(g, [5, 3, 1])
    _give(g, 2, NavalOptOut(), color=Color.YELLOW)
    g._resolve_naval(1)
    assert [p.naval_tokens for p in g.players] == [[3], [-1], []]


def test_boarding_adds_a_conflict_that_diplomacy_does_not_stop():
    g = _game(n=4)
    _give(g, 0, Military(2), Boarding('right2'), color=Color.RED)
    g.players[2].diplomacy = 1  # sits out the conflicts with its neighbours, not the Boarding
    g._activate(0, g.players[0].cards[-1].effects)
    assert g.players[2].boarded_by == [0]
    g.resolve_conflict(1)
    assert g.players[0].tokens == [1, 1, 1] and g.players[2].tokens == [-1] and g.players[2].boarded_by == []


@pytest.mark.parametrize('expansions, wins', [(SECOND, 3), (FIRST, 2)])
def test_at_three_players_a_boarding_fights_a_neighbour_again_in_the_second_edition(expansions, wins):
    g = _game(expansions=expansions)
    _give(g, 0, Military(2), Boarding('left2'), color=Color.RED)
    g._activate(0, g.players[0].cards[-1].effects)
    g.resolve_conflict(1)
    assert len(g.players[0].tokens) == wins  # the first edition's FAQ: one conflict per pair and Age


def test_a_guild_scores_at_most_ten():
    for expansions, points in ((SECOND, 10), (('base2',), 14)):
        g = Game([Player(f'P{i}', wonder_gizah_a()) for i in range(3)], expansions)
        g.players[0].add_card(Card('Big Guild', Color.PURPLE, effects=(VictoryPoints(14),), guild=True))
        assert g.score(0).guilds == points


# -- islands and free fleet advances ------------------------------------------------------------


def _pick_first(seen):
    def choose(point):
        seen.append(point)
        return {i: Action.build(cards[0]) for i, cards in point.options.items()}

    return choose


def test_a_lone_explorer_sees_four_islands_and_the_rest_go_back():
    g = _game()
    g.players[0].pending_explores = [1]
    seen = []
    drive(g._power_steps(), _pick_first(seen))
    assert [p.kind for p in seen] == [DecisionKind.EXPLORE] and len(seen[0].options[0]) == 4
    assert len(g.players[0].islands) == 1 and len(g.island_decks[1]) == 8
    assert g.island_log[0][2:4] == [0, 1] and len(g.island_log[0][4]) == 4


def test_explorers_of_one_level_share_its_deck():
    g = _game()
    g.players[0].pending_explores = [2]
    g.players[2].pending_explores = [2]
    seen = []
    drive(g._power_steps(), _pick_first(seen))
    assert {i: len(c) for i, c in seen[0].options.items()} == {0: 4, 2: 4}  # 9 islands: 4 each, 1 not dealt
    assert len(g.players[0].islands) == len(g.players[2].islands) == 1 and len(g.island_decks[2]) == 7


def test_a_free_advance_of_two_different_fleets():
    g = _game()
    g.players[0].pending_fleet_choices = [2]
    seen = []
    drive(g._power_steps(), _pick_first(seen))
    assert [p.kind for p in seen] == [DecisionKind.ADVANCE_FLEET] * 2
    assert [c.name for c in seen[1].options[0]] == ['Yellow Fleet', 'Blue Fleet', 'Green Fleet']
    assert (g.players[0].fleets['red'], g.players[0].fleets['yellow']) == (1, 1)


def test_the_green_fleet_explores_at_the_end_of_the_turn():
    g = _game()
    p = g.players[0]
    p.advance_fleet('green')
    p.coins = 2  # the second green space costs 2 coins
    card = Card('Lab', Color.GREEN, age=1)
    g.apply(0, Action.build(card).with_naval())
    assert p.fleets['green'] == 2 and p.coins == 0 and p.pending_explores == [1]


# -- whole games -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    'combo', [SECOND, SECOND + ('cities2',), SECOND + ('leaders2', 'cities2'), FIRST, FIRST + ('leaders', 'cities')]
)
@pytest.mark.parametrize('n', [3, 5, 7])
def test_armada_games_play_and_replay(combo, n):
    content = load_content(*combo)
    rng = random.Random(n * 7 + len(combo))
    names = sorted({w.name for w in content.wonders})
    setups = [PlayerSetup(f'p{i}', w, rng.choice('AB')) for i, w in enumerate(rng.sample(names, n))]
    kinds = ['heuristic', 'board', 'greedy', 'random']
    agents = [AGENTS[kinds[i % 4]].build(random.Random(i)) for i in range(n)]
    game, record = play_match(content, setups, agents, rng=rng, expansions=Ruleset.of(*combo).expansions)
    assert all(s.shipyard for s in record.players) and len({s.shipyard for s in record.players}) == n
    assert any(move[0].endswith('+') for turns in record.moves.values() for turn in turns for move in turn)
    replayed = GameRecord.from_json(record.to_json()).replay(load_content(*record.expansions))
    assert [replayed.score(i).as_dict() for i in range(n)] == [game.score(i).as_dict() for i in range(n)]
    assert [p.fleets for p in replayed.players] == [p.fleets for p in game.players]


def test_the_env_has_a_naval_block_and_plays_armada():
    env = WondersEnv(n_players=4, expansions=SECOND)
    assert env.action_space.naval and env.action_space.n == 2 * env.action_space.n_base
    pick = random.Random(0)
    observations, _ = env.reset(3)
    while observations:
        actions = {s: pick.choice([i for i, ok in enumerate(o['action_mask']) if ok]) for s, o in observations.items()}
        observations, *_ = env.step(actions)
    record = env.record()
    replayed = record.replay(load_content(*SECOND))
    assert [replayed.score(i).total for i in range(4)] == env.scores()
