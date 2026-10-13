"""Second-edition Leaders and Cities (``leaders2`` / ``cities2``): content, setup rules and the new powers."""

import random
from collections import Counter

import pytest

from wonders.agents.bench import AGENTS
from wonders.agents.match import play_match
from wonders.cards.base import Card, Color, Cost, ScienceSymbol
from wonders.cards.content import load_content
from wonders.cards.decks import deck_size
from wonders.cards.library import wonder_gizah_a, wonder_rhodes_a
from wonders.cards.wonder import Wonder, WonderStage
from wonders.effects.library import (
    BuildIncome,
    ExtraRecruit,
    FreeBuildPerAge,
    GainToken,
    Majority,
    NoDefeats,
    OnlyLeader,
    Plunder,
    ReserveBonus,
    StagePlunder,
    TokenPairs,
    TokenPurge,
    VictoryPoints,
)
from wonders.engine.actions import Action, DecisionKind
from wonders.engine.game import Game, drive, science_score
from wonders.engine.player import Player
from wonders.engine.record import GameRecord, PlayerSetup
from wonders.play.session import GameConfig, HumanSession, SeatConfig
from wonders.ruleset import Ruleset, rules_of

SECOND = ('base2', 'leaders2', 'cities2')

# -- content and setup ----------------------------------------------------------


def test_second_edition_leaders_and_cities_content():
    leaders = load_content('base2', 'leaders2')
    both = load_content(*SECOND)
    assert len(leaders.leaders()) == 49  # 34 standard + 15 expert; the 6 bonus leaders need Cities
    assert len(both.leaders()) == 55
    bonus = {c.name for c in both.leaders()} - {c.name for c in leaders.leaders()}
    assert bonus == {'Caligula', 'Diocletian', 'Octavia', 'Arsinoe', 'Aspasia', 'Darius'}
    assert len(both.guilds()) == 10  # the second-edition Leaders brings no guilds
    assert {w.name for w in both.wonders} >= {'Roma', 'Abu Simbel', 'Byzantium', 'Petra'}
    assert sum(1 for c in both.cards if c.color is Color.BLACK) == 42
    for n in range(3, 8):
        for age in (1, 2, 3):
            assert (deck_size(both.cards, age, n).hand, deck_size(both.cards, age, n).leftover) == (8, 0)
    ages = {c.name for c in both.leaders() if c.age_cost}
    assert ages == {'Aganice', 'Arsinoe', 'Cornelia', 'Cynisca', 'Euryptyle', 'Makeda', 'Nitocris', 'Phryne', 'Theano'}


def test_second_edition_leader_rules_only_apply_with_leaders2():
    assert (rules_of(('base', 'leaders')).leader_draft_left, rules_of(('base', 'leaders')).return_last_leaders) == (
        1,
        0,
    )
    rules = rules_of(('base2', 'leaders2'))
    assert (rules.leader_draft_left, rules.return_last_leaders, rules.starting_coins) == (0, 1, 6)


def test_the_menu_maps_leaders_and_cities_to_the_second_edition():
    config = GameConfig(expansions=['leaders', 'cities'], edition=2, seats=[SeatConfig(), SeatConfig('greedy')] * 1)
    config.seats = [SeatConfig(), SeatConfig('greedy'), SeatConfig('heuristic')]
    assert HumanSession(config).ruleset.expansions == SECOND


def _game(n=3, expansions=SECOND, wonders=None):
    boards = wonders or [wonder_gizah_a(), wonder_rhodes_a(), wonder_gizah_a()][:n]
    return Game([Player(f'P{i}', w) for i, w in enumerate(boards)], expansions)


def test_the_second_edition_leader_draft_passes_right():
    g = _game()
    hands = [[Card(f'L{s}{k}', Color.WHITE) for k in range(3)] for s in range(3)]
    seen = []

    def choose(point):
        seen.append({i: [c.name for c in cards] for i, cards in point.options.items()})
        return {i: Action.build(cards[0]) for i, cards in point.options.items()}

    drive(g.draft_steps(hands), choose)
    assert seen[1][0] == ['L11', 'L12']  # seat 0 now holds what seat 1 (its right neighbour) passed


def test_unchosen_leaders_return_to_the_box_after_age_three_but_not_on_roma_night():
    roma_night = Wonder('Roma', None, (WonderStage(Cost(), (ExtraRecruit(1),)),), side='B')
    g = _game(wonders=[wonder_gizah_a(), roma_night, wonder_rhodes_a()])
    for p in g.players:
        p.leaders_hand = [Card(f'{p.name}-a', Color.WHITE), Card(f'{p.name}-b', Color.WHITE)]
    drive(g.recruit_steps(3), lambda point: {i: Action.discard(cards[0]) for i, cards in point.options.items()})
    assert g.players[0].leaders_hand == [] and g.players[2].leaders_hand == []
    assert [c.name for c in g.players[1].leaders_hand] == ['P1-b']  # Roma night keeps it for an extra recruit
    assert {'P0-b', 'P2-b'} <= {c.name for c in g.leader_box}


# -- leaders that cost the Age --------------------------------------------------


def test_a_leader_costs_the_current_age_and_entombs_for_six():
    nitocris = next(c for c in load_content(*SECOND).leaders() if c.name == 'Nitocris')
    g = _game()
    p = g.players[0]
    for age in (1, 2, 3):
        g.age = age
        assert g._coin_cost(0, nitocris) == age
    g.age, p.coins = 2, 2
    g.decision = DecisionKind.RECRUIT
    g.apply(0, Action.build(nitocris))
    assert p.coins == 0 and p.tokens == [3]  # paid 2 coins, took the Age II victory token
    p.entomb(nitocris)
    assert g.score(0).wonders == 6


# -- the new powers ---------------------------------------------------------------


def grant(game, seat, *effects, color=Color.WHITE, name=None):
    game.players[seat].add_card(Card(name or f'Power {seat}', color, effects=effects))


def test_majorities_need_more_than_each_neighbour():
    g = _game()
    blue = Majority('colors', 5, [Color.BLUE])
    grant(g, 0, VictoryPoints(1), color=Color.BLUE, name='B0')
    grant(g, 0, VictoryPoints(1), color=Color.BLUE, name='B1')
    grant(g, 1, VictoryPoints(1), color=Color.BLUE, name='B2')
    assert blue.victory_points(g.context(0)) == 5
    grant(g, 2, VictoryPoints(1), color=Color.BLUE, name='B3')
    grant(g, 2, VictoryPoints(1), color=Color.BLUE, name='B4')
    assert blue.victory_points(g.context(0)) == 0  # a tie with one neighbour is not more
    g.players[0].coins, g.players[1].coins, g.players[2].coins = 9, 8, 3
    assert Majority('coins').victory_points(g.context(0)) == 5


def test_military_token_powers():
    g = _game()
    p = g.players[0]
    p.tokens = [1, 1, 3, 5, 5, 5, -1]
    assert NoDefeats(6).victory_points(g.context(0)) == 0
    assert TokenPairs().victory_points(g.context(0)) == 1 + 5  # one pair of 1s, one pair of 5s
    g.players[1].tokens, g.players[2].tokens = [3, 1], []
    TokenPurge().on_build(g.context(0))
    assert p.tokens == [1, 1, 3, 5, 5, 5] and g.players[1].tokens == [3] and g.players[2].tokens == []
    assert NoDefeats(6).victory_points(g.context(0)) == 6
    g.age = 2
    GainToken(current_age=True).on_build(g.context(0))
    assert p.tokens[-1] == 3


def test_only_leader():
    g = _game()
    agrippina = Card('Agrippina', Color.WHITE, effects=(OnlyLeader(7),))
    g.players[0].add_card(agrippina)
    assert OnlyLeader(7).victory_points(g.context(0)) == 7
    grant(g, 0, VictoryPoints(1))
    assert OnlyLeader(7).victory_points(g.context(0)) == 0


def test_berenice_takes_one_extra_coin_per_turn_from_the_reserve():
    g = _game()
    grant(g, 0, ReserveBonus(1))
    p = g.players[0]
    p.coins = 0
    g._dispatch(0, Action.discard(Card('X', Color.BLUE)))
    g._dispatch(0, Action.discard(Card('Y', Color.BLUE)))  # two reserve gains in one turn
    g._flush_pending()
    assert p.coins == 3 + 3 + 1
    g.players[1].coins, g.players[0].coins = 10, 0
    g._settle(1, 0, type('Pay', (), {'total': 2, 'left': 0, 'right': 2})())  # seat 1 buys from seat 0 (its right)
    g._flush_pending()
    assert p.coins == 2  # coins from a neighbour are not from the reserve


def test_octavia_earns_on_wonder_stages_and_others_lose_a_coin():
    free_stage = Wonder('Free', None, (WonderStage(Cost(), (VictoryPoints(1),)),))
    g = _game(wonders=[free_stage, wonder_rhodes_a(), wonder_gizah_a()])
    grant(g, 0, BuildIncome(2, wonder=True), StagePlunder(1))
    for p in g.players:
        p.coins = 5
    g.apply(0, Action.wonder(Card('S', Color.BLUE)))
    assert [p.coins for p in g.players] == [7, 4, 4]
    g.apply(0, Action.build(Card('C', Color.BLUE)))  # a card is no wonder stage
    assert [p.coins for p in g.players] == [7, 4, 4]


def test_caligula_builds_one_black_card_per_age_for_free():
    g = _game()
    grant(g, 0, FreeBuildPerAge(1, ('black',)))
    g.players[0].coins = 0
    costly = Cost(coins=3)
    black, blue = Card('K', Color.BLACK, costly), Card('U', Color.BLUE, costly)
    assert g._can_build_free(0, black) and not g._can_build_free(0, blue)
    g.apply(0, Action.free(black))
    assert not g._can_build_free(0, Card('K2', Color.BLACK, costly))  # once per Age


def test_arsinoe_takes_the_age_in_coins_from_everyone_else():
    g = _game()
    g.age = 3
    for p in g.players:
        p.coins = 5
    Plunder(0, 'others', per_age=True).on_build(g.context(0))
    assert [p.coins for p in g.players] == [5, 2, 2]


def test_science_most_and_replace():
    counts = Counter({ScienceSymbol.TABLET: 3, ScienceSymbol.GEAR: 1})
    plain = science_score(counts, set_value=7)
    assert plain == 9 + 1  # no complete set
    assert science_score(counts, set_value=7, most=1) == 16 + 1  # Enheduanna: a fourth tablet
    assert science_score(counts, set_value=7, replace=1) == 16  # Aganice: the gear becomes a fourth tablet
    assert (
        science_score(Counter({ScienceSymbol.TABLET: 2, ScienceSymbol.GEAR: 2}), set_value=7, replace=1)
        == 4 + 1 + 1 + 7
    )


# -- whole games --------------------------------------------------------------------


@pytest.mark.parametrize('combo', [('base2', 'leaders2'), ('base2', 'cities2'), SECOND])
@pytest.mark.parametrize('n', [3, 6])
def test_second_edition_expansion_games_play_and_replay(combo, n):
    content = load_content(*combo)
    rng = random.Random(n * 10 + len(combo))
    names = sorted({w.name for w in content.wonders})
    setups = [PlayerSetup(f'p{i}', w, rng.choice('AB')) for i, w in enumerate(rng.sample(names, n))]
    agents = [AGENTS[a].build(random.Random(i)) for i, a in enumerate(['heuristic', 'board', 'greedy'] * 3)][:n]
    game, record = play_match(content, setups, agents, rng=rng, expansions=Ruleset.of(*combo).expansions)
    replayed = GameRecord.from_json(record.to_json()).replay(load_content(*record.expansions))
    assert [replayed.score(i).total for i in range(n)] == [game.score(i).total for i in range(n)]
