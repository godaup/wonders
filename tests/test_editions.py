"""The game editions: the second-edition base game (``base2``), the edition selector and its new powers."""

import random
from collections import Counter

import pytest

from wonders.agents.bench import AGENTS
from wonders.agents.match import play_match
from wonders.cards.base import Card, Color, Cost
from wonders.cards.base import Resource as R
from wonders.cards.content import load_content
from wonders.cards.decks import deck_size
from wonders.cards.library import wonder_gizah_a, wonder_rhodes_a
from wonders.effects.base import Context
from wonders.effects.library import FreeFirstOfColor, FreeTurnBuild, Military, WonderComplete
from wonders.engine.actions import Action, ActionKind
from wonders.engine.game import Game
from wonders.engine.player import Player
from wonders.engine.record import GameRecord, PlayerSetup
from wonders.ruleset import EDITIONS, Ruleset, family_of, has_family, rules_of, with_edition

# -- the edition selector ------------------------------------------------------


def test_the_first_edition_base_game_stays_the_implied_default():
    assert Ruleset.of().expansions == ('base',)
    assert Ruleset.of('leaders', 'cities').expansions == ('base', 'leaders', 'cities')
    assert Ruleset.of('leaders').edition == 1


def test_a_second_edition_ruleset_has_its_own_base_game_first():
    ruleset = Ruleset.of('base2')
    assert ruleset.expansions == ('base2',) and ruleset.edition == 2
    assert ruleset.uses('base') and not ruleset.uses('leaders')
    assert rules_of(ruleset.expansions) == rules_of(('base',))  # the meta rules are the base game's


@pytest.mark.parametrize(
    'ids, message',
    [
        (('base', 'base2'), 'one edition'),
        (('base2', 'leaders'), 'Leaders'),
        (('base2', 'cities', 'wonderpack'), 'Cities'),
    ],
)
def test_editions_do_not_mix(ids, message):
    with pytest.raises(ValueError, match=message):
        Ruleset.of(*ids)


def test_with_edition_maps_every_family_to_its_edition():
    assert with_edition([], 1) == ('base',)
    assert with_edition(['leaders', 'cities'], 1) == ('base', 'leaders', 'cities')
    assert with_edition([], 2) == ('base2',)
    assert with_edition(['base'], 2) == ('base2',)
    with pytest.raises(ValueError, match='Leaders is not available in the second edition'):
        with_edition(['leaders'], 2)
    with pytest.raises(ValueError, match='unknown edition'):
        with_edition([], 3)
    assert set(EDITIONS) == {1, 2}


def test_families_answer_for_every_edition():
    assert family_of('base2') == 'base' and family_of('cities') == 'cities' and family_of('my-ext') == 'my-ext'
    assert has_family(('base2',), 'base') and has_family(('base', 'cities'), 'cities')
    assert not has_family(('base2',), 'cities')


# -- the second-edition content -------------------------------------------------

# The official "List of cards" sheet: 148 Age cards (49 / 49 / 50 incl. 10 guilds), every card at 3-7 players.
OFFICIAL_AT_SEVEN = {1: 49, 2: 49, 3: 40}


@pytest.fixture(scope='module')
def base2():
    return load_content('base2')


def test_second_edition_deck_matches_the_official_card_list(base2):
    cards = [c for c in base2.cards if not c.guild]
    for age, total in OFFICIAL_AT_SEVEN.items():
        assert sum(c.copies_for(7) for c in cards if c.age == age) == total
    assert len(base2.guilds()) == 10
    names = {c.name for c in base2.cards}
    assert {'Well', 'Ludus', 'Castrum', 'Decorators Guild'} <= names
    assert not {'Pawnshop', 'Strategists Guild'} & names
    for n in range(3, 8):
        for age in (1, 2, 3):
            size = deck_size(base2.cards, age, n)
            assert (size.hand, size.leftover) == (7, 0), (n, age)


def test_second_edition_chains_and_changed_cards(base2):
    by_name = {(c.age, c.name): c for c in base2.cards}
    names = {c.name for c in base2.cards}
    assert all(target in names for c in base2.cards for target in c.chain_enables)
    assert by_name[(1, 'Altar')].chain_enables == ('Pantheon',)  # skips Age II now
    assert by_name[(1, 'Well')].chain_enables == ('Statue',)
    assert by_name[(1, 'Theater')].chain_enables == ('Gardens',)
    assert sum(e.victory_points(None) for e in by_name[(1, 'Altar')].effects) == 3  # type: ignore[arg-type]
    assert by_name[(3, 'Fortifications')].cost.resources == Counter({R.ORE: 3, R.BRICKS: 1})


def test_second_edition_wonders(base2):
    boards = {(w.name, w.side): w for w in base2.wonders}
    assert len(boards) == 14 and {name for name, _side in boards} == {
        'Alexandria',
        'Babylon',
        'Ephesos',
        'Gizah',
        'Halikarnassos',
        'Olympia',
        'Rhodos',
    }
    assert boards[('Babylon', 'A')].starting_resource is R.WOOD
    assert boards[('Olympia', 'A')].starting_resource is R.BRICKS
    assert len(boards[('Babylon', 'B')].stages) == 2 and len(boards[('Gizah', 'B')].stages) == 4
    first = {w.name for w in load_content('base').wonders}
    assert not first & {name for name, _side in boards}  # the editions' boards never share a name


# -- the new powers ------------------------------------------------------------


def two_players():
    return Game([Player('P0', wonder_gizah_a()), Player('P1', wonder_rhodes_a())])


def grant(game, seat, effect):
    game.players[seat].add_card(Card(f'Power {seat}', Color.BLUE, effects=(effect,)))


def red(name, cost=Cost(coins=2)):
    return Card(name, Color.RED, cost, (Military(1),))


def test_the_first_card_of_each_colour_is_free():
    g = two_players()
    grant(g, 0, FreeFirstOfColor())  # the power card itself is blue
    p = g.players[0]
    p.coins = g.players[1].coins = 0
    assert g.can_build_card(0, red('R1'))  # no red card yet: free although it costs 2 coins
    g.apply(0, Action.build(red('R1')))
    assert p.coins == 0
    assert not g.can_build_card(0, red('R2'))  # now red is not the first any more
    assert not g.can_build_card(0, Card('B2', Color.BLUE, Cost(coins=1)))  # blue: the power card came first
    assert not g.can_build_card(1, red('R1'))  # only the owner


def _offers_by_turn(game, hands):
    offered = []

    def policy(g, seat, hand):
        actions = g.legal_actions(seat, hand)
        if seat == 0:
            offered.append((g.turn, any(a.kind is ActionKind.FREE for a in actions)))
        return next(
            (a for a in actions if a.kind is ActionKind.FREE), next(a for a in actions if a.kind is ActionKind.DISCARD)
        )

    game.play_age(1, hands, policy)
    return offered


@pytest.mark.parametrize('turn, free_on', [('first', [True, False, False]), ('last', [False, False, True])])
def test_the_first_or_last_card_of_each_age_is_free(turn, free_on):
    g = two_players()
    grant(g, 0, FreeTurnBuild(turn))
    costly = Cost(resources=Counter({R.GLASS: 2}))  # unaffordable: only a free build plays it
    hands = [[Card(f'C{k}', Color.BLUE, costly) for k in range(4)], [Card(f'D{k}', Color.BLUE) for k in range(4)]]
    assert [free for _t, free in _offers_by_turn(g, hands)] == free_on
    assert len(g.players[0].cards) == 2  # the power card and the one free build
    assert g.players[0].free_builds_used == 0  # no per-Age allowance spent


def test_decorators_guild_scores_only_a_complete_wonder():
    g = two_players()
    p = g.players[0]
    effect = WonderComplete(7)
    stages = list(p.wonder.stages)
    for k, stage in enumerate(stages):
        assert effect.victory_points(Context(p, g.left_of(0), g.right_of(0))) == 0
        p.add_stage(stage, k, Card(f'S{k}', Color.BROWN))
    assert effect.victory_points(Context(p, g.left_of(0), g.right_of(0))) == 7


# -- whole games ---------------------------------------------------------------


@pytest.mark.parametrize('n', [3, 5, 7])
def test_second_edition_games_play_and_replay(base2, n):
    rng = random.Random(n)
    names = sorted({w.name for w in base2.wonders})
    setups = [PlayerSetup(f'p{i}', w, rng.choice('AB')) for i, w in enumerate(rng.sample(names, n))]
    agents = [AGENTS[a].build(random.Random(i)) for i, a in enumerate(['heuristic', 'board', 'greedy'] * 3)][:n]
    game, record = play_match(base2, setups, agents, rng=rng, expansions=('base2',))
    assert record.expansions == ('base2',)
    replayed = GameRecord.from_json(record.to_json()).replay(load_content(*record.expansions))
    assert [replayed.score(i).total for i in range(n)] == [game.score(i).total for i in range(n)]
