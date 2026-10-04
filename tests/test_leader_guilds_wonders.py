"""Tests for the Leaders guilds, the Rome / Petra / Byzantium boards and the score columns."""

import random

import pytest

from wonders.agents import BuilderAgent, GreedyAgent, HeuristicAgent
from wonders.agents.match import play_match
from wonders.cards.base import Card, Color
from wonders.cards.content import load_content
from wonders.cards.library import wonder_ephesos_a, wonder_rhodes_a
from wonders.engine.actions import Action, DecisionKind
from wonders.engine.game import Game, drive
from wonders.engine.player import Player
from wonders.engine.record import GameRecord, PlayerSetup

CONTENT = load_content('base', 'leaders', 'cities')


def table(first_wonder, n=3):
    others = [wonder_rhodes_a, wonder_ephesos_a]
    return Game([Player('P0', first_wonder, coins=20)] + [Player(f'P{i}', others[i % 2]()) for i in range(1, n)])


def own(g, seat, name):
    g.players[seat].add_card(CONTENT.card(name))


def builder(g):
    """Let seat 0 build guilds (Ramses) and wonder stages (Architect Firm) without resources."""
    own(g, 0, 'Ramses')
    own(g, 0, 'Architect Firm')


def pick(kind_names):
    """A chooser for power decisions: pick the option with the given name per decision kind."""

    def choose(point):
        name = kind_names[point.kind]
        return {seat: Action.build(next(c for c in opts if c.name == name)) for seat, opts in point.options.items()}

    return choose


# -- content ------------------------------------------------------------------------


def test_new_guilds_and_boards_are_loaded():
    guilds = {c.name for c in CONTENT.guilds()}
    assert {'Gamers Guild', 'Courtesans Guild', 'Diplomats Guild', 'Architects Guild'} <= guilds
    boards = {(w.name, w.side): w for w in CONTENT.wonders}
    assert len(boards[('The Colosseum of Rome', 'A')].stages) == 2
    assert len(boards[('The Colosseum of Rome', 'B')].stages) == 3
    assert boards[('The Colosseum of Rome', 'A')].starting_resource is None
    assert len(boards[('The Al-Khazneh of Petra', 'B')].stages) == 2
    assert boards[('The Al-Khazneh of Petra', 'B')].stages[1].cost.coins == 14
    assert len(boards[('The Hagia Sophia of Byzantium', 'A')].stages) == 3


def test_leaders_guilds_only_join_age_iii_with_the_leaders_expansion():
    base = load_content('base')
    assert 'Gamers Guild' not in {c.name for c in base.guilds()}


# -- guilds ---------------------------------------------------------------------------


def test_gamers_diplomats_architects_guild_scoring():
    g = table(CONTENT.wonder('The Pyramids of Giza', 'A'))
    g.players[0].coins = 10
    own(g, 0, 'Gamers Guild')
    assert g.score(0).guilds == 3  # 10 // 3
    own(g, 0, 'Diplomats Guild')
    own(g, 1, 'Plato')
    own(g, 2, 'Nero')
    assert g.score(0).guilds == 3 + 2  # one leader in each neighbouring city
    own(g, 0, 'Architects Guild')
    own(g, 1, 'Workers Guild')
    assert g.score(0).guilds == 3 + 2 + 3  # 3 VP per purple card next door


def test_courtesans_guild_copies_a_neighbour_leader():
    g = table(CONTENT.wonder('The Pyramids of Giza', 'A'))
    own(g, 1, 'Croesus')  # immediate 6 coins
    own(g, 2, 'Nefertiti')  # 4 VP at game end
    builder(g)
    coins = g.players[0].coins
    g.apply(0, Action.build(CONTENT.card('Courtesans Guild')))
    assert g.players[0].pending_leader_copies == 1
    drive(g._power_steps(), pick({DecisionKind.COPY_LEADER: 'Croesus'}))
    assert [c.name for c in g.players[0].copied_leaders] == ['Croesus']
    assert g.players[0].coins == coins + 6  # Croesus's immediate effect is copied too
    assert not g.players[0].has('Croesus')  # a copy, not an owned leader
    h = table(CONTENT.wonder('The Pyramids of Giza', 'A'))
    own(h, 2, 'Nefertiti')
    builder(h)
    h.apply(0, Action.build(CONTENT.card('Courtesans Guild')))
    drive(h._power_steps(), pick({DecisionKind.COPY_LEADER: 'Nefertiti'}))
    assert h.score(0).leaders == 4  # the copied leader scores in the leaders column


def test_courtesans_guild_fizzles_without_neighbour_leaders():
    g = table(CONTENT.wonder('The Pyramids of Giza', 'A'))
    builder(g)
    g.apply(0, Action.build(CONTENT.card('Courtesans Guild')))
    with pytest.raises(StopIteration):
        next(g._power_steps())
    assert g.players[0].copied_leaders == []


# -- Rome -----------------------------------------------------------------------------


def test_rome_a_produces_nothing_and_recruits_leaders_free():
    g = table(CONTENT.wonder('The Colosseum of Rome', 'A'))
    assert g.players[0].production_slots() == []
    g.players[0].coins = 0
    assert g.can_build_card(0, CONTENT.card('Cleopatra'))


def test_rome_b_discounts_leaders_for_self_and_neighbours():
    g = table(CONTENT.wonder('The Colosseum of Rome', 'B'))
    cleopatra = CONTENT.card('Cleopatra')
    printed = cleopatra.cost.coins
    assert g._coin_cost(0, cleopatra) == max(0, printed - 2)
    assert g._coin_cost(1, cleopatra) == max(0, printed - 1)  # a neighbour of Rome


def test_rome_b_stage_one_draws_from_the_leader_box_and_stage_two_recruits():
    g = table(CONTENT.wonder('The Colosseum of Rome', 'B'))
    g.leader_box = [CONTENT.card(n) for n in ('Plato', 'Nero', 'Midas', 'Sappho', 'Croesus')]
    builder(g)
    g.apply(0, Action.wonder(Card('x', Color.BROWN)))
    assert [c.name for c in g.players[0].leaders_hand] == ['Croesus', 'Sappho', 'Midas', 'Nero']
    assert [c.name for c in g.leader_box] == ['Plato']
    g.apply(0, Action.wonder(Card('y', Color.BROWN)))
    assert g.players[0].pending_extra_recruits == 1
    drive(g._power_steps(), pick({DecisionKind.LEADER_BUILD: 'Sappho'}))
    assert g.players[0].has('Sappho') and len(g.players[0].leaders_hand) == 3


# -- Petra / Byzantium -----------------------------------------------------------------


def test_petra_b_stage_one_hits_every_other_player():
    g = table(CONTENT.wonder('The Al-Khazneh of Petra', 'B'), n=4)
    for p in g.players[1:]:
        p.coins = 5
    builder(g)
    g.apply(0, Action.wonder(Card('x', Color.BROWN)))
    assert [p.coins for p in g.players[1:]] == [3, 3, 3]


def test_byzantium_stages_grant_diplomacy():
    g = table(CONTENT.wonder('The Hagia Sophia of Byzantium', 'B'))
    builder(g)
    g.apply(0, Action.wonder(Card('x', Color.BROWN)))
    assert g.players[0].diplomacy == 1 and g.score(0).wonders == 3


# -- scoring columns --------------------------------------------------------------------


def test_leaders_and_cities_have_their_own_score_columns():
    g = table(CONTENT.wonder('The Pyramids of Giza', 'A'))
    own(g, 0, 'Nefertiti')
    own(g, 0, 'Capitol')
    b = g.score(0)
    assert b.leaders == 4 and b.cities > 0 and b.commercial == 0
    assert b.total == sum(v for k, v in b.as_dict().items() if k != 'total')


# -- end to end ---------------------------------------------------------------------------


@pytest.mark.parametrize('side', ['A', 'B'])
def test_games_with_new_boards_replay_exactly(side):
    expansions = ('base', 'leaders', 'cities')
    setups = [
        PlayerSetup('Rome', 'The Colosseum of Rome', side),
        PlayerSetup('Petra', 'The Al-Khazneh of Petra', side),
        PlayerSetup('Byz', 'The Hagia Sophia of Byzantium', side),
        PlayerSetup('Hal', 'The Mausoleum of Halicarnassus', 'A'),
    ]
    for seed in range(4):
        agents = [BuilderAgent(), BuilderAgent(), HeuristicAgent(), GreedyAgent()]
        game, record = play_match(CONTENT, setups, agents, rng=random.Random(seed), expansions=expansions)
        replayed = GameRecord.from_json(record.to_json()).replay(CONTENT)
        assert [replayed.score(i).as_dict() for i in range(4)] == [game.score(i).as_dict() for i in range(4)]
        assert record.leader_box  # the undealt leaders are recorded for Rome B's draws


def test_solomon_builds_from_the_discard_pile_during_recruitment():
    g = table(CONTENT.wonder('The Pyramids of Giza', 'A'))
    g.discard_pile.append(CONTENT.card('Nefertiti'))
    g.players[0].leaders_hand = [CONTENT.card('Solomon')]
    asked = []

    def choose(point):
        asked.append(point.kind)
        if point.kind is DecisionKind.RECRUIT:
            return {0: Action.build(point.options[0][0])}
        return {0: Action.free(point.options[0][0])}

    drive(g.recruit_steps(1), choose)
    assert asked == [DecisionKind.RECRUIT, DecisionKind.DISCARD_BUILD]
    assert g.players[0].has('Solomon') and g.players[0].has('Nefertiti')
