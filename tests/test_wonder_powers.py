"""Tests for the special wonder powers: Babylon B, Olympia A/B and Halicarnassus."""

import random

import pytest

from wonders.agents import HeuristicAgent, RandomAgent
from wonders.agents.match import play_match
from wonders.cards.base import Card, Color, Cost
from wonders.cards.base import Ressource as R
from wonders.cards.content import load_content
from wonders.cards.library import wonder_gizah_a, wonder_rhodes_a
from wonders.effects.library import (
    BuildFromDiscard,
    CopyGuild,
    FreeBuildPerAge,
    PlayLastCard,
    ScienceWild,
    Tally,
    VictoryPoints,
)
from wonders.engine.actions import Action, ActionKind, DecisionKind
from wonders.engine.game import Game, IllegalAction
from wonders.engine.observation import build_observation
from wonders.engine.player import Player
from wonders.engine.record import GameRecord, PlayerSetup


def two_players():
    return Game([Player('P0', wonder_gizah_a()), Player('P1', wonder_rhodes_a())])


def grant(game, seat, effect):
    """Give a seat a power by pre-building a dummy card carrying it."""
    game.players[seat].add_card(Card(f'Power {seat}', Color.BLUE, effects=(effect,)))


def blue(name, points=1, cost=Cost()):
    return Card(name, Color.BLUE, cost, (VictoryPoints(points),))


def first_of(kind):
    def policy(game, seat, hand):
        actions = game.legal_actions(seat, hand)
        return next((a for a in actions if a.kind is kind), actions[0])

    return policy


# -- Babylon B: play the 7th card ------------------------------------------


def test_last_card_is_played_instead_of_discarded():
    g = two_players()
    grant(g, 0, PlayLastCard())
    hands = [[blue('A0'), blue('A1'), blue('A2')], [blue('B0'), blue('B1'), blue('B2')]]
    asked = []
    g.play_age(1, hands, first_of(ActionKind.BUILD), on_extra=lambda point, actions: asked.append(point))
    # 2 regular plays each, plus the Babylon seat's leftover card.
    assert len(g.players[0].cards) == 1 + 3
    assert len(g.players[1].cards) == 2
    assert len(g.discard_pile) == 1  # only seat 1's leftover
    assert [(p.kind, list(p.options)) for p in asked] == [(DecisionKind.LAST_CARD, [0])]


# -- Olympia A: one free build per Age ---------------------------------------


def test_free_build_is_offered_once_per_age_and_costs_nothing():
    g = two_players()
    grant(g, 0, FreeBuildPerAge())
    g.players[0].coins = 0
    palace = blue('Palace', 8, Cost.of(R.GLASS, R.GLASS, R.ORE))
    actions = g.legal_actions(0, [palace])
    assert Action.free(palace) in actions and Action.build(palace) not in actions
    g.apply(0, Action.free(palace))
    assert g.players[0].has('Palace') and g.players[0].coins == 0
    assert Action.free(blue('Temple', 3, Cost(coins=2))) not in g.legal_actions(0, [blue('Temple', 3, Cost(coins=2))])
    with pytest.raises(IllegalAction):
        g.apply(0, Action.free(blue('Other', 1, Cost(coins=2))))


def test_free_build_not_offered_for_already_free_cards():
    g = two_players()
    grant(g, 0, FreeBuildPerAge())
    altar = blue('Altar')  # costs nothing anyway
    assert all(a.kind is not ActionKind.FREE for a in g.legal_actions(0, [altar]))


def test_free_build_resets_each_age():
    g = two_players()
    grant(g, 0, FreeBuildPerAge())
    g.players[0].free_builds_used = 1
    hands = [[blue('A0'), blue('A1')], [blue('B0'), blue('B1')]]
    g.play_age(2, hands, first_of(ActionKind.DISCARD))
    assert g.players[0].free_builds_left == 1


# -- Halicarnassus: build from the discard pile -------------------------------


def test_discard_build_after_the_turn_sees_this_turns_discards():
    g = two_players()
    statue = blue('Statue', 4, Cost.of(R.ORE, R.ORE, R.WOOD))
    trigger = Card('Mausoleum', Color.BLUE, effects=(BuildFromDiscard(),))
    hands = [[trigger, blue('X')], [statue, blue('Y')]]

    def policy(game, seat, hand):
        legal = game.legal_actions(seat, hand)
        if game.decision is DecisionKind.DISCARD_BUILD:
            assert {a.kind for a in legal} == {ActionKind.FREE}
            return next(a for a in legal if a.card.name == 'Statue')
        want = ActionKind.BUILD if seat == 0 else ActionKind.DISCARD
        return next(a for a in legal if a.kind is want and a.card.name in ('Mausoleum', 'Statue'))

    g.play_age(1, hands, policy)
    assert g.players[0].has('Statue')
    assert all(c.name != 'Statue' for c in g.discard_pile)
    assert g.players[0].pending_discard_builds == 0


def test_discard_build_fizzles_on_empty_pile():
    g = two_players()
    g.apply(0, Action.build(Card('Mausoleum', Color.BLUE, effects=(BuildFromDiscard(),))))
    steps = g._discard_builds()
    with pytest.raises(StopIteration):
        next(steps)
    assert g.players[0].pending_discard_builds == 0


def test_discard_build_offers_only_unowned_cards_once():
    g = two_players()
    g.players[0].add_card(blue('Owned'))
    g.decision = DecisionKind.DISCARD_BUILD
    pile = [blue('Owned'), blue('Dup'), blue('Dup')]
    assert [a.card.name for a in g.legal_actions(0, pile)] == ['Dup']


# -- Olympia B: copy a neighbour guild ----------------------------------------


def test_copy_guild_takes_best_neighbour_guild():
    g = Game([Player('P0', wonder_gizah_a()), Player('P1', wonder_rhodes_a()), Player('P2', wonder_rhodes_a())])
    grant(g, 0, CopyGuild())
    g.players[0].add_card(Card('Mine', Color.BROWN))
    g.players[0].add_card(Card('Pit', Color.BROWN))
    # Copied guilds score in the copier's own context: here, its own brown cards.
    g.players[1].add_card(Card('Small Guild', Color.PURPLE, effects=(VictoryPoints(1),)))
    g.players[2].add_card(
        Card('Craft Guild', Color.PURPLE, effects=(Tally('colors', 'self', {Color.BROWN}, vp_each=2),))
    )
    assert g.score(0).guilds == 4


def test_copy_guild_can_take_science():
    g = two_players()
    grant(g, 0, CopyGuild())
    g.players[1].add_card(Card('Scientists Guild', Color.PURPLE, effects=(ScienceWild(),)))
    assert g.score(0).science == 1 and g.score(0).guilds == 0


def test_copy_guild_without_candidates_scores_nothing():
    g = two_players()
    grant(g, 0, CopyGuild())
    assert g.score(0).guilds == 0


# -- integration: content, observation, record ---------------------------------


def test_base_wonders_carry_mechanized_powers():
    content = load_content('base')
    powers = {
        (w.name, w.side): {type(e).__name__ for s in w.stages for e in s.effects}
        for w in content.wonders
        if w.name
        in ('The Hanging Gardens of Babylon', 'The Statue of Zeus in Olympia', 'The Mausoleum of Halicarnassus')
    }
    assert 'PlayLastCard' in powers[('The Hanging Gardens of Babylon', 'B')]
    assert 'FreeBuildPerAge' in powers[('The Statue of Zeus in Olympia', 'A')]
    assert 'CopyGuild' in powers[('The Statue of Zeus in Olympia', 'B')]
    assert 'BuildFromDiscard' in powers[('The Mausoleum of Halicarnassus', 'A')]
    assert 'BuildFromDiscard' in powers[('The Mausoleum of Halicarnassus', 'B')]
    assert all('Special' not in kinds for kinds in powers.values())


def test_observation_exposes_decision_and_free_builds():
    g = two_players()
    grant(g, 0, FreeBuildPerAge())
    obs = build_observation(g, 0, [blue('A')])
    assert obs.decision == 'play' and obs.free_builds_left == 1
    assert obs.to_dict()['free_builds_left'] == 1


SPECIAL_SETUPS = [
    PlayerSetup('Bab', 'The Hanging Gardens of Babylon', 'B'),
    PlayerSetup('Hal', 'The Mausoleum of Halicarnassus', 'B'),
    PlayerSetup('Oly', 'The Statue of Zeus in Olympia', 'A'),
]


def test_special_powers_record_and_replay_exactly():
    content = load_content('base')
    seen = set()
    for seed in range(12):
        agents = [HeuristicAgent(), HeuristicAgent(), RandomAgent(random.Random(seed))]
        game, record = play_match(content, SPECIAL_SETUPS, agents, rng=random.Random(seed))
        seen |= {e[1] for entries in record.extras.values() for e in entries}
        seen |= {code for turns in record.moves.values() for turn in turns for code, _ in turn}
        replayed = GameRecord.from_json(record.to_json()).replay(content)
        assert [replayed.score(i).as_dict() for i in range(3)] == [game.score(i).as_dict() for i in range(3)]
    assert {'last_card', 'discard_build', 'F'} <= seen
