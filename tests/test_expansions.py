"""Tests for the optional, combinable expansion framework and its mechanics."""

import random

import pytest

from wonders.agents.baselines import GreedyAgent, RandomAgent
from wonders.agents.match import play_match
from wonders.cards.base import Card, Color
from wonders.cards.content import build_age_deck, load_content
from wonders.effects.library import Debt, GainToken, Grant, Military, Plunder, Tally
from wonders.engine.actions import Action
from wonders.engine.game import Game
from wonders.engine.player import Player
from wonders.engine.record import GameRecord, PlayerSetup
from wonders.ruleset import Ruleset

SETUPS = [
    PlayerSetup('Alice', 'The Pyramids of Giza'),
    PlayerSetup('Bob', 'The Colossus of Rhodes'),
    PlayerSetup('Cara', 'The Temple of Artemis in Ephesus'),
]


def _game(n=3):
    content = load_content('base')
    return Game([Player(s.name, content.wonder(s.wonder, s.side)) for s in SETUPS[:n]])


# -- ruleset composition --------------------------------------------------
def test_ruleset_always_includes_base_and_dedupes():
    assert Ruleset.of().expansions == ('base',)
    assert Ruleset.of('leaders', 'cities').expansions == ('base', 'leaders', 'cities')
    assert Ruleset.of('cities', 'cities').expansions == ('base', 'cities')


def test_starting_coins_compose():
    assert Ruleset.of().starting_coins == 3
    assert Ruleset.of('cities').starting_coins == 3
    assert Ruleset.of('leaders').starting_coins == 6
    assert Ruleset.of('leaders', 'cities').starting_coins == 6  # max wins


def test_unimplemented_and_unknown():
    assert Ruleset.of('armada', 'babel').unimplemented == ('armada', 'babel')
    assert Ruleset.of('leaders').unimplemented == ()
    assert Ruleset.of('leaders').uses('leaders') and not Ruleset.of('leaders').uses('cities')
    with pytest.raises(ValueError):
        Ruleset.of('bogus')


# -- content combination --------------------------------------------------
def test_content_merges_expansions():
    content = load_content('base', 'leaders', 'cities')
    assert len(content.leaders()) == 36  # official leader set (white cards)
    assert sum(1 for c in content.cards if c.color is Color.BLACK) == 42  # Cities black cards


def test_leaders_excluded_but_cities_included_in_age_decks():
    content = load_content('base', 'leaders', 'cities')
    deck = build_age_deck(content, 1, 3, rng=random.Random(0))
    assert not any(c.color is Color.WHITE for c in deck)  # leaders drafted separately
    assert sum(1 for c in deck if c.color is Color.BLACK) == 3  # n_players black cards dealt in
    assert len(deck) == 8 * 3  # base 7*n + n black = one extra card per player


def test_base_only_deck_unaffected_by_expansion_files():
    # Loading only base must still validate to 7*players (no leaked black/white cards).
    content = load_content('base')
    for n in (3, 5, 7):
        assert len(build_age_deck(content, 1, n, rng=random.Random(n))) == 7 * n


# -- cities mechanics -----------------------------------------------------
def test_debt_imposed_on_neighbours_reduces_score():
    g = _game()
    g.players[1].coins = g.players[2].coins = 9  # 3 treasury each before debt
    g.apply(0, Action.build(Card('Raiders', Color.BLACK, effects=(Debt(2, 'neighbours'),))))
    assert g.players[1].debt == 2 and g.players[2].debt == 2
    assert g.score(1).treasury == 9 // 3 - 2  # 1


def test_diplomacy_token_sits_out_military():
    g = _game()
    g.players[0].add_card(Card('Fort', Color.RED, effects=(Military(2),)))
    g.players[1].diplomacy = 1
    g.resolve_conflict(age=1)
    assert g.players[1].tokens == [] and g.players[1].diplomacy == 0  # sat out, token spent
    # with P1 removed, P0 and P2 are adjacent: P0 (2 shields) beats P2 (0 shields)
    assert sum(g.players[0].tokens) == 1
    assert sum(g.players[2].tokens) == -1


def test_plunder_takes_coins_and_overflows_to_debt():
    g = _game()
    g.players[1].coins = 1
    g.players[2].coins = 5
    g.apply(0, Action.build(Card('Raiders', Color.BLACK, effects=(Plunder(3, 'neighbours'),))))
    assert g.players[1].coins == 0 and g.players[1].debt == 2  # only had 1; 2 unpaid -> debt
    assert g.players[2].coins == 2 and g.players[2].debt == 0  # paid all 3


def test_grant_gives_neighbours_coins():
    g = _game()
    before = [g.players[1].coins, g.players[2].coins]
    g.apply(0, Action.build(Card('Largesse', Color.BLACK, effects=(Grant(2, 'neighbours'),))))
    assert g.players[1].coins == before[0] + 2 and g.players[2].coins == before[1] + 2


def test_gain_token_adds_a_military_token():
    g = _game()
    g.apply(0, Action.build(Card('Raid', Color.BLACK, effects=(GainToken(3),))))
    assert g.players[0].tokens == [3]


def test_tally_scores_per_three_coins():
    g = _game()
    g.players[0].coins = 9
    g.players[0].add_card(Card('Midas', Color.WHITE, effects=(Tally('coins', 'self', vp_each=1),)))
    assert g.score(0).commercial == 3  # 9 // 3, on top of the treasury's own 3


# -- combined game end to end ---------------------------------------------
def test_combined_expansion_game_runs_and_replays():
    expansions = ('base', 'leaders', 'cities')
    content = load_content(*expansions)
    agents = [GreedyAgent(), RandomAgent(random.Random(1)), GreedyAgent()]
    game, record = play_match(content, SETUPS, agents, rng=random.Random(0), expansions=expansions)
    # replay reproduces scores — implicitly checks starting coins (6) are threaded
    replay = GameRecord.from_json(record.to_json()).replay(content)
    assert [replay.score(i).total for i in range(3)] == [game.score(i).total for i in range(3)]
    assert record.expansions == expansions
