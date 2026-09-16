"""Tests for JSON content loading and deck building."""

import random

import pytest

from wonders.cards.base import Color
from wonders.cards.base import Ressource as R
from wonders.cards.base import ScienceSymbol as S
from wonders.cards.content import build_age_deck, build_effect, deal_hands, load_content
from wonders.effects.library import Production, Science, Tally


def test_load_base_content():
    content = load_content('base')
    assert content.card('Lumber Yard').color is Color.BROWN
    assert content.wonder('The Pyramids of Giza').starting_resource is R.STONE
    assert len(content.guilds()) == 10
    assert len(content.wonders) == 14  # 7 wonders x A/B sides
    assert content.card('Palace').effects[0].points == 8


def test_wonder_sides_are_distinguishable():
    content = load_content('base')
    a = content.wonder('The Colossus of Rhodes', 'A')
    b = content.wonder('The Colossus of Rhodes', 'B')
    assert len(a.stages) == 3 and len(b.stages) == 2


@pytest.mark.parametrize('n', [3, 4, 5, 6, 7])
def test_age_decks_total_seven_times_players(n):
    # The defining invariant of a correct base deck: every Age deals 7 cards each.
    content = load_content('base')
    for age in (1, 2, 3):
        deck = build_age_deck(content, age, n, rng=random.Random(n))
        assert len(deck) == 7 * n


def test_card_effects_parsed_from_data():
    content = load_content('base')
    lumber = content.card('Lumber Yard')
    (effect,) = lumber.effects
    assert isinstance(effect, Production)
    assert effect.production() == (frozenset({R.WOOD}),)

    scriptorium = content.card('Scriptorium')
    assert isinstance(scriptorium.effects[0], Science)
    assert scriptorium.effects[0].science() is S.TABLET


def test_effect_registry_round_trip():
    tally = build_effect({'type': 'tally', 'what': 'colors', 'scope': 'neighbours', 'colors': ['brown'], 'vp_each': 1})
    assert isinstance(tally, Tally)
    assert tally.colors == frozenset({Color.BROWN})


def test_copies_scale_with_player_count():
    content = load_content('base')
    # Stone Pit is stamped at_players [3, 5]: 1 copy at 3-4 players, 2 at 5+.
    stone_pit = content.card('Stone Pit')
    assert stone_pit.copies_for(3) == 1
    assert stone_pit.copies_for(4) == 1
    assert stone_pit.copies_for(5) == 2

    deck3 = build_age_deck(content, age=1, n_players=3, rng=random.Random(0))
    deck5 = build_age_deck(content, age=1, n_players=5, rng=random.Random(0))
    assert deck5.count(stone_pit) == 2
    assert deck3.count(stone_pit) == 1
    assert len(deck5) > len(deck3)  # more players -> more copies


def test_age_three_draws_players_plus_two_guilds():
    content = load_content('base')
    deck = build_age_deck(content, age=3, n_players=3, rng=random.Random(1))
    guilds_in_deck = [c for c in deck if c.guild]
    # 3 players + 2 = 5 guilds requested; base has exactly 5, so all appear once.
    assert len(guilds_in_deck) == 5

    # With fewer requested than available it should sample without duplicates.
    content2 = load_content('base')
    small = build_age_deck(content2, age=3, n_players=1, rng=random.Random(2))
    picked = [c for c in small if c.guild]
    assert len(picked) == 3  # 1 + 2
    assert len(set(id(c) for c in picked)) == 3


def test_deck_building_is_reproducible_with_seed():
    content = load_content('base')
    a = build_age_deck(content, age=3, n_players=4, rng=random.Random(42))
    b = build_age_deck(content, age=3, n_players=4, rng=random.Random(42))
    assert [c.name for c in a] == [c.name for c in b]


def test_deal_hands_splits_evenly():
    content = load_content('base')
    deck = build_age_deck(content, age=1, n_players=3, rng=random.Random(0))
    hands = deal_hands(deck, 3)
    assert len(hands) == 3
    assert len({len(h) for h in hands}) == 1  # equal sizes
