"""Tests for deck sizes: every Age deck must split evenly into the hands, with any bundled expansions."""

import random
from typing import Any, Dict

import pytest

from tests.test_builder import _project
from tests.test_extensions import isolated  # noqa: F401 - fixture
from wonders.cards.base import Card
from wonders.cards.content import build_age_deck, build_card, load_content
from wonders.cards.decks import check_decks, deck_size, deck_table, extra_guilds_with, summary
from wonders.play import builder

ONE_PER_PLAYER = [3, 3, 3, 4, 5, 6, 7]


def _card(age: int = 1, at_players: Any = ONE_PER_PLAYER, **extra: Any) -> Card:
    spec: Dict[str, Any] = {'name': f'New {age}', 'color': 'blue', 'age': age, 'at_players': at_players}
    spec['effects'] = [{'type': 'victory_points', 'points': 1}]
    return build_card({**spec, **extra}, 'test')


@pytest.mark.parametrize('combo', [('base',), ('base', 'leaders'), ('base', 'cities'), ('base', 'leaders', 'cities')])
def test_counted_sizes_match_the_dealt_decks(combo):
    content = load_content(*combo)
    for extra_guilds in (0, 2, 5):
        for age in (1, 2, 3):
            for n in range(3, 8):
                dealt = build_age_deck(content, age, n, random.Random(0), extra_guilds)
                assert deck_size(content.cards, age, n, extra_guilds).cards == len(dealt)


def test_the_official_decks_deal_evenly():
    assert check_decks([]) == []


def test_one_card_per_player_changes_the_hands_without_leftovers():
    problems = check_decks([_card()])
    assert problems and all(p.level == 'warning' and p.age == 1 for p in problems)
    assert 'hands of 8 instead of 7' in (summary(problems, 'warning') or '')


def test_cards_left_over_are_errors_in_every_combination():
    problems = check_decks([_card(at_players=[3, 5])])
    errors = [p for p in problems if p.level == 'error']
    assert {p.expansions for p in errors} == {(), ('leaders',), ('cities',), ('leaders', 'cities')}
    text = summary(problems, 'error') or ''
    assert text.startswith('cards would be left over') and 'Age I at 3 players (every combination)' in text


def test_extra_guilds_must_be_balanced_by_age_three_cards():
    assert extra_guilds_with(('base',), {'add': 1}) == 3 and extra_guilds_with(('base',), 0) == 0
    assert extra_guilds_with(('base',), {'add': 9}) == 10  # clamped to the rule's range
    assert extra_guilds_with(('base',), {'add': 99}) == 2  # an invalid rule (reported by the rules check) is ignored
    errors = [p for p in check_decks([], {'add': 1}) if p.level == 'error']
    assert {(p.age, p.size.players) for p in errors} == {(3, n) for n in range(3, 8)}
    # One guild fewer per game and one more Age III card at every count: deals evenly again.
    assert check_decks([_card(3, at_players=[3])], {'add': -1}) == []


def test_a_small_guild_pool_is_reported():
    problems = check_decks([], 6)  # 13 guilds wanted at 7 players; the base pool has 10, Leaders 14
    short = [p for p in problems if p.size.short_of_guilds]
    assert short and 'the pool has 10' in short[0].detail()
    table = deck_table([], 6)
    assert table[3][7]['guilds_wanted'] == 13 and table[3][7]['guild_pool'] == 10


def test_builder_pins_deck_errors_to_cards_and_rules(isolated):  # noqa: F811
    card = dict(_project()['cards'][0], at_players=[3])
    report = builder.check_project(_project(cards=[card], rules={'extra_guilds': {'add': 1}}))
    errors = {(p['section'], p['index']): p['message'] for p in report['problems'] if p['level'] == 'error'}
    assert 'Age I at 4 players' in errors[('cards', None)]
    assert 'Age III at 3 players' in errors[('rules', None)]
    assert not report['ok'] and report['decks'][1][4] == {
        'cards': 29,
        'hand': 7,
        'leftover': 1,
        'base': 28,
        'guilds_wanted': 0,
        'guild_pool': 0,  # guilds are drawn in Age III only
    }
    result = builder.save_project(_project(cards=[card]))
    assert not result['check']['loaded']  # an expansion that leaves cards over is not played
