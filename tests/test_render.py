"""Tests for the SVG card, wonder, and table renderer."""

from wonders.cards.content import load_content
from wonders.engine.game import Game
from wonders.engine.player import Player
from wonders.viz.render import card_svg, gallery_html, wonder_svg
from wonders.viz.table import table_html, table_svg


def _table_game():
    content = load_content('base')
    game = Game(
        [
            Player(n, content.wonder(w))
            for n, w in [
                ('Ada', 'The Pyramids of Giza'),
                ('Bo', 'The Colossus of Rhodes'),
                ('Cy', 'The Temple of Artemis in Ephesus'),
            ]
        ]
    )
    game.age = 2
    game.players[0].add_card(content.card('Baths'))
    game.players[0].add_card(content.card('Scriptorium'))
    game.players[1].add_card(content.card('Stockade'))
    game.players[1].tokens.append(1)
    return game


def test_card_svg_contains_name_and_effects():
    content = load_content('base')
    svg = card_svg(content.card('Scriptorium'))
    assert svg.startswith('<svg') and svg.rstrip().endswith('</svg>')
    assert 'Scriptorium' in svg
    assert 'Tablet' in svg  # science tablet effect label


def test_card_svg_shows_cost_and_free():
    content = load_content('base')
    assert 'Free' in card_svg(content.card('Lumber Yard'))  # no cost
    assert 'Stone' in card_svg(content.card('Baths'))  # costs one stone (icon title)


def test_every_base_card_renders():
    content = load_content('base')
    for card in content.cards:
        svg = card_svg(card)
        assert svg.count('<svg') == 1


def test_special_characters_are_escaped():
    from wonders.cards.base import Card, Color

    svg = card_svg(Card('A & B <x>', Color.BLUE))
    assert '&amp;' in svg and '&lt;' in svg
    assert '<x>' not in svg  # raw angle brackets from the name must be escaped


def test_gallery_html_wraps_cards():
    content = load_content('base')
    html = gallery_html(content.cards[:5], title='Sample')
    assert html.startswith('<!doctype html>')
    assert html.count('<svg') == 5
    assert 'Sample' in html


def test_gallery_includes_wonders_when_given():
    content = load_content('base')
    html = gallery_html(content.cards[:3], content.wonders, title='All')
    assert html.count('<svg') == 3 + len(content.wonders)
    assert 'Wonder boards' in html


def test_wonder_svg_shows_stages_and_starting_resource():
    content = load_content('base')
    svg = wonder_svg(content.wonder('The Pyramids of Giza'))
    assert svg.startswith('<svg') and svg.rstrip().endswith('</svg>')
    assert 'STAGE 1' in svg and 'STAGE 3' in svg
    assert 'Stone' in svg  # starting resource + stage costs
    assert 'Victory pts' in svg  # stage effect labels


def test_every_wonder_renders():
    content = load_content('base')
    for wonder in content.wonders:
        assert wonder_svg(wonder).count('<svg') == 1


def test_table_svg_shows_every_player_and_age():
    game = _table_game()
    svg = table_svg(game)
    assert svg.startswith('<svg') and svg.rstrip().endswith('</svg>')
    for name in ('Ada', 'Bo', 'Cy'):
        assert name in svg
    assert 'Age 2' in svg
    assert 'leader:' in svg  # the leader is highlighted/labelled


def test_table_html_is_self_contained():
    html = table_html(_table_game(), title='Mid-game')
    assert html.startswith('<!doctype html>')
    assert html.count('<svg') == 1 and 'Mid-game' in html


def test_long_effect_text_wraps_inside_the_card():
    import re

    from wonders.cards.content import load_content

    for card in load_content('base', 'leaders', 'cities').cards:
        svg = card_svg(card)
        assert '…' not in svg, card.name
        for x, line in re.findall(r'<text x="([\d.]+)" y="[\d.]+" font-size="15.0" fill="[^"]+">([^<]*)<', svg):
            assert float(x) + 0.53 * 15 * len(line) <= 250, (card.name, line)  # stays inside the card
