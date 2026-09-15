"""Tests for the SVG card and wonder renderer."""

from wonders.cards.content import load_content
from wonders.viz.render import card_svg, gallery_html, wonder_svg


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
