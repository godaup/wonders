"""Render cards to original, self-contained SVG from their data.

This deliberately does **not** use any 7 Wonders publisher artwork (which is
copyrighted). Instead it draws original, functional card layouts — a coloured
frame, the cost, and iconified effects — straight from our JSON content. The result
is license-clean (our own code + our own data), needs no external assets, and stays
in sync with the game automatically. Swapping in a richer open-licensed icon set
later only means changing the small ``_badges`` helpers.
"""

from __future__ import annotations

from html import escape
from typing import Iterable, List, Tuple

from wonders.cards.base import Card, Color, Ressource
from wonders.effects.library import (
    Coins,
    Military,
    MultiProduction,
    Production,
    Science,
    ScienceWild,
    Tally,
    Trade,
    VictoryPoints,
)

# Original palette (not the publisher's) keyed by card colour.
CARD_FILL = {
    Color.BROWN: '#8a5a2b',
    Color.GREY: '#8a8f96',
    Color.BLUE: '#2f6fb0',
    Color.YELLOW: '#e0a91b',
    Color.RED: '#b23b3b',
    Color.GREEN: '#3f9e5a',
    Color.PURPLE: '#7a4fb0',
    Color.WHITE: '#d9d9d9',
    Color.BLACK: '#333333',
}

RESOURCE_FILL = {
    Ressource.WOOD: '#6b8e23',
    Ressource.STONE: '#9aa0a6',
    Ressource.ORE: '#5b7fa6',
    Ressource.BRICKS: '#c0663a',
    Ressource.GLASS: '#5bc0c0',
    Ressource.LINEN: '#e6c78a',
    Ressource.PAPER: '#efeadf',
    Ressource.GOLD: '#e0a91b',
}
RESOURCE_ABBR = {
    Ressource.WOOD: 'Wd',
    Ressource.STONE: 'St',
    Ressource.ORE: 'Or',
    Ressource.BRICKS: 'Cl',
    Ressource.GLASS: 'Gl',
    Ressource.LINEN: 'Ln',
    Ressource.PAPER: 'Pa',
    Ressource.GOLD: '$',
}

SCIENCE_ABBR = {'tablet': 'Tab', 'gear': 'Cog', 'compass': 'Com'}

# A badge is (label, fill, text_colour).
Badge = Tuple[str, str, str]

_W, _H = 240, 320


def _text_fill(hex_color: str) -> str:
    """Black or white text depending on background luminance."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return '#111' if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else '#fff'


def _cost_badges(card: Card) -> List[Badge]:
    badges: List[Badge] = []
    if card.cost.coins:
        badges.append((f'{card.cost.coins}$', RESOURCE_FILL[Ressource.GOLD], '#111'))
    for res, count in sorted(card.cost.resources.items(), key=lambda kv: kv[0].value):
        fill = RESOURCE_FILL[res]
        for _ in range(count):
            badges.append((RESOURCE_ABBR[res], fill, _text_fill(fill)))
    return badges


def _resource_options(slot: Iterable[Ressource]) -> str:
    return '/'.join(RESOURCE_ABBR[r] for r in sorted(slot, key=lambda r: r.value))


def _effect_badges(effect) -> List[Badge]:
    if isinstance(effect, Production):
        return [(_resource_options(effect.options), '#efeadf', '#111')]
    if isinstance(effect, MultiProduction):
        return [(_resource_options(slot), '#efeadf', '#111') for slot in effect.slots]
    if isinstance(effect, Military):
        return [(f'{effect.shields_count} shield', '#b23b3b', '#fff')]
    if isinstance(effect, Science):
        return [(SCIENCE_ABBR[effect.symbol.value], '#3f9e5a', '#fff')]
    if isinstance(effect, ScienceWild):
        return [('any sci', '#3f9e5a', '#fff')]
    if isinstance(effect, VictoryPoints):
        return [(f'{effect.points} VP', '#2f6fb0', '#fff')]
    if isinstance(effect, Coins):
        return [(f'+{effect.amount}$', '#e0a91b', '#111')]
    if isinstance(effect, Trade):
        return [(f'buy {effect.category}/{effect.side} @{effect.price}', '#e0a91b', '#111')]
    if isinstance(effect, Tally):
        target = ('/'.join(sorted(c.value for c in effect.colors)) or effect.what).replace('_', ' ')
        bits = []
        if effect.vp_each:
            bits.append(f'{effect.vp_each}VP')
        if effect.coins_each:
            bits.append(f'{effect.coins_each}$')
        return [(f'{"+".join(bits)}/{target} ({effect.scope})', '#7a4fb0', '#fff')]
    return [(type(effect).__name__, '#777', '#fff')]


def _chip(x: float, y: float, label: str, fill: str, text_fill: str) -> Tuple[str, float]:
    w = 12 + 6.2 * len(label)
    svg = (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="20" rx="6" fill="{fill}"/>'
        f'<text x="{x + w / 2:.1f}" y="{y + 14:.1f}" text-anchor="middle" '
        f'font-size="11" fill="{text_fill}">{escape(label)}</text>'
    )
    return svg, w


def _flow(badges: List[Badge], x0: float, y0: float, max_x: float) -> str:
    out, x, y = [], x0, y0
    for label, fill, tf in badges:
        chip, w = _chip(x, y, label, fill, tf)
        if x + w > max_x and x > x0:
            x, y = x0, y + 24
            chip, w = _chip(x, y, label, fill, tf)
        out.append(chip)
        x += w + 5
    return ''.join(out)


def card_svg(card: Card) -> str:
    """A standalone ``<svg>`` element for one card."""
    fill = CARD_FILL[card.color]
    name = card.name if len(card.name) <= 24 else card.name[:22] + '…'
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_W} {_H}" '
        f'width="{_W}" height="{_H}" font-family="sans-serif">',
        f'<rect x="4" y="4" width="{_W - 8}" height="{_H - 8}" rx="14" '
        f'fill="#fff" stroke="{fill}" stroke-width="6"/>',
        f'<rect x="12" y="12" width="{_W - 24}" height="40" rx="8" fill="{fill}"/>',
        f'<text x="{_W / 2:.0f}" y="38" text-anchor="middle" font-size="15" '
        f'font-weight="bold" fill="{_text_fill(fill)}">{escape(name)}</text>',
        '<text x="16" y="76" font-size="11" fill="#666">Cost</text>',
    ]
    cost = _cost_badges(card)
    parts.append(
        _flow(cost, 16, 82, _W - 16) if cost else '<text x="16" y="96" font-size="11" fill="#3f9e5a">free</text>'
    )
    parts.append('<text x="16" y="132" font-size="11" fill="#666">Effects</text>')
    effect_badges: List[Badge] = []
    for effect in card.effects:
        effect_badges += _effect_badges(effect)
    parts.append(_flow(effect_badges, 16, 138, _W - 16))
    if card.chain_requires:
        parts.append(
            f'<text x="16" y="250" font-size="10" fill="#7a4fb0">chain: {escape(card.chain_requires)} → free</text>'
        )
    if card.chain_enables:
        parts.append(
            '<text x="16" y="264" font-size="10" fill="#7a4fb0">'
            f'unlocks: {escape(", ".join(card.chain_enables))}</text>'
        )
    meta = f'Age {card.age or "?"} · {card.expansion}'
    if card.at_players:
        meta += ' · players ' + ','.join(str(p) for p in card.at_players)
    if card.guild:
        meta += ' · guild'
    parts.append(
        f'<text x="{_W / 2:.0f}" y="{_H - 14}" text-anchor="middle" '
        f'font-size="10" fill="#888">{escape(meta)}</text>'
    )
    parts.append('</svg>')
    return ''.join(parts)


def gallery_html(cards: Iterable[Card], title: str = 'Cards') -> str:
    """A full HTML page laying every card out in a responsive grid."""
    cells = ''.join(f'<div class="cell">{card_svg(c)}</div>' for c in cards)
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        f'<title>{escape(title)}</title><style>'
        'body{background:#f4f2ee;font-family:sans-serif;margin:24px}'
        'h1{font-size:20px}'
        '.grid{display:flex;flex-wrap:wrap;gap:16px}'
        '.cell{background:#fff;border-radius:14px;box-shadow:0 1px 4px rgba(0,0,0,.15)}'
        f'</style></head><body><h1>{escape(title)}</h1><div class="grid">{cells}</div>'
        '</body></html>'
    )
