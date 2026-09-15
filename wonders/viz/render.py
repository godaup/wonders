"""Render cards to original, self-contained SVG from their data.

This deliberately uses **no** 7 Wonders publisher artwork (which is copyrighted).
Every glyph is an original inline line-icon authored here, and every card is drawn
from our JSON content — so the result is license-clean (our own code + our own
data), needs no external assets or attribution, and stays in sync with the game.

Layout per card: a category-coloured header ribbon, a cost bar, an effects panel of
icon+label rows, an optional chaining line, and a metadata footer.
"""

from __future__ import annotations

import re
from html import escape
from typing import List, Tuple

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

_W, _H = 250, 350

# --- palette -------------------------------------------------------------
CARD_FILL = {
    Color.BROWN: '#8a5a2b',
    Color.GREY: '#7c828a',
    Color.BLUE: '#2f6fb0',
    Color.YELLOW: '#d99a1c',
    Color.RED: '#b23b3b',
    Color.GREEN: '#3f9e5a',
    Color.PURPLE: '#7a4fb0',
    Color.WHITE: '#c8ccd2',
    Color.BLACK: '#2b2f36',
}
# Icon tints (darkened so light resources still read on white).
RES_COLOR = {
    Ressource.WOOD: '#5f7d22',
    Ressource.STONE: '#6b7278',
    Ressource.ORE: '#3f6187',
    Ressource.BRICKS: '#a8552f',
    Ressource.GLASS: '#2e9d9d',
    Ressource.LINEN: '#b0863a',
    Ressource.PAPER: '#8f8560',
    Ressource.GOLD: '#c8951a',
}
RES_ICON = {
    Ressource.WOOD: 'wood',
    Ressource.STONE: 'stone',
    Ressource.ORE: 'ore',
    Ressource.BRICKS: 'brick',
    Ressource.GLASS: 'glass',
    Ressource.LINEN: 'loom',
    Ressource.PAPER: 'scroll',
    Ressource.GOLD: 'coin',
}
RES_NAME = {
    Ressource.WOOD: 'Wood',
    Ressource.STONE: 'Stone',
    Ressource.ORE: 'Ore',
    Ressource.BRICKS: 'Clay',
    Ressource.GLASS: 'Glass',
    Ressource.LINEN: 'Loom',
    Ressource.PAPER: 'Papyrus',
    Ressource.GOLD: 'Coins',
}
SCI_NAME = {'tablet': 'Tablet', 'gear': 'Gear', 'compass': 'Compass'}
SCIENCE = '#2f7d46'
MILITARY = '#b23b3b'
CIVIC = '#2f6fb0'
GOLD = '#c8951a'
GUILD = '#6b46a8'
INK = '#233043'
MUTED = '#6b7684'
SCOPE = {'self': 'self', 'neighbours': 'neighbors', 'both': 'all cities'}

# --- icon set (original 24x24 line icons, currentColor) ------------------
ICONS = {
    'wood': '<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">'
    '<rect x="3" y="7" width="18" height="4.6" rx="2.3"/><rect x="6" y="13" width="15" height="4.6" rx="2.3"/>'
    '<circle cx="5.5" cy="9.3" r=".8"/><circle cx="8.5" cy="15.3" r=".8"/></g>',
    'stone': '<g fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round">'
    '<path d="M4 9 8.5 5 20 7 19 16 8 19 4 14Z"/><path d="M8.5 5 9.5 11.5 19 16M9.5 11.5 4 14"/></g>',
    'ore': '<g fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round">'
    '<path d="M6 4h12l4 5-10 12L2 9Z"/><path d="M2 9h20M9 4 7 9l5 12M15 4l2 5"/></g>',
    'brick': '<g fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round">'
    '<rect x="3" y="6" width="18" height="12" rx="1.5"/><path d="M3 12h18M12 6v6M7.5 12v6M16.5 12v6"/></g>',
    'glass': '<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M7 4h10c0 6-3 8-5 8s-5-2-5-8Z"/><path d="M12 12v6M8 20h8"/></g>',
    'loom': '<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M6 4h12M6 20h12"/><rect x="9" y="4" width="6" height="16" rx="1"/>'
    '<path d="M9 8h6M9 12h6M9 16h6"/></g>',
    'scroll': '<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M7 4h12v13a2 2 0 0 1-2 2H7"/><path d="M7 4a2 2 0 0 0 0 4h2"/>'
    '<path d="M10 8h6M10 12h6"/></g>',
    'gear': '<g fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="3.6"/>'
    '<path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2.2 2.2M16.8 16.8 19 19M19 5l-2.2 2.2M7.2 16.8 5 19"/></g>',
    'tablet': '<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M8 8h8M8 12h8M8 16h5"/></g>',
    'compass': '<g fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="9"/><path d="M12 12 16 8 13 13Z" fill="currentColor"/>'
    '<path d="M12 12 8 16 11 11Z"/></g>',
    'shield': '<g fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round">'
    '<path d="M12 3 20 6v5c0 5-4 8.5-8 10-4-1.5-8-5-8-10V6Z"/></g>',
    'coin': '<g fill="none" stroke="currentColor" stroke-width="2">'
    '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4.5"/></g>',
    'vp': '<path fill="currentColor" d="M12 3 14.6 9 21 9.5 16.1 13.8 17.6 20 12 16.7 6.4 20 '
    '7.9 13.8 3 9.5 9.4 9Z"/>',
    'chain': '<g fill="none" stroke="currentColor" stroke-width="2">'
    '<rect x="3" y="8" width="11" height="8" rx="4"/><rect x="10" y="8" width="11" height="8" rx="4"/></g>',
    'trade': '<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M4 9h14M15 6l3 3-3 3"/><path d="M20 15H6M9 12l-3 3 3 3"/></g>',
}

# A row = (list of (icon_key, color), label).
Row = Tuple[List[Tuple[str, str]], str]


def _slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', name.lower()) or 'card'


def _icon(key: str, x: float, y: float, size: float, color: str, title: str = '') -> str:
    scale = size / 24.0
    tip = f'<title>{escape(title)}</title>' if title else ''
    return (
        f'<g transform="translate({x:.1f},{y:.1f}) scale({scale:.3f})" ' f'style="color:{color}">{tip}{ICONS[key]}</g>'
    )


def _wrap(name: str, limit: int = 20) -> List[str]:
    words, lines, cur = name.split(), [], ''
    for w in words:
        if cur and len(cur) + 1 + len(w) > limit:
            lines.append(cur)
            cur = w
        else:
            cur = f'{cur} {w}'.strip()
        if len(lines) == 2:
            break
    if cur and len(lines) < 2:
        lines.append(cur)
    if len(lines) == 2 and (len(''.join(words)) > limit * 2):
        lines[1] = lines[1][: limit - 1].rstrip() + '…'
    return lines[:2] or ['']


def _resource_rows(effect) -> List[Row]:
    rows: List[Row] = []
    slots = effect.slots if isinstance(effect, MultiProduction) else [effect.options]
    for opts in slots:
        res = sorted(opts, key=lambda r: r.value)
        icons = [(RES_ICON[r], RES_COLOR[r]) for r in res]
        label = ' / '.join(RES_NAME[r] for r in res)
        rows.append((icons, label))
    return rows


def _describe(effect) -> List[Row]:
    if isinstance(effect, (Production, MultiProduction)):
        return _resource_rows(effect)
    if isinstance(effect, Military):
        n = effect.shields_count
        return [([('shield', MILITARY)], f'{n} Shield' + ('s' if n != 1 else ''))]
    if isinstance(effect, Science):
        sym = effect.symbol.value
        return [([(sym, SCIENCE)], SCI_NAME[sym])]
    if isinstance(effect, ScienceWild):
        return [([('tablet', SCIENCE), ('gear', SCIENCE), ('compass', SCIENCE)], 'Any science')]
    if isinstance(effect, VictoryPoints):
        return [([('vp', CIVIC)], f'{effect.points} Victory pts')]
    if isinstance(effect, Coins):
        return [([('coin', GOLD)], f'+{effect.amount} Coins')]
    if isinstance(effect, Trade):
        return [([('trade', GOLD)], f'Buy {effect.category} ({effect.side}) {effect.price}c')]
    if isinstance(effect, Tally):
        target = '/'.join(sorted(c.value for c in effect.colors)) or effect.what.replace('_', ' ')
        parts = []
        if effect.vp_each:
            parts.append(f'{effect.vp_each} VP')
        if effect.coins_each:
            parts.append(f'{effect.coins_each}c')
        icon = ('vp', GUILD) if effect.vp_each else ('coin', GUILD)
        return [([icon], f'{" +".join(parts)} / {target} ({SCOPE.get(effect.scope, effect.scope)})')]
    return [([('coin', MUTED)], type(effect).__name__)]


def _row_svg(icons: List[Tuple[str, str]], label: str, x: float, y: float, size: float = 19) -> str:
    out = []
    ix = x
    for key, color in icons[:3]:
        out.append(_icon(key, ix, y - size + 2, size, color))
        ix += size + 3
    tx = ix + 4
    out.append(f'<text x="{tx:.1f}" y="{y - 3:.1f}" font-size="12.5" fill="{INK}">{escape(label)}</text>')
    return ''.join(out)


def card_svg(card: Card) -> str:
    """A standalone, self-contained ``<svg>`` element for one card."""
    uid = _slug(card.name)
    fill = CARD_FILL[card.color]
    lines = _wrap(card.name)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_W} {_H}" width="{_W}" height="{_H}" '
        f'font-family="ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif" role="img">',
        f'<title>{escape(card.name)}</title>',
        f'<defs><clipPath id="clip-{uid}"><rect x="3" y="3" width="{_W - 6}" height="{_H - 6}" rx="16"/>'
        f'</clipPath>'
        f'<filter id="sh-{uid}" x="-10%" y="-10%" width="120%" height="120%">'
        f'<feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#233043" flood-opacity="0.18"/>'
        f'</filter></defs>',
        f'<rect x="3" y="3" width="{_W - 6}" height="{_H - 6}" rx="16" fill="#fff" '
        f'stroke="#e2e6ea" stroke-width="1.5" filter="url(#sh-{uid})"/>',
        f'<g clip-path="url(#clip-{uid})">',
        f'<rect x="3" y="3" width="{_W - 6}" height="56" fill="{fill}"/>',
    ]
    # header name (1-2 lines) + age badge
    if len(lines) == 1:
        parts.append(f'<text x="18" y="37" font-size="17" font-weight="700" fill="#fff">{escape(lines[0])}</text>')
    else:
        parts.append(
            f'<text x="18" y="27" font-size="15.5" font-weight="700" fill="#fff">{escape(lines[0])}</text>'
            f'<text x="18" y="45" font-size="15.5" font-weight="700" fill="#fff">{escape(lines[1])}</text>'
        )
    roman = {1: 'I', 2: 'II', 3: 'III'}.get(card.age or 0, '')
    if roman:
        parts.append(
            f'<rect x="{_W - 44}" y="14" width="30" height="28" rx="7" fill="#ffffff" fill-opacity="0.22"/>'
            f'<text x="{_W - 29}" y="33" text-anchor="middle" font-size="13" font-weight="700" '
            f'fill="#fff">{escape(roman)}</text>'
        )
    parts.append('</g>')

    # cost bar
    parts.append(f'<text x="18" y="76" font-size="10" letter-spacing="1.2" fill="{MUTED}">COST</text>')
    cost = list(card.cost.resources.items())
    if card.cost.coins:
        parts.append(_icon('coin', 18, 82, 22, GOLD, 'Coins'))
        parts.append(f'<text x="42" y="99" font-size="12.5" fill="{INK}">×{card.cost.coins}</text>')
        cx = 66.0
    else:
        cx = 18.0
    if not card.cost.coins and not cost:
        parts.append(f'<text x="18" y="99" font-size="12.5" fill="{SCIENCE}">Free</text>')
    for res, n in sorted(cost, key=lambda kv: kv[0].value):
        parts.append(_icon(RES_ICON[res], cx, 82, 22, RES_COLOR[res], RES_NAME[res]))
        if n > 1:
            parts.append(f'<text x="{cx + 22:.0f}" y="99" font-size="11.5" fill="{INK}">×{n}</text>')
            cx += 40
        else:
            cx += 28

    # effects panel
    rows: List[Row] = []
    for effect in card.effects:
        rows += _describe(effect)
    parts.append(
        f'<rect x="12" y="112" width="{_W - 24}" height="{_H - 112 - 42}" rx="12" '
        f'fill="{fill}" fill-opacity="0.07"/>'
        f'<rect x="12" y="112" width="4" height="{_H - 112 - 42}" rx="2" fill="{fill}"/>'
        f'<text x="26" y="132" font-size="10" letter-spacing="1.2" fill="{MUTED}">EFFECTS</text>'
    )
    y = 156
    for icons, label in rows:
        parts.append(_row_svg(icons, label, 26, y))
        y += 28
    if card.chain_requires:
        parts.append(_icon('chain', 24, y - 18, 18, GUILD))
        parts.append(
            f'<text x="47" y="{y - 4}" font-size="11.5" fill="{GUILD}">' f'free if {escape(card.chain_requires)}</text>'
        )
        y += 24
    elif card.chain_enables:
        parts.append(_icon('chain', 24, y - 18, 18, MUTED))
        parts.append(
            f'<text x="47" y="{y - 4}" font-size="11" fill="{MUTED}">' f'unlocks {escape(card.chain_enables[0])}</text>'
        )

    # footer
    meta = []
    if card.at_players:
        meta.append(','.join(str(p) for p in card.at_players) + '+ players')
    meta.append(card.expansion)
    if card.guild:
        meta.append('guild')
    parts.append(
        f'<text x="{_W / 2:.0f}" y="{_H - 16}" text-anchor="middle" font-size="10.5" '
        f'fill="{MUTED}">{escape(" · ".join(meta))}</text>'
    )
    parts.append('</svg>')
    return ''.join(parts)


def gallery_html(cards, title: str = 'Cards') -> str:
    """A full HTML page laying every card out in a responsive grid."""
    cells = ''.join(f'<div class="cell">{card_svg(c)}</div>' for c in cards)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{escape(title)}</title><style>'
        'body{background:#eceff3;margin:0;padding:32px 24px 56px;'
        'font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif;color:#233043}'
        'h1{font-size:20px;font-weight:650;margin:0 auto 24px;max-width:1160px}'
        '.grid{display:flex;flex-wrap:wrap;gap:20px;justify-content:center;max-width:1160px;margin:0 auto}'
        '.cell svg{display:block}'
        f'</style></head><body><h1>{escape(title)}</h1><div class="grid">{cells}</div></body></html>'
    )
