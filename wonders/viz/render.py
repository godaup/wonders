"""Render cards and wonder boards to original, self-contained SVG from their data.

This uses **no** 7 Wonders publisher artwork (which is copyrighted). Every glyph is
an original inline line-icon authored here and every layout is our own, drawn from
our JSON content — license-clean (our code + our data), no external assets or
attribution, always in sync with the game.
"""

from __future__ import annotations

import re
from html import escape
from typing import Iterable, List, Sequence, Tuple

from wonders.cards.base import Card, Color, Ressource
from wonders.cards.wonder import Wonder, WonderStage
from wonders.effects.library import (
    Coins,
    Debt,
    Diplomacy,
    GainToken,
    Grant,
    Military,
    MultiProduction,
    Plunder,
    Production,
    Science,
    ScienceWild,
    Special,
    Tally,
    Trade,
    VictoryPoints,
)

_W, _H = 250, 352

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
CAT_NAME = {
    Color.BROWN: 'Raw material',
    Color.GREY: 'Manufactured good',
    Color.BLUE: 'Civilian',
    Color.YELLOW: 'Commercial',
    Color.RED: 'Military',
    Color.GREEN: 'Scientific',
    Color.PURPLE: 'Guild',
    Color.WHITE: 'Leader',
    Color.BLACK: 'City',
}
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
SCIENCE, MILITARY, CIVIC, GOLD, GUILD = '#2f7d46', '#b23b3b', '#2f6fb0', '#c8951a', '#6b46a8'
INK, MUTED, SLATE = '#233043', '#6b7684', '#334155'
TILE_BG, TILE_LINE = '#f1f3f5', '#e4e7eb'
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
    '<path d="M6 4h12M6 20h12"/><rect x="9" y="4" width="6" height="16" rx="1"/><path d="M9 8h6M9 12h6M9 16h6"/></g>',
    'scroll': '<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M7 4h12v13a2 2 0 0 1-2 2H7"/><path d="M7 4a2 2 0 0 0 0 4h2"/><path d="M10 8h6M10 12h6"/></g>',
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
    'vp': '<path fill="currentColor" d="M12 3 14.6 9 21 9.5 16.1 13.8 17.6 20 12 16.7 6.4 20 7.9 13.8 3 9.5 9.4 9Z"/>',
    'chain': '<g fill="none" stroke="currentColor" stroke-width="2">'
    '<rect x="3" y="8" width="11" height="8" rx="4"/><rect x="10" y="8" width="11" height="8" rx="4"/></g>',
    'trade': '<g fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M4 9h14M15 6l3 3-3 3"/><path d="M20 15H6M9 12l-3 3 3 3"/></g>',
    'special': '<path fill="currentColor" d="M12 2 13.7 9.3 21 11 13.7 12.7 12 20 10.3 12.7 3 11 10.3 9.3Z"/>',
}

Row = Tuple[List[Tuple[str, str]], str]  # (icons, label)


def _slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', name.lower()) or 'x'


def _shade(hexc: str, f: float) -> str:
    r, g, b = (int(hexc[i : i + 2], 16) for i in (1, 3, 5))
    return '#%02x%02x%02x' % (int(r * f), int(g * f), int(b * f))


def _icon(key: str, x: float, y: float, size: float, color: str, title: str = '') -> str:
    tip = f'<title>{escape(title)}</title>' if title else ''
    return (
        f'<g transform="translate({x:.1f},{y:.1f}) scale({size / 24:.3f})" '
        f'style="color:{color}">{tip}{ICONS[key]}</g>'
    )


def _wrap(name: str, limit: int = 20, maxlines: int = 2) -> List[str]:
    lines: List[str] = []
    cur = ''
    for word in name.split():
        if cur and len(cur) + 1 + len(word) > limit:
            lines.append(cur)
            cur = word
        else:
            cur = f'{cur} {word}'.strip()
    if cur:
        lines.append(cur)
    if len(lines) > maxlines:
        lines = lines[:maxlines]
        lines[-1] = lines[-1][: limit - 1].rstrip() + '…'
    return lines or ['']


def _cost_items(coins: int, resources) -> List[Tuple[str, str, int, str]]:
    items: List[Tuple[str, str, int, str]] = []
    if coins:
        items.append(('coin', RES_COLOR[Ressource.GOLD], coins, 'Coins'))
    for res, n in sorted(resources.items(), key=lambda kv: kv[0].value):
        items.append((RES_ICON[res], RES_COLOR[res], n, RES_NAME[res]))
    return items


def _cost_chip(x: float, y: float, key: str, color: str, count: int, title: str, tile: float = 30) -> str:
    out = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{tile}" height="{tile}" rx="8" fill="{TILE_BG}" stroke="{TILE_LINE}"/>',
        _icon(key, x + (tile - 20) / 2, y + (tile - 20) / 2, 20, color, title),
    ]
    if count > 1:
        cx, cy = x + tile - 5, y + tile - 5
        out.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="8" fill="{INK}"/>'
            f'<text x="{cx:.1f}" y="{cy + 3:.1f}" text-anchor="middle" font-size="9" '
            f'font-weight="700" fill="#fff">{count}</text>'
        )
    return ''.join(out)


def _cost_row(coins: int, resources, x: float, y: float, tile: float = 30, gap: float = 7) -> str:
    items = _cost_items(coins, resources)
    if not items:
        return f'<text x="{x:.0f}" y="{y + 20:.0f}" font-size="12.5" fill="{SCIENCE}">Free</text>'
    out, cx = [], x
    for key, color, count, title in items:
        out.append(_cost_chip(cx, y, key, color, count, title, tile))
        cx += tile + gap
    return ''.join(out)


def _resource_rows(effect) -> List[Row]:
    slots = effect.slots if isinstance(effect, MultiProduction) else [effect.options]
    rows: List[Row] = []
    for opts in slots:
        res = sorted(opts, key=lambda r: r.value)
        rows.append(([(RES_ICON[r], RES_COLOR[r]) for r in res], ' / '.join(RES_NAME[r] for r in res)))
    return rows


def _describe(effect) -> List[Row]:
    if isinstance(effect, (Production, MultiProduction)):
        return _resource_rows(effect)
    if isinstance(effect, Military):
        n = effect.shields_count
        return [([('shield', MILITARY)], f'{n} Shield' + ('s' if n != 1 else ''))]
    if isinstance(effect, Science):
        return [([(effect.symbol.value, SCIENCE)], SCI_NAME[effect.symbol.value])]
    if isinstance(effect, ScienceWild):
        return [([('tablet', SCIENCE), ('gear', SCIENCE), ('compass', SCIENCE)], 'Any science')]
    if isinstance(effect, VictoryPoints):
        return [([('vp', CIVIC)], f'{effect.points} Victory pts')]
    if isinstance(effect, Coins):
        return [([('coin', GOLD)], f'+{effect.amount} Coins')]
    if isinstance(effect, Trade):
        return [([('trade', GOLD)], f'Buy {effect.category} ({effect.side}) {effect.price}c')]
    if isinstance(effect, Debt):
        return [([('coin', MILITARY)], f'{effect.amount} debt ({effect.scope})')]
    if isinstance(effect, Diplomacy):
        label = 'Diplomacy' + (f' x{effect.count}' if effect.count > 1 else '')
        return [([('shield', CIVIC)], label)]
    if isinstance(effect, Plunder):
        return [([('coin', MILITARY)], f'others -{effect.amount}c ({effect.scope})')]
    if isinstance(effect, Grant):
        return [([('coin', GOLD)], f'others +{effect.amount}c ({effect.scope})')]
    if isinstance(effect, GainToken):
        return [([('shield', MILITARY)], f'+war token ({effect.value})')]
    if isinstance(effect, Special):
        return [([('special', GUILD)], effect.description or effect.power.replace('_', ' '))]
    if isinstance(effect, Tally):
        target = '/'.join(sorted(c.value for c in effect.colors)) or effect.what.replace('_', ' ')
        parts = [
            p
            for p in (
                f'{effect.vp_each} VP' if effect.vp_each else '',
                f'{effect.coins_each}c' if effect.coins_each else '',
            )
            if p
        ]
        icon = ('vp', GUILD) if effect.vp_each else ('coin', GUILD)
        return [([icon], f'{" +".join(parts)} / {target} ({SCOPE.get(effect.scope, effect.scope)})')]
    return [([('coin', MUTED)], type(effect).__name__)]


def _row_svg(icons: List[Tuple[str, str]], label: str, x: float, y: float, size: float = 19, font: float = 12.5) -> str:
    out, ix = [], x
    for key, color in icons[:3]:
        out.append(_icon(key, ix, y - size + 2, size, color))
        ix += size + 3
    out.append(f'<text x="{ix + 4:.1f}" y="{y - 3:.1f}" font-size="{font}" fill="{INK}">{escape(label)}</text>')
    return ''.join(out)


def card_svg(card: Card) -> str:
    """A standalone, self-contained ``<svg>`` element for one card."""
    uid = _slug(card.name)
    fill = CARD_FILL[card.color]
    lines = _wrap(card.name, 22, 2)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_W} {_H}" width="{_W}" height="{_H}" '
        f'font-family="ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif" role="img">',
        f'<title>{escape(card.name)}</title>',
        f'<defs><clipPath id="cl{uid}"><rect x="3" y="3" width="{_W - 6}" height="{_H - 6}" rx="16"/></clipPath>'
        f'<linearGradient id="hg{uid}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{_shade(fill, 0.82)}"/><stop offset="1" stop-color="{fill}"/></linearGradient>'
        f'<filter id="sh{uid}" x="-8%" y="-6%" width="116%" height="112%">'
        f'<feDropShadow dx="0" dy="2" stdDeviation="3.2" flood-color="#233043" flood-opacity="0.16"/></filter></defs>',
        f'<rect x="3" y="3" width="{_W - 6}" height="{_H - 6}" rx="16" fill="#fff" stroke="#e2e6ea" '
        f'stroke-width="1.5" filter="url(#sh{uid})"/>',
        f'<g clip-path="url(#cl{uid})">',
        f'<rect x="3" y="3" width="{_W - 6}" height="64" fill="url(#hg{uid})"/>',
        f'<rect x="3" y="65" width="{_W - 6}" height="2" fill="{_shade(fill, 0.62)}"/>',
        f'<text x="18" y="26" font-size="10" letter-spacing="1.4" fill="#ffffff" fill-opacity="0.8">'
        f'{escape(CAT_NAME[card.color].upper())}</text>',
    ]
    if len(lines) == 1:
        parts.append(f'<text x="18" y="52" font-size="17" font-weight="700" fill="#fff">{escape(lines[0])}</text>')
    else:
        parts.append(
            f'<text x="18" y="46" font-size="14.5" font-weight="700" fill="#fff">{escape(lines[0])}</text>'
            f'<text x="18" y="61" font-size="14.5" font-weight="700" fill="#fff">{escape(lines[1])}</text>'
        )
    roman = {1: 'I', 2: 'II', 3: 'III'}.get(card.age or 0, '')
    if roman:
        parts.append(
            f'<rect x="{_W - 44}" y="12" width="30" height="26" rx="7" fill="#fff" fill-opacity="0.22"/>'
            f'<text x="{_W - 29}" y="30" text-anchor="middle" font-size="13" font-weight="700" '
            f'fill="#fff">{roman}</text>'
        )
    parts.append('</g>')

    parts.append(f'<text x="18" y="86" font-size="10" letter-spacing="1.2" fill="{MUTED}">COST</text>')
    parts.append(_cost_row(card.cost.coins, card.cost.resources, 18, 92))

    rows: List[Row] = []
    for effect in card.effects:
        rows += _describe(effect)
    panel_h = _H - 132 - 40
    parts.append(
        f'<rect x="12" y="132" width="{_W - 24}" height="{panel_h}" rx="12" fill="{fill}" fill-opacity="0.06"/>'
        f'<rect x="12" y="132" width="4" height="{panel_h}" rx="2" fill="{fill}"/>'
        f'<text x="26" y="152" font-size="10" letter-spacing="1.2" fill="{MUTED}">EFFECTS</text>'
    )
    y = 178
    for icons, label in rows:
        parts.append(_row_svg(icons, label, 26, y))
        y += 28
    if card.chain_requires:
        parts.append(_icon('chain', 24, y - 18, 18, GUILD))
        parts.append(
            f'<text x="47" y="{y - 4}" font-size="11.5" fill="{GUILD}">free if {escape(card.chain_requires)}</text>'
        )
    elif card.chain_enables:
        parts.append(_icon('chain', 24, y - 18, 18, MUTED))
        parts.append(
            f'<text x="47" y="{y - 4}" font-size="11" fill="{MUTED}">unlocks {escape(card.chain_enables[0])}</text>'
        )

    meta = []
    if card.at_players:
        meta.append(','.join(str(p) for p in card.at_players) + '+ players')
    meta.append(card.expansion)
    if card.guild:
        meta.append('guild')
    parts.append(
        f'<text x="{_W / 2:.0f}" y="{_H - 16}" text-anchor="middle" font-size="10.5" fill="{MUTED}">'
        f'{escape(" · ".join(meta))}</text>'
    )
    parts.append('</svg>')
    return ''.join(parts)


# --- wonder boards -------------------------------------------------------
_WLEFT, _WSTAGE, _WH = 172, 138, 214


def _stage_panel(stage: WonderStage, index: int, x: float) -> str:
    y0, w, h = 18, _WSTAGE - 14, _WH - 36
    out = [
        f'<rect x="{x:.0f}" y="{y0}" width="{w}" height="{h}" rx="10" fill="#fff" stroke="{TILE_LINE}"/>',
        f'<rect x="{x:.0f}" y="{y0}" width="{w}" height="4" rx="2" fill="{GOLD}"/>',
        f'<text x="{x + 12:.0f}" y="{y0 + 26}" font-size="10" letter-spacing="1.2" fill="{MUTED}">'
        f'STAGE {index}</text>',
        f'<text x="{x + 12:.0f}" y="{y0 + 48}" font-size="9.5" fill="{MUTED}">COST</text>',
        _cost_row(stage.cost.coins, stage.cost.resources, x + 12, y0 + 54, tile=26, gap=5),
    ]
    ey = y0 + 108
    for icons, label in [r for e in stage.effects for r in _describe(e)][:2]:
        out.append(_row_svg(icons, label, x + 12, ey, size=18, font=11.5))
        ey += 26
    return ''.join(out)


def wonder_svg(wonder: Wonder) -> str:
    """A standalone ``<svg>`` board for one wonder (starting resource + stages)."""
    uid = _slug(wonder.name + wonder.side)
    n = len(wonder.stages)
    width = _WLEFT + n * _WSTAGE + 12
    lines = _wrap(wonder.name, 16, 4)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {_WH}" width="{width}" height="{_WH}" '
        f'font-family="ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif" role="img">',
        f'<title>{escape(wonder.name)}</title>',
        f'<defs><filter id="wsh{uid}" x="-4%" y="-6%" width="108%" height="112%">'
        f'<feDropShadow dx="0" dy="2" stdDeviation="3.2" flood-color="#233043" flood-opacity="0.16"/></filter></defs>',
        f'<rect x="3" y="3" width="{width - 6}" height="{_WH - 6}" rx="16" fill="#fbfbfc" stroke="#e2e6ea" '
        f'stroke-width="1.5" filter="url(#wsh{uid})"/>',
        f'<rect x="3" y="3" width="{_WLEFT - 3}" height="{_WH - 6}" rx="16" fill="{SLATE}"/>',
        f'<rect x="{_WLEFT - 14}" y="3" width="14" height="{_WH - 6}" fill="{SLATE}"/>',
        '<text x="20" y="34" font-size="10" letter-spacing="1.4" fill="#9fb2c7">WONDER</text>',
    ]
    ty = 34 + 22
    for line in lines:
        parts.append(f'<text x="20" y="{ty}" font-size="15" font-weight="700" fill="#fff">{escape(line)}</text>')
        ty += 19
    parts.append(
        f'<rect x="{_WLEFT - 46}" y="16" width="30" height="24" rx="7" fill="#ffffff" fill-opacity="0.18"/>'
        f'<text x="{_WLEFT - 31}" y="33" text-anchor="middle" font-size="12" font-weight="700" '
        f'fill="#fff">{escape(wonder.side)}</text>'
    )
    parts.append(f'<text x="20" y="{_WH - 44}" font-size="10" letter-spacing="1.2" fill="#9fb2c7">PRODUCES</text>')
    res = wonder.starting_resource
    parts.append(_icon(RES_ICON[res], 20, _WH - 38, 24, '#e7edf3', RES_NAME[res]))
    parts.append(f'<text x="50" y="{_WH - 20}" font-size="13" fill="#e7edf3">{escape(RES_NAME[res])}</text>')
    for i, stage in enumerate(wonder.stages, 1):
        parts.append(_stage_panel(stage, i, _WLEFT + (i - 1) * _WSTAGE))
    parts.append('</svg>')
    return ''.join(parts)


def gallery_html(cards: Iterable[Card], wonders: Sequence[Wonder] = (), title: str = 'Cards') -> str:
    """A full HTML page with card and (optional) wonder-board grids."""
    card_cells = ''.join(f'<div class="cell">{card_svg(c)}</div>' for c in cards)
    body = f'<h2>Cards</h2><div class="grid">{card_cells}</div>' if wonders else f'<div class="grid">{card_cells}</div>'
    if wonders:
        wonder_cells = ''.join(f'<div class="cell">{wonder_svg(w)}</div>' for w in wonders)
        body += f'<h2>Wonder boards</h2><div class="grid wide">{wonder_cells}</div>'
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{escape(title)}</title><style>'
        'body{background:#eceff3;margin:0;padding:32px 24px 56px;'
        'font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif;color:#233043}'
        'h1{font-size:21px;font-weight:650;margin:0 auto 8px;max-width:1180px}'
        'h2{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:#6b7684;'
        'max-width:1180px;margin:28px auto 14px}'
        '.grid{display:flex;flex-wrap:wrap;gap:20px;max-width:1180px;margin:0 auto}'
        '.grid.wide{flex-direction:column;align-items:flex-start}'
        '.cell svg{display:block}'
        f'</style></head><body><h1>{escape(title)}</h1>{body}</body></html>'
    )
