"""Render the whole table as a simple stats GUI (not a literal tabletop).

Each player becomes a compact "city panel": header (name / wonder / score), a row
of stat chips (coins, shields, military tokens, wonder progress), science symbols, a
per-colour card tally, any Cities debt/diplomacy, and a stacked bar showing how the
score breaks down by category. Built on the card renderer's icon set and palette.
"""

from __future__ import annotations

from collections import Counter
from html import escape
from typing import TYPE_CHECKING, List, Tuple

from wonders.cards.base import Color
from wonders.viz.render import CARD_FILL, CIVIC, GOLD, INK, MILITARY, MUTED, SCIENCE, SLATE, _icon

if TYPE_CHECKING:  # pragma: no cover - typing only
    from wonders.engine.game import Game

_PW, _PH, _GAP = 328, 212, 16
_HEADER = 44

# Score-breakdown segments -> (label, colour).
BREAKDOWN: List[Tuple[str, str]] = [
    ('military', MILITARY),
    ('treasury', GOLD),
    ('wonders', '#5b6470'),
    ('civilian', CIVIC),
    ('commercial', '#d99a1c'),
    ('guilds', '#7a4fb0'),
    ('science', SCIENCE),
    ('leaders', '#c9c2a8'),
    ('cities', '#2b2b2b'),
]
CATEGORY_COLORS = [
    ('brown', CARD_FILL[Color.BROWN]),
    ('grey', CARD_FILL[Color.GREY]),
    ('red', CARD_FILL[Color.RED]),
    ('blue', CARD_FILL[Color.BLUE]),
    ('green', CARD_FILL[Color.GREEN]),
    ('yellow', CARD_FILL[Color.YELLOW]),
    ('purple', CARD_FILL[Color.PURPLE]),
]


def _stat(icon: str, value: str, x: float, y: float, color: str) -> str:
    return (
        _icon(icon, x, y, 18, color) + f'<text x="{x + 22:.0f}" y="{y + 14:.0f}" font-size="13" font-weight="600" '
        f'fill="{INK}">{escape(value)}</text>'
    )


def _score_bar(breakdown: dict, x: float, y: float, width: float) -> str:
    segments = [(k, c, max(0, int(breakdown.get(k, 0)))) for k, c in BREAKDOWN]
    total_pos = sum(v for _k, _c, v in segments) or 1
    out = [f'<rect x="{x:.0f}" y="{y:.0f}" width="{width:.0f}" height="12" rx="6" fill="#eef1f4"/>']
    cx = x
    for _label, color, value in segments:
        if value <= 0:
            continue
        w = width * value / total_pos
        out.append(f'<rect x="{cx:.1f}" y="{y:.0f}" width="{w:.1f}" height="12" fill="{color}"/>')
        cx += w
    # re-round the caps so the bar reads as one rounded pill
    out.append(
        f'<rect x="{x:.0f}" y="{y:.0f}" width="{width:.0f}" height="12" rx="6" fill="none" '
        f'stroke="#e2e6ea" stroke-width="1"/>'
    )
    return ''.join(out)


def _panel(game: 'Game', i: int, x: float, y: float, leader: bool) -> str:
    p = game.players[i]
    score = game.score(i)
    b = score.as_dict()
    colors = Counter(c.color.value for c in p.cards)
    sci = {s.value: n for s, n in p.science().items()}
    stroke = GOLD if leader else '#e2e6ea'
    sw = 2.5 if leader else 1.5

    parts = [
        f'<g transform="translate({x:.0f},{y:.0f})">',
        f'<rect x="0" y="0" width="{_PW}" height="{_PH}" rx="14" fill="#fff" '
        f'stroke="{stroke}" stroke-width="{sw}"/>',
        f'<rect x="0" y="0" width="{_PW}" height="{_HEADER}" rx="14" fill="{SLATE}"/>',
        f'<rect x="0" y="{_HEADER - 14}" width="{_PW}" height="14" fill="{SLATE}"/>',
        f'<text x="14" y="20" font-size="14" font-weight="700" fill="#fff">{escape(p.name)}</text>',
        f'<text x="14" y="36" font-size="10" fill="#c3ceda">'
        f'{escape(p.wonder.name)} · {escape(p.wonder.side)}</text>',
        f'<text x="{_PW - 14}" y="30" text-anchor="end" font-size="22" font-weight="800" '
        f'fill="#fff">{b["total"]}</text>',
    ]
    # stat chips
    row = _HEADER + 26
    parts.append(_stat('coin', str(p.coins), 14, row, GOLD))
    parts.append(_stat('shield', str(p.shields()), 96, row, MILITARY))
    tokens = sum(p.tokens)
    parts.append(_stat('special', f'{tokens:+d}', 168, row, MILITARY if tokens < 0 else CIVIC))
    parts.append(_stat('vp', f'{p.wonder_stages_built}/{len(p.wonder.stages)}', 236, row, '#5b6470'))
    # science symbols
    row2 = row + 30
    sx = 14
    for sym in ('tablet', 'gear', 'compass'):
        parts.append(_stat(sym, str(sci.get(sym, 0)), sx, row2, SCIENCE))
        sx += 62
    if p.debt:
        parts.append(_stat('coin', f'-{p.debt}', sx, row2, MILITARY))
        sx += 62
    if p.diplomacy:
        parts.append(_stat('shield', f'D{p.diplomacy}', sx, row2, CIVIC))
    # per-colour card tally
    row3 = row2 + 28
    parts.append(f'<text x="14" y="{row3}" font-size="9" letter-spacing="1" fill="{MUTED}">CARDS</text>')
    dx = 14
    for name, color in CATEGORY_COLORS:
        n = colors.get(name, 0)
        parts.append(
            f'<rect x="{dx:.0f}" y="{row3 + 6}" width="26" height="20" rx="6" '
            f'fill="{color}" fill-opacity="{0.9 if n else 0.15}"/>'
            f'<text x="{dx + 13:.0f}" y="{row3 + 20}" text-anchor="middle" font-size="11" '
            f'font-weight="700" fill="{"#fff" if n else MUTED}">{n}</text>'
        )
        dx += 31
    # score-breakdown bar
    parts.append(_score_bar(b, 14, _PH - 24, _PW - 28))
    parts.append('</g>')
    return ''.join(parts)


def table_svg(game: 'Game', columns: int = 3) -> str:
    """A standalone ``<svg>`` overview of every player's city."""
    n = game.n
    cols = min(columns, n)
    rows = (n + cols - 1) // cols
    leader = max(range(n), key=lambda i: game.score(i).total)
    width = cols * _PW + (cols + 1) * _GAP
    top = 40
    height = top + rows * _PH + (rows + 1) * _GAP
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" font-family="ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#eef1f4"/>',
        f'<text x="{_GAP}" y="26" font-size="15" font-weight="700" fill="{INK}">'
        f'Age {game.age or "-"} · {escape(game.phase.value)}</text>',
        f'<text x="{width - _GAP}" y="26" text-anchor="end" font-size="12" fill="{MUTED}">'
        f'leader: {escape(game.players[leader].name)}</text>',
    ]
    for i in range(n):
        col, r = i % cols, i // cols
        x = _GAP + col * (_PW + _GAP)
        y = top + _GAP + r * (_PH + _GAP)
        parts.append(_panel(game, i, x, y, leader=i == leader))
    parts.append('</svg>')
    return ''.join(parts)


def table_html(game: 'Game', title: str = 'Table') -> str:
    """A full HTML page wrapping :func:`table_svg`."""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{escape(title)}</title>'
        '<style>body{margin:0;background:#eef1f4}svg{display:block;max-width:100%;height:auto;margin:0 auto}</style>'
        f'</head><body>{table_svg(game)}</body></html>'
    )
