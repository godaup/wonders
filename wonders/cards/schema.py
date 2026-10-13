"""Parameter descriptions of the effect types card and wonder JSON can use.

:data:`EFFECT_SCHEMAS` describes, for every type in :data:`~wonders.cards.content.EFFECT_BUILDERS`, the
fields its spec takes — enough to generate a form (the GUI's expansion builder does) and to explain
an effect without reading code. Plugins describe their own effect types through
``Registry.register_effect(..., schema=...)``; an effect type without a schema is edited as raw JSON.

A parameter's ``kind`` says what its value looks like in the spec:

* ``int`` / ``bool`` / ``text`` — a number, a flag, free text;
* ``choice`` — one of ``choices``;
* ``resource`` / ``resources`` — one resource id / a list of them (``production``: one per turn);
* ``slots`` — a list of resource-id lists (several productions at once);
* ``color`` / ``colors`` — one card colour id / a list of them (official or the expansion's own);
* ``target`` — a card colour id or ``'wonder'``;
* ``symbol`` — a science symbol.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

SCOPES = ('self', 'neighbours', 'both', 'others')
TALLY_TARGETS = ('colors', 'wonder_stages', 'defeat_tokens', 'tokens', 'victory_tokens', 'coins', 'resource_symbols')
KINDS = ('int', 'bool', 'text', 'choice', 'resource', 'resources', 'slots', 'color', 'colors', 'target', 'symbol')
GROUPS = ('Resources', 'Points', 'Coins', 'Military', 'Science', 'Trade', 'Powers', 'Leaders', 'Cities')


@dataclass(frozen=True)
class Param:
    """One field of an effect spec."""

    name: str
    kind: str
    label: str
    required: bool = False
    default: Any = None
    choices: Tuple[str, ...] = ()
    help: str = ''


@dataclass(frozen=True)
class EffectSchema:
    """What an effect type does and the fields its spec takes (``description`` is always allowed)."""

    label: str
    group: str
    help: str
    params: Tuple[Param, ...] = ()
    example: Dict[str, Any] = field(default_factory=dict)  # a valid spec (without 'type')

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _p(name: str, kind: str, label: str, **kw: Any) -> Param:
    if 'choices' in kw:
        kw['choices'] = tuple(kw['choices'])
    return Param(name, kind, label, **kw)


def _count(default: int = 1, label: str = 'Count', help: str = '') -> Param:
    return _p('count', 'int', label, default=default, help=help)


def _scope(default: str) -> Param:
    return _p(
        'scope',
        'choice',
        'Whom',
        default=default,
        choices=SCOPES,
        help='whose cities count or are affected',
    )


EFFECT_SCHEMAS: Dict[str, EffectSchema] = {
    'production': EffectSchema(
        'Production',
        'Resources',
        'Produces one of the listed resources every turn (list one for a plain resource card).',
        (_p('resources', 'resources', 'Resources', required=True),),
        {'resources': ['wood']},
    ),
    'multi_production': EffectSchema(
        'Several productions',
        'Resources',
        'Produces several resources at once: one from each slot every turn.',
        (_p('slots', 'slots', 'Slots', required=True, help='each slot produces one of its resources'),),
        {'slots': [['wood'], ['stone', 'ore']]},
    ),
    'coins': EffectSchema(
        'Coins',
        'Coins',
        'Coins from the bank when built.',
        (_p('amount', 'int', 'Coins', required=True),),
        {'amount': 5},
    ),
    'military': EffectSchema(
        'Shields',
        'Military',
        'Military strength for the conflicts at the end of each Age.',
        (_p('shields', 'int', 'Shields', required=True),),
        {'shields': 1},
    ),
    'science': EffectSchema(
        'Science symbol',
        'Science',
        'One science symbol.',
        (_p('symbol', 'symbol', 'Symbol', required=True),),
        {'symbol': 'compass'},
    ),
    'science_wild': EffectSchema(
        'Any science symbol',
        'Science',
        'Counts as whichever science symbol scores best at the end.',
        (_count(),),
        {'count': 1},
    ),
    'trade': EffectSchema(
        'Trading discount',
        'Trade',
        'Buy resources from neighbours for a lower price.',
        (
            _p('side', 'choice', 'Neighbour', default='both', choices=('left', 'right', 'both')),
            _p('category', 'choice', 'Resources', default='both', choices=('raw', 'manufactured', 'both')),
            _p('price', 'int', 'Price', default=1, help='coins per resource'),
        ),
        {'side': 'left', 'category': 'raw'},
    ),
    'victory_points': EffectSchema(
        'Victory points',
        'Points',
        'Victory points at the end of the game.',
        (_p('points', 'int', 'Points', required=True),),
        {'points': 3},
    ),
    'tally': EffectSchema(
        'Count and reward',
        'Points',
        'Counts something in the chosen cities: coins when built and/or points at the end per unit.',
        (
            _p(
                'what',
                'choice',
                'Count',
                required=True,
                choices=TALLY_TARGETS,
                help='cards of colours, wonder stages, military tokens, or coins (per 3)',
            ),
            _scope('self'),
            _p('colors', 'colors', 'Colours', help='for counting cards'),
            _p('coins_each', 'int', 'Coins each', default=0),
            _p('vp_each', 'int', 'Points each', default=0),
            _p('token_value', 'int', 'Token value', help='for victory tokens: only count tokens of this value'),
            _p('resource', 'resource', 'Resource', help='for resource symbols: which resource to count'),
        ),
        {'what': 'colors', 'scope': 'both', 'colors': ['brown'], 'coins_each': 1},
    ),
    'special': EffectSchema(
        'Text only',
        'Powers',
        'Shows a text on the card but has no effect on play.',
        (_p('power', 'text', 'Name', required=True),),
        {'power': 'my_power', 'description': 'text only'},
    ),
    'debt': EffectSchema(
        'Debt',
        'Cities',
        'Gives debt tokens (-1 point each at the end).',
        (_p('amount', 'int', 'Debt tokens', required=True), _scope('self')),
        {'amount': 1, 'scope': 'neighbours'},
    ),
    'diplomacy': EffectSchema(
        'Diplomacy',
        'Cities',
        'Skip the next military conflict (the neighbours then fight each other).',
        (_count(),),
        {'count': 1},
    ),
    'plunder': EffectSchema(
        'Plunder',
        'Cities',
        'The chosen players lose coins; what they cannot pay becomes debt.',
        (_p('amount', 'int', 'Coins', required=True), _scope('others')),
        {'amount': 1, 'scope': 'others'},
    ),
    'grant': EffectSchema(
        'Grant',
        'Cities',
        'The chosen players gain coins.',
        (_p('amount', 'int', 'Coins', required=True), _scope('others')),
        {'amount': 1, 'scope': 'others'},
    ),
    'gain_token': EffectSchema(
        'Military token',
        'Military',
        'Take a military victory token of this value right away.',
        (_p('value', 'int', 'Value', required=True),),
        {'value': 1},
    ),
    'play_last_card': EffectSchema('Play the last card', 'Powers', 'Also play the seventh card of each Age (Babylon).'),
    'free_build': EffectSchema(
        'Free build', 'Powers', 'Build one card per Age for free (Olympia).', (_count(),), {'count': 1}
    ),
    'build_from_discard': EffectSchema(
        'Build from the discard pile',
        'Powers',
        'Build one discarded card for free (Halicarnassus).',
        (
            _count(),
            _p('priority', 'int', 'Order', default=0, help='when several are owed in one turn: lower first'),
        ),
        {'count': 1},
    ),
    'copy_guild': EffectSchema(
        'Copy a guild', 'Powers', "At the end, copy a neighbour's guild (Olympia B).", (_count(),), {'count': 1}
    ),
    'color_sets': EffectSchema(
        'Colour sets',
        'Points',
        'Points per set of one card of each listed colour.',
        (_p('colors', 'colors', 'Colours', required=True), _p('vp_each', 'int', 'Points per set', required=True)),
        {'colors': ['blue', 'red', 'green'], 'vp_each': 3},
    ),
    'science_set_bonus': EffectSchema(
        'Science set bonus',
        'Science',
        'Extra points per complete set of the three science symbols.',
        (_p('points', 'int', 'Points', default=3),),
        {'points': 3},
    ),
    'science_copy': EffectSchema(
        'Copy a science symbol',
        'Science',
        "At the end, copy a science symbol from a neighbour's green cards.",
        (_count(),),
        {'count': 1},
    ),
    'build_income': EffectSchema(
        'Income per build',
        'Coins',
        'Coins each time you build a card that matches every condition.',
        (
            _p('coins', 'int', 'Coins', required=True),
            _p('color', 'color', 'Colour', help='only cards of this colour'),
            _p('via_chain', 'bool', 'Built through a chain', default=False),
            _p('after_trade', 'bool', 'Per neighbour bought from', default=False),
        ),
        {'coins': 2, 'color': 'yellow'},
    ),
    'victory_income': EffectSchema(
        'Income per victory',
        'Military',
        'Coins each time you win a military victory token.',
        (_p('coins', 'int', 'Coins', default=2),),
        {'coins': 2},
    ),
    'redirect_defeats': EffectSchema(
        'Pass on defeats', 'Military', 'Your defeat tokens go to the neighbour who beat you.'
    ),
    'resource_discount': EffectSchema(
        'Resource discount',
        'Powers',
        'Pay fewer resources for cards of a colour or for wonder stages.',
        (_p('target', 'target', 'For', required=True), _p('amount', 'int', 'Resources fewer', default=1)),
        {'target': 'green', 'amount': 1},
    ),
    'build_free': EffectSchema(
        'Build for free',
        'Powers',
        'Cards of these colours cost you nothing.',
        (_p('colors', 'colors', 'Colours', required=True),),
        {'colors': ['white']},
    ),
    'bank_purchase': EffectSchema(
        'Buy from the bank',
        'Trade',
        'Once per turn, buy any resource from the bank for 1 coin.',
        (_count(),),
        {'count': 1},
    ),
    'produce_unowned': EffectSchema(
        'Produce a missing resource', 'Resources', 'Once per turn, produce one resource you do not produce yet.'
    ),
    'produce_owned': EffectSchema(
        'Produce an extra resource', 'Resources', 'Once per turn, produce one more of a resource you produce.'
    ),
    'starting_resource_discount': EffectSchema(
        'Cheaper starting resource',
        'Trade',
        "Buy your board's resource cheaper from neighbours.",
        (_p('amount', 'int', 'Coins less', default=1),),
        {'amount': 1},
    ),
    'cash_defeats': EffectSchema(
        'Cash in defeats',
        'Military',
        'Gain coins per defeat token, then discard your defeat tokens.',
        (_p('coins_each', 'int', 'Coins each', default=2),),
        {'coins_each': 2},
    ),
    'plunder_per': EffectSchema(
        'Plunder per unit',
        'Cities',
        'The chosen players lose coins per unit of what they have; the rest becomes debt.',
        (
            _p('what', 'choice', 'Per', required=True, choices=TALLY_TARGETS),
            _p('amount', 'int', 'Coins each', default=1),
            _scope('others'),
        ),
        {'what': 'victory_tokens', 'amount': 1, 'scope': 'others'},
    ),
    'coin_discount': EffectSchema(
        'Coin discount',
        'Leaders',
        'Cards of these colours cost fewer coins, for you and for your neighbours.',
        (
            _p('colors', 'colors', 'Colours', required=True),
            _p('amount', 'int', 'Coins less for you', default=0),
            _p('neighbours_amount', 'int', 'Coins less for neighbours', default=0),
        ),
        {'colors': ['white'], 'amount': 2, 'neighbours_amount': 1},
    ),
    'leader_draw': EffectSchema(
        'Draw leaders', 'Leaders', 'Draw leaders from the box into your hand (Rome B).', (_count(4),), {'count': 4}
    ),
    'extra_recruit': EffectSchema(
        'Recruit a leader', 'Leaders', 'Recruit an extra leader right away (Rome B).', (_count(),), {'count': 1}
    ),
    'copy_leader': EffectSchema(
        'Copy a leader',
        'Leaders',
        "Copy one of your neighbours' leaders (Courtesans' Guild).",
        (_count(),),
        {'count': 1},
    ),
    'buried_color': EffectSchema(
        'Colour of the buried card',
        'Points',
        "At the end, points per card of the buried card's colour in the chosen cities (Stonehenge B).",
        (_p('vp_each', 'int', 'Points each', default=1), _scope('neighbours')),
        {'vp_each': 1},
    ),
    'entomb_leader': EffectSchema(
        'Entomb a leader',
        'Leaders',
        'Entomb one of your recruited leaders: its power is lost, it scores twice its cost (Abu Simbel).',
        (_count(),),
        {'count': 1},
    ),
    'free_first_of_color': EffectSchema(
        'First card of each colour free',
        'Powers',
        'The first card of each colour in your city costs nothing (Olympia, second edition).',
    ),
    'free_turn_build': EffectSchema(
        'Free card on the first or last turn',
        'Powers',
        'Build the card of the first or the last turn of each Age for free (Olympia, second edition).',
        (_p('turn', 'choice', 'Turn', default='first', choices=('first', 'last')),),
        {'turn': 'first'},
    ),
    'wonder_complete': EffectSchema(
        'Points for a complete wonder',
        'Points',
        'Victory points only if every stage of your wonder is built (Decorators Guild).',
        (_p('points', 'int', 'Points', default=7),),
        {'points': 7},
    ),
}


def schema_for(effect_type: str) -> Optional[EffectSchema]:
    return EFFECT_SCHEMAS.get(effect_type)


def describe() -> List[Dict[str, Any]]:
    """Every effect type that can be built, with its schema (``None``: no schema, edited as JSON)."""
    from wonders.cards.content import EFFECT_BUILDERS

    return [
        {'type': name, 'schema': (EFFECT_SCHEMAS[name].as_dict() if name in EFFECT_SCHEMAS else None)}
        for name in EFFECT_BUILDERS
    ]
