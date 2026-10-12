"""Optional, combinable expansions.

An expansion is a named module that can adjust setup and rules. A :class:`Ruleset`
is assembled from any combination of expansion ids (``base`` is always implied) and
composes their contributions — e.g. Leaders raises the starting treasury to 6. The
content side combines the same way: ``load_content(*ruleset.content_expansions)``
merges each expansion's cards and wonders.

Most rule tweaks are expressed as effects/player-state that are simply inert when the
expansion is absent (debt tokens are 0, diplomacy tokens are 0), so combining is
just a union — no special-casing in the engine.

The few numbers the engine itself fixes — starting coins, coins for a discard, coins per treasury
point, military tokens, the science set bonus, the guild count — are **meta rules** (:data:`RULES`):
an expansion may change any of them (``Expansion.rules``; extension files say ``"rules": {...}``),
either setting a value (``"discard_coins": 2``) or changing it by a delta
(``"starting_coins": {"add": 2}``). :func:`rules_of` resolves a combination from the base values,
applying the expansions in a fixed order however they were chosen — ``base``, the bundled expansions,
then extension expansions by id — so deltas add up, a set replaces what came before it, and the result
is clamped to the rule's range. ``Game`` resolves its own rules from its expansions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

DEFAULT_STARTING_COINS = 3

# Version of the game rules as implemented by the engine. Bump it whenever a rules change alters
# how games play out, so models trained under older rules can be recognised (checkpoints record
# the version they were trained under; see wonders.ml.model.save_checkpoint).
#   1 - everything before the October 2026 fixes (also every checkpoint without a version)
#   2 - neighbours may only buy board and brown / grey production; cards buried under a wonder
#       stage and discarded leaders no longer go to the discard pile
RULES_VERSION = 2
LEGACY_RULES_VERSION = 1


@dataclass(frozen=True)
class RuleSpec:
    """One meta rule an expansion may override: its default and the allowed range."""

    name: str
    default: int
    minimum: int
    maximum: int
    label: str
    help: str


RULES: Tuple[RuleSpec, ...] = (
    RuleSpec('starting_coins', DEFAULT_STARTING_COINS, 0, 30, 'Starting coins', 'Coins every player starts with.'),
    RuleSpec('discard_coins', 3, 0, 20, 'Coins for a discard', 'Coins for discarding a card.'),
    RuleSpec('coins_per_point', 3, 1, 20, 'Coins per point', 'At the end, one victory point per this many coins.'),
    RuleSpec('science_set_points', 7, 0, 30, 'Science set bonus', 'Victory points per set of the three symbols.'),
    RuleSpec('victory_age_1', 1, 0, 20, 'Victory in Age I', 'Victory points of a military victory token in Age I.'),
    RuleSpec('victory_age_2', 3, 0, 20, 'Victory in Age II', 'Victory points of a military victory token in Age II.'),
    RuleSpec('victory_age_3', 5, 0, 20, 'Victory in Age III', 'Victory points of a military victory token in Age III.'),
    RuleSpec('defeat_points', -1, -10, 0, 'Defeat', 'Victory points of a military defeat token (zero or less).'),
    RuleSpec('extra_guilds', 2, 0, 10, 'Extra guilds', 'Age III deals as many guilds as players plus this number.'),
)
RULE_SPECS: Dict[str, RuleSpec] = {r.name: r for r in RULES}


@dataclass(frozen=True)
class Rules:
    """The resolved meta rules of a game (every field defaults to the base game)."""

    starting_coins: int = DEFAULT_STARTING_COINS
    discard_coins: int = 3
    coins_per_point: int = 3
    science_set_points: int = 7
    victory_age_1: int = 1
    victory_age_2: int = 3
    victory_age_3: int = 5
    defeat_points: int = -1
    extra_guilds: int = 2

    def victory(self, age: int) -> int:
        """Victory points of a military victory token in ``age``."""
        return (self.victory_age_1, self.victory_age_2, self.victory_age_3)[age - 1]

    def changed(self) -> Dict[str, int]:
        """The rules that differ from the base game."""
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) != f.default}


RuleValue = Union[int, Mapping[str, int]]  # a value to set, or {"add": delta}


def _whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def check_rules(rules: Any) -> List[str]:
    """Problems with an expansion's ``"rules"`` (an empty list means they are fine).

    Each rule is a whole number to set, within the rule's range, or ``{"add": n}`` to change it by
    ``n`` (at most the width of the range either way).
    """
    if not isinstance(rules, Mapping):
        return ['rules must be an object of rule name to number']
    problems = []
    for name, value in rules.items():
        spec = RULE_SPECS.get(name)
        if spec is None:
            problems.append(f'unknown rule {name!r} (known: {", ".join(RULE_SPECS)})')
        elif isinstance(value, Mapping):
            width = spec.maximum - spec.minimum
            if set(value) != {'add'} or not _whole(value['add']):
                problems.append(f'rule {name!r}: a change is written {{"add": n}} with a whole number n, not {value!r}')
            elif not -width <= value['add'] <= width:
                problems.append(f'rule {name!r}: a change must be between {-width} and {width}, not {value["add"]}')
        elif not _whole(value):
            problems.append(f'rule {name!r} must be a whole number or {{"add": n}}, not {value!r}')
        elif not spec.minimum <= value <= spec.maximum:
            problems.append(f'rule {name!r} must be between {spec.minimum} and {spec.maximum}, not {value}')
    return problems


def apply_rule(current: int, value: RuleValue) -> int:
    """One expansion's rule applied to the value so far: a number sets it, ``{"add": n}`` changes it."""
    return current + int(value['add']) if isinstance(value, Mapping) else int(value)


@dataclass(frozen=True)
class Expansion:
    id: str
    name: str
    starting_coins: Optional[int] = None  # shorthand for rules['starting_coins']
    # False for expansions whose subsystems are not yet mechanized (content/registry
    # only). They can still be listed in a ruleset; the engine flags the gap.
    implemented: bool = True
    note: str = ''
    rules: Mapping[str, RuleValue] = field(default_factory=dict)  # meta-rule changes (see RULES)

    @property
    def overrides(self) -> Dict[str, RuleValue]:
        """The meta rules this expansion sets or changes."""
        out = dict(self.rules)
        if self.starting_coins is not None:
            out.setdefault('starting_coins', self.starting_coins)
        return out


def rule_order(expansions: Sequence[str]) -> List[str]:
    """The order meta rules apply in, whatever order the expansions were chosen in.

    ``base`` first, then the bundled expansions (in their registry order), then extension expansions by id.
    """
    unique = list(dict.fromkeys(expansions))
    return sorted(unique, key=lambda i: (0, BUNDLED.index(i), '') if i in BUNDLED else (1, 0, i))


def rules_of(expansions: Sequence[str]) -> Rules:
    """The meta rules of a combination of expansions.

    The base values with every expansion's rules applied in :func:`rule_order` (deltas add up, a set
    replaces what came before), clamped to each rule's range.
    """
    if any(i not in EXPANSIONS for i in expansions):
        from wonders.extensions import provide

        provide(expansions)  # an extension expansion: discover extensions (once), or load it from the catalog
    values: Dict[str, int] = {}
    for i in rule_order(expansions):
        exp = EXPANSIONS.get(i)
        if exp is not None:
            for name, value in exp.overrides.items():
                values[name] = apply_rule(values.get(name, RULE_SPECS[name].default), value)
    if not values:
        return _DEFAULT_RULES
    return Rules(**{n: min(max(v, RULE_SPECS[n].minimum), RULE_SPECS[n].maximum) for n, v in values.items()})


_DEFAULT_RULES = Rules()


EXPANSIONS: Dict[str, Expansion] = {
    'base': Expansion('base', 'Base game'),
    'leaders': Expansion('leaders', 'Leaders', starting_coins=6),
    'cities': Expansion('cities', 'Cities'),
    'wonderpack': Expansion('wonderpack', 'Wonder Pack'),  # four boards; Abu Simbel needs Leaders
    'armada': Expansion(
        'armada',
        'Armada',
        implemented=False,
        note='naval boards, fleets and naval combat are not yet mechanized',
    ),
}
BUNDLED: Tuple[str, ...] = tuple(EXPANSIONS)  # the expansions this package ships (registered above)


@dataclass(frozen=True)
class Ruleset:
    """A combination of active expansions (``base`` always included)."""

    expansions: Tuple[str, ...]

    @classmethod
    def of(cls, *ids: str) -> 'Ruleset':
        combined = tuple(dict.fromkeys(('base', *ids)))  # dedupe, base first, keep order
        unknown = [i for i in combined if i not in EXPANSIONS]
        if unknown:  # maybe an extension expansion: discover extensions (once) or use the catalog, look again
            from wonders.extensions import provide

            provide(unknown)
            unknown = [i for i in combined if i not in EXPANSIONS]
        if unknown:
            raise ValueError(f'unknown expansion(s): {unknown}')
        return cls(combined)

    def uses(self, expansion_id: str) -> bool:
        return expansion_id in self.expansions

    @property
    def content_expansions(self) -> Tuple[str, ...]:
        return self.expansions

    @property
    def rules(self) -> Rules:
        return rules_of(self.expansions)

    @property
    def starting_coins(self) -> int:
        return self.rules.starting_coins

    @property
    def unimplemented(self) -> Tuple[str, ...]:
        """Active expansions whose subsystems are not fully mechanized yet."""
        return tuple(i for i in self.expansions if not EXPANSIONS[i].implemented)
