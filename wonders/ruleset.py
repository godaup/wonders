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
an expansion may override any of them (``Expansion.rules``; extension files say ``"rules": {...}``).
:class:`Rules` resolves them for a combination: defaults, then each expansion's overrides in order,
so a later expansion wins. ``Game`` resolves its own rules from its expansions (:func:`rules_of`).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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


def check_rules(rules: Any) -> List[str]:
    """Problems with an expansion's ``"rules"`` overrides (an empty list means they are fine)."""
    if not isinstance(rules, Mapping):
        return ['rules must be an object of rule name to number']
    problems = []
    for name, value in rules.items():
        spec = RULE_SPECS.get(name)
        if spec is None:
            problems.append(f'unknown rule {name!r} (known: {", ".join(RULE_SPECS)})')
        elif not isinstance(value, int) or isinstance(value, bool):
            problems.append(f'rule {name!r} must be a whole number, not {value!r}')
        elif not spec.minimum <= value <= spec.maximum:
            problems.append(f'rule {name!r} must be between {spec.minimum} and {spec.maximum}, not {value}')
    return problems


@dataclass(frozen=True)
class Expansion:
    id: str
    name: str
    starting_coins: Optional[int] = None  # shorthand for rules['starting_coins']
    # False for expansions whose subsystems are not yet mechanized (content/registry
    # only). They can still be listed in a ruleset; the engine flags the gap.
    implemented: bool = True
    note: str = ''
    rules: Mapping[str, int] = field(default_factory=dict)  # meta-rule overrides (see RULES)

    @property
    def overrides(self) -> Dict[str, int]:
        """The meta rules this expansion sets."""
        out = dict(self.rules)
        if self.starting_coins is not None:
            out.setdefault('starting_coins', self.starting_coins)
        return out


def rules_of(expansions: Sequence[str]) -> Rules:
    """The meta rules of a combination: defaults, then each expansion's overrides in order (later wins)."""
    if any(i not in EXPANSIONS for i in expansions):
        from wonders.extensions import ensure_loaded

        ensure_loaded()  # an extension expansion: discover extensions (once)
    overrides: Dict[str, int] = {}
    for i in expansions:
        exp = EXPANSIONS.get(i)
        if exp is not None:
            overrides.update(exp.overrides)
    return Rules(**overrides) if overrides else _DEFAULT_RULES


_DEFAULT_RULES = Rules()


EXPANSIONS: Dict[str, Expansion] = {
    'base': Expansion('base', 'Base game'),
    'leaders': Expansion('leaders', 'Leaders', starting_coins=6),
    'cities': Expansion('cities', 'Cities'),
    'armada': Expansion(
        'armada',
        'Armada',
        implemented=False,
        note='naval boards, fleets and naval combat are not yet mechanized',
    ),
}


@dataclass(frozen=True)
class Ruleset:
    """A combination of active expansions (``base`` always included)."""

    expansions: Tuple[str, ...]

    @classmethod
    def of(cls, *ids: str) -> 'Ruleset':
        combined = tuple(dict.fromkeys(('base', *ids)))  # dedupe, base first, keep order
        unknown = [i for i in combined if i not in EXPANSIONS]
        if unknown:  # maybe an extension expansion: discover extensions (once) and look again
            from wonders.extensions import ensure_loaded

            ensure_loaded()
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
