"""Deck sizes per Age and player count, and the check that every Age deck splits evenly into hands.

Each Age's deck is dealt into hands of ``len(deck) // players`` cards; cards left over are silently put
aside (:func:`~wonders.cards.content.deal_hands`), so an expansion must keep every Age deck a multiple
of the player count — at every count from 3 to 7 and in every combination with the bundled expansions
that change Age decks (Leaders adds guilds to the Age III draw, Cities a black card per player and
Age). Deck sizes here are counted exactly, without dealing:

* Age cards contribute one copy per ``at_players`` entry up to the player count;
* Age III draws ``players + extra_guilds`` guilds, or every guild when the pool is smaller;
* Cities draws one black card per player and Age.

:func:`check_decks` reports what an expansion's cards and its ``extra_guilds`` rule do to the deck
sizes: **errors** where cards are left over, **warnings** where the hand size changes without leftovers
(a design choice, e.g. hands of 8) or the guild pool is too small for the guilds a game wants.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

from wonders.cards.base import Card, Color
from wonders.ruleset import RULE_SPECS, RuleValue, apply_rule, check_rules, rules_of

PLAYERS: Tuple[int, ...] = (3, 4, 5, 6, 7)
AGES: Tuple[int, ...] = (1, 2, 3)
# Bundled expansions that change Age decks (the others add no Age cards): every subset is checked.
DECK_EXPANSIONS: Tuple[str, ...] = ('leaders', 'cities')
NUMERALS = {1: 'I', 2: 'II', 3: 'III'}


@dataclass(frozen=True)
class DeckSize:
    """One Age's deck at one player count."""

    cards: int
    players: int
    guilds_wanted: int = 0  # Age III: players + extra guilds
    guild_pool: int = 0

    @property
    def hand(self) -> int:
        return self.cards // self.players

    @property
    def leftover(self) -> int:
        return self.cards % self.players

    @property
    def short_of_guilds(self) -> bool:
        return self.guilds_wanted > self.guild_pool


def deck_size(cards: Sequence[Card], age: int, players: int, extra_guilds: int = 2) -> DeckSize:
    """The size of ``age``'s deck at ``players`` (see the module docstring), counted without dealing."""
    size = 0
    guilds = black = 0
    for card in cards:
        if card.color is Color.WHITE or card.age != age:
            continue
        if card.guild:
            guilds += 1
        elif card.color is Color.BLACK:
            black += 1
        else:
            size += card.copies_for(players)
    wanted = 0
    if age == 3:
        wanted = players + extra_guilds
        size += min(wanted, guilds)
    size += min(players, black)
    return DeckSize(size, players, wanted, guilds)


def extra_guilds_with(expansions: Sequence[str], rule: Optional[RuleValue]) -> int:
    """``extra_guilds`` of a bundled combination with an extension's own rule applied last (as the game does)."""
    value = rules_of(expansions).extra_guilds
    if rule is None or check_rules({'extra_guilds': rule}):
        return value
    spec = RULE_SPECS['extra_guilds']
    return max(spec.minimum, min(spec.maximum, apply_rule(value, rule)))


@dataclass(frozen=True)
class DeckProblem:
    """An Age deck at a player count that leaves cards over (``error``) or changes the hands (``warning``)."""

    level: str
    age: int
    size: DeckSize
    base_hand: int
    expansions: Tuple[str, ...]  # the bundled expansions it was combined with (besides base)

    @property
    def combination(self) -> str:
        return ' + '.join(e.capitalize() for e in self.expansions) if self.expansions else 'base game'

    @property
    def where(self) -> str:
        return f'Age {NUMERALS[self.age]} at {self.size.players} players'

    def detail(self) -> str:
        """What is wrong, without the combination (the same in several combinations is told once)."""
        s = self.size
        if s.leftover:
            text = (
                f'{s.leftover} card(s) left over: {s.players - s.leftover} more or {s.leftover} fewer would deal evenly'
            )
        elif s.hand != self.base_hand:
            text = f'hands of {s.hand} instead of {self.base_hand}'
        else:
            text = 'deals evenly'
        if s.short_of_guilds:
            text += f'; it wants {s.guilds_wanted} guilds, the pool has {s.guild_pool}'
        return text

    def message(self) -> str:
        return f'{self.where} ({self.combination}): {self.size.cards} cards, {self.detail()}'


def check_decks(cards: Sequence[Card], extra_guilds: Optional[RuleValue] = None) -> List[DeckProblem]:
    """What an expansion's ``cards`` and ``extra_guilds`` rule do to every Age deck (see the module docstring).

    Each combination of base with the bundled expansions in :data:`DECK_EXPANSIONS` is checked, the
    expansion's rule applied after theirs, and compared with the same combination without it.
    """
    from wonders.cards.content import load_content

    problems: List[DeckProblem] = []
    for k in range(len(DECK_EXPANSIONS) + 1):
        for extra in combinations(DECK_EXPANSIONS, k):
            combo = ('base', *extra)
            official = load_content(*combo).cards
            ours = list(official) + list(cards)
            guilds_base = rules_of(combo).extra_guilds
            guilds_ours = extra_guilds_with(combo, extra_guilds)
            for age in AGES:
                for n in PLAYERS:
                    base = deck_size(official, age, n, guilds_base)
                    size = deck_size(ours, age, n, guilds_ours)
                    if size.leftover:
                        level = 'error'
                    elif size.hand != base.hand or (size.short_of_guilds and not base.short_of_guilds):
                        level = 'warning'
                    else:
                        continue
                    problems.append(DeckProblem(level, age, size, base.hand, tuple(extra)))
    return problems


def summary(problems: Sequence[DeckProblem], level: str, limit: int = 4) -> Optional[str]:
    """One line for the problems of ``level``, or ``None`` without any.

    The same finding in several combinations (base game, with Leaders, …) is told once, with the
    combinations it applies to; the first ``limit`` findings are told in full.
    """
    chosen = [p for p in problems if p.level == level]
    if not chosen:
        return None
    groups: Dict[Tuple[int, int, str], List[DeckProblem]] = {}
    for p in chosen:
        groups.setdefault((p.age, p.size.players, p.detail()), []).append(p)
    every = 2 ** len(DECK_EXPANSIONS)  # all combinations
    lines = []
    for (age, n, detail), group in sorted(groups.items(), key=lambda kv: kv[0][:2]):
        where = 'every combination' if len(group) == every else ', '.join(p.combination for p in group)
        lines.append(f'Age {NUMERALS[age]} at {n} players ({where}): {detail}')
    head = (
        'cards would be left over when dealing (every Age deck must split evenly into the hands)'
        if level == 'error'
        else 'hand sizes change'
    )
    more = f' (+{len(lines) - limit} more)' if len(lines) > limit else ''
    return f'{head}: ' + '; '.join(lines[:limit]) + more


def deck_table(cards: Sequence[Card], extra_guilds: Optional[RuleValue] = None) -> Dict[int, Dict[int, Dict[str, int]]]:
    """Every Age's deck with the base game: ``{age: {players: {cards, hand, leftover, base}}}`` (for the builder)."""
    from wonders.cards.content import load_content

    official = load_content('base').cards
    ours = list(official) + list(cards)
    guilds = extra_guilds_with(('base',), extra_guilds)
    table: Dict[int, Dict[int, Dict[str, int]]] = {}
    for age in AGES:
        table[age] = {}
        for n in PLAYERS:
            size = deck_size(ours, age, n, guilds)
            table[age][n] = {
                'cards': size.cards,
                'hand': size.hand,
                'leftover': size.leftover,
                'base': deck_size(official, age, n).cards,
                'guilds_wanted': size.guilds_wanted,
                'guild_pool': size.guild_pool,
            }
    return table


__all__ = ['DeckProblem', 'DeckSize', 'check_decks', 'deck_size', 'deck_table', 'extra_guilds_with', 'summary']
