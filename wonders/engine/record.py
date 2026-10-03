"""A replayable notation for games (like PGN for chess).

A :class:`GameRecord` captures only what is needed to reproduce a game with zero
ambiguity: the setup (players, wonders, and the *ordered* per-Age decks) plus every
player's decision, turn by turn. Everything else — trades, coin flows, military,
scores — is re-derived by :meth:`GameRecord.replay`, which re-simulates the recorded
decisions through the deterministic engine.

Extra decisions granted by wonder powers (Babylon B's 7th card, Halicarnassus'
discard build) are kept in ``extras``, one ``[turn, kind, seat, code, card]`` entry per
decision in the order the engine asked for them.

The decks are stored explicitly (not as an RNG seed) so replay does not depend on
the shuffle implementation or engine version — only on the rules staying stable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from wonders.cards.base import Card
from wonders.cards.content import Content, build_age_deck, deal_hands
from wonders.engine.actions import Action, ActionKind, DecisionKind, DecisionPoint
from wonders.engine.game import Game, Policy, drive
from wonders.engine.phase import Phase
from wonders.engine.player import Player
from wonders.ruleset import Ruleset

# A per-seat recruit decision at the start of an Age: (game, seat) -> Action | None.
RecruitFn = Callable[[Game, int], Optional[Action]]

_CODE = {ActionKind.BUILD: 'B', ActionKind.WONDER: 'W', ActionKind.DISCARD: 'D', ActionKind.FREE: 'F'}
_KIND = {code: kind for kind, code in _CODE.items()}


@dataclass
class PlayerSetup:
    """Which wonder (and side) a seat plays."""

    name: str
    wonder: str
    side: str = 'A'


# moves[age] -> list of turns; each turn -> per-seat [code, card_name]
Moves = Dict[int, List[List[List[str]]]]
# extras[age] -> ordered [turn, decision_kind, seat, code, card_name] entries
Extras = Dict[int, List[List[Any]]]


class ReplayError(ValueError):
    """The record does not match what the engine asks for (corrupt or outdated record)."""


@dataclass
class GameRecord:
    players: List[PlayerSetup]
    expansions: Tuple[str, ...]
    age_decks: Dict[int, List[str]]
    moves: Moves
    version: str = '2'
    # Leaders expansion (empty for base games): the 4 leaders each seat drafted, and
    # the per-Age recruitment play (per-seat [code, leader_name] or None).
    leader_hands: Dict[int, List[str]] = field(default_factory=dict)
    recruits: Dict[int, List[Optional[List[str]]]] = field(default_factory=dict)
    extras: Extras = field(default_factory=dict)

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            'version': self.version,
            'expansions': list(self.expansions),
            'players': [{'name': s.name, 'wonder': s.wonder, 'side': s.side} for s in self.players],
            'age_decks': {str(a): list(v) for a, v in self.age_decks.items()},
            'moves': {str(a): turns for a, turns in self.moves.items()},
        }
        if self.leader_hands:
            out['leader_hands'] = {str(i): names for i, names in self.leader_hands.items()}
            out['recruits'] = {str(a): plays for a, plays in self.recruits.items()}
        if any(self.extras.values()):
            out['extras'] = {str(a): entries for a, entries in self.extras.items()}
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GameRecord':
        players = [PlayerSetup(**p) for p in d['players']]
        age_decks = {int(a): list(v) for a, v in d['age_decks'].items()}
        moves = {int(a): turns for a, turns in d['moves'].items()}
        leader_hands = {int(i): list(v) for i, v in d.get('leader_hands', {}).items()}
        recruits = {int(a): plays for a, plays in d.get('recruits', {}).items()}
        extras = {int(a): entries for a, entries in d.get('extras', {}).items()}
        return cls(
            players, tuple(d['expansions']), age_decks, moves, d.get('version', '1'), leader_hands, recruits, extras
        )

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> 'GameRecord':
        return cls.from_dict(json.loads(text))

    def text(self) -> str:
        """A compact, human-readable transcript."""
        lines = [
            f"# 7 Wonders ({', '.join(self.expansions)})",
            'Players: ' + ', '.join(f'{s.name} [{s.wonder}/{s.side}]' for s in self.players),
        ]
        for age in (1, 2, 3):
            lines.append(f'Age {age}:')
            for t, turn in enumerate(self.moves.get(age, []), 1):
                parts = [f'{self.players[seat].name} {code}:{name}' for seat, (code, name) in enumerate(turn)]
                lines.append(f'  {t}. ' + ' | '.join(parts))
                for turn_idx, kind, seat, code, name in self.extras.get(age, []):
                    if turn_idx == t - 1:
                        lines.append(f'     + {self.players[seat].name} {code}:{name} ({kind})')
        return '\n'.join(lines)

    # -- replay ------------------------------------------------------------
    def replay(self, content: Content) -> Game:
        """Re-simulate the recorded decisions and return the finished game."""
        coins = Ruleset.of(*self.expansions).starting_coins
        players = [Player(s.name, content.wonder(s.wonder, s.side), coins) for s in self.players]
        game = Game(players)
        for seat, names in self.leader_hands.items():
            players[seat].leaders_hand = [content.card(n) for n in names]
        for age in (1, 2, 3):
            _apply_recruits(game, self.recruits.get(age))
            deck = [content.card(name) for name in self.age_decks[age]]
            hands = deal_hands(deck, game.n)
            drive(game.age_steps(age, hands), self._chooser(age))
        game.phase = Phase.FINISHED
        return game

    def _chooser(self, age: int) -> Callable[[DecisionPoint], Dict[int, Action]]:
        """Answer the engine's decision points for ``age`` from the recorded moves."""
        extras = iter(self.extras.get(age, []))

        def pick(options: List[Card], code: str, name: str) -> Action:
            card = next((c for c in options if c.name == name), None)
            if card is None:
                raise ReplayError(f'Age {age}: recorded card {name!r} is not among the options')
            return Action(_KIND[code], card)

        def choose(point: DecisionPoint) -> Dict[int, Action]:
            if point.kind is DecisionKind.PLAY:
                row = self.moves[age][point.turn]
                return {seat: pick(opts, *row[seat]) for seat, opts in point.options.items()}
            actions = {}
            for seat, opts in point.options.items():
                entry = next(extras, None)
                if entry is None or entry[:3] != [point.turn, point.kind.value, seat]:
                    raise ReplayError(f'Age {age}: no recorded {point.kind.value} decision for seat {seat}')
                actions[seat] = pick(opts, entry[3], entry[4])
            return actions

        return choose


def _apply_recruits(game: Game, plays: Optional[List[Optional[List[str]]]]) -> None:
    """Apply one Age's recorded leader recruitment (per-seat play or None)."""
    if not plays:
        return
    for seat, entry in enumerate(plays):
        if entry is None:
            continue
        code, name = entry
        player = game.players[seat]
        card = next(c for c in player.leaders_hand if c.name == name)
        player.leaders_hand.remove(card)
        game.apply(seat, Action(_KIND[code], card))


class RecordBuilder:
    """Accumulates decks and moves during a live game into a :class:`GameRecord`."""

    def __init__(self, setups: Sequence[PlayerSetup], expansions: Tuple[str, ...]):
        self.setups = list(setups)
        self.expansions = expansions
        self.age_decks: Dict[int, List[str]] = {1: [], 2: [], 3: []}
        self.moves: Moves = {1: [], 2: [], 3: []}
        self.leader_hands: Dict[int, List[str]] = {}
        self.recruits: Dict[int, List[Optional[List[str]]]] = {}
        self.extras: Extras = {1: [], 2: [], 3: []}

    def set_deck(self, age: int, names: Sequence[str]) -> None:
        self.age_decks[age] = list(names)

    def add_turn(self, age: int, actions: Sequence[Action]) -> None:
        self.moves[age].append([[_CODE[a.kind], a.card.name] for a in actions])

    def add_extra(self, point: DecisionPoint, actions: Dict[int, Action]) -> None:
        for seat in sorted(actions):
            a = actions[seat]
            self.extras[point.age].append([point.turn, point.kind.value, seat, _CODE[a.kind], a.card.name])

    def set_leader_hands(self, hands: Dict[int, List[Card]]) -> None:
        self.leader_hands = {i: [c.name for c in cards] for i, cards in hands.items()}

    def add_recruits(self, age: int, actions: Sequence[Optional[Action]]) -> None:
        self.recruits[age] = [None if a is None else [_CODE[a.kind], a.card.name] for a in actions]

    def build(self) -> GameRecord:
        return GameRecord(
            self.setups,
            self.expansions,
            self.age_decks,
            self.moves,
            leader_hands=self.leader_hands,
            recruits=self.recruits,
            extras=self.extras,
        )


def play_and_record(
    content: Content,
    setups: Sequence[PlayerSetup],
    policy: Policy,
    rng,
    expansions: Tuple[str, ...] = ('base',),
    on_turn_extra: Optional[Callable[[int, int, Sequence[Optional[Action]]], None]] = None,
    leader_hands: Optional[Dict[int, List[Card]]] = None,
    recruit: Optional[RecruitFn] = None,
) -> Tuple[Game, GameRecord]:
    """Play a fresh game with ``policy`` and capture a :class:`GameRecord`.

    ``on_turn_extra`` (if given) is chained after recording each turn — used by the
    match runner to maintain an observation history. It also sees the extra decisions
    granted by wonder powers, as a per-seat list with ``None`` for seats not acting. ``leader_hands`` (already
    drafted, per seat) and ``recruit`` (called before each Age) drive the Leaders
    recruitment phase.
    """
    coins = Ruleset.of(*expansions).starting_coins
    players = [Player(s.name, content.wonder(s.wonder, s.side), coins) for s in setups]
    game = Game(players)
    builder = RecordBuilder(setups, tuple(expansions))
    if leader_hands:
        for seat, cards in leader_hands.items():
            players[seat].leaders_hand = list(cards)
        builder.set_leader_hands(leader_hands)
    for age in (1, 2, 3):
        if recruit is not None:  # recruitment phase precedes the Age's card play
            plays = [recruit(game, seat) for seat in range(game.n)]
            for seat, action in enumerate(plays):
                if action is not None:
                    players[seat].leaders_hand.remove(action.card)
                    game.apply(seat, action)
            builder.add_recruits(age, plays)
        deck = build_age_deck(content, age, game.n, rng)
        builder.set_deck(age, [c.name for c in deck])
        hands = deal_hands(deck, game.n)

        def on_turn(a: int, t: int, actions: List[Action], _builder=builder) -> None:
            _builder.add_turn(a, actions)
            if on_turn_extra is not None:
                on_turn_extra(a, t, actions)

        def on_extra(point: DecisionPoint, actions: Dict[int, Action], _builder=builder) -> None:
            _builder.add_extra(point, actions)
            if on_turn_extra is not None:
                on_turn_extra(point.age, point.turn, [actions.get(i) for i in range(game.n)])

        game.play_age(age, hands, policy, on_turn=on_turn, on_extra=on_extra)
    game.phase = Phase.FINISHED
    return game, builder.build()
