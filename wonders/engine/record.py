"""A replayable notation for games (like PGN for chess).

A :class:`GameRecord` captures only what is needed to reproduce a game with zero
ambiguity: the setup (players, wonders, and the *ordered* per-Age decks) plus every
player's decision, turn by turn. Everything else — trades, coin flows, military,
scores — is re-derived by :meth:`GameRecord.replay`, which re-simulates the recorded
decisions through the deterministic engine.

The decks are stored explicitly (not as an RNG seed) so replay does not depend on
the shuffle implementation or engine version — only on the rules staying stable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from wonders.cards.content import Content, build_age_deck, deal_hands
from wonders.engine.actions import Action, ActionKind
from wonders.engine.game import Game, Policy
from wonders.engine.phase import Phase
from wonders.engine.player import Player
from wonders.ruleset import Ruleset

_CODE = {ActionKind.BUILD: 'B', ActionKind.WONDER: 'W', ActionKind.DISCARD: 'D'}
_KIND = {code: kind for kind, code in _CODE.items()}


@dataclass
class PlayerSetup:
    """Which wonder (and side) a seat plays."""

    name: str
    wonder: str
    side: str = 'A'


# moves[age] -> list of turns; each turn -> per-seat [code, card_name]
Moves = Dict[int, List[List[List[str]]]]


@dataclass
class GameRecord:
    players: List[PlayerSetup]
    expansions: Tuple[str, ...]
    age_decks: Dict[int, List[str]]
    moves: Moves
    version: str = '1'

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        return {
            'version': self.version,
            'expansions': list(self.expansions),
            'players': [{'name': s.name, 'wonder': s.wonder, 'side': s.side} for s in self.players],
            'age_decks': {str(a): list(v) for a, v in self.age_decks.items()},
            'moves': {str(a): turns for a, turns in self.moves.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GameRecord':
        players = [PlayerSetup(**p) for p in d['players']]
        age_decks = {int(a): list(v) for a, v in d['age_decks'].items()}
        moves = {int(a): turns for a, turns in d['moves'].items()}
        return cls(players, tuple(d['expansions']), age_decks, moves, d.get('version', '1'))

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
        return '\n'.join(lines)

    # -- replay ------------------------------------------------------------
    def replay(self, content: Content) -> Game:
        """Re-simulate the recorded decisions and return the finished game."""
        coins = Ruleset.of(*self.expansions).starting_coins
        players = [Player(s.name, content.wonder(s.wonder, s.side), coins) for s in self.players]
        game = Game(players)
        for age in (1, 2, 3):
            deck = [content.card(name) for name in self.age_decks[age]]
            hands = deal_hands(deck, game.n)
            recorded = self.moves[age]
            cursor = {'turn': 0}

            def policy(g: Game, seat: int, hand: List, recorded=recorded, cursor=cursor) -> Action:
                code, name = recorded[cursor['turn']][seat]
                card = next(c for c in hand if c.name == name)
                return Action(_KIND[code], card)

            def advance(_age: int, _turn: int, _actions, cursor=cursor) -> None:
                cursor['turn'] += 1

            game.play_age(age, hands, policy, on_turn=advance)
        game.phase = Phase.FINISHED
        return game


class RecordBuilder:
    """Accumulates decks and moves during a live game into a :class:`GameRecord`."""

    def __init__(self, setups: Sequence[PlayerSetup], expansions: Tuple[str, ...]):
        self.setups = list(setups)
        self.expansions = expansions
        self.age_decks: Dict[int, List[str]] = {1: [], 2: [], 3: []}
        self.moves: Moves = {1: [], 2: [], 3: []}

    def set_deck(self, age: int, names: Sequence[str]) -> None:
        self.age_decks[age] = list(names)

    def add_turn(self, age: int, actions: Sequence[Action]) -> None:
        self.moves[age].append([[_CODE[a.kind], a.card.name] for a in actions])

    def build(self) -> GameRecord:
        return GameRecord(self.setups, self.expansions, self.age_decks, self.moves)


def play_and_record(
    content: Content,
    setups: Sequence[PlayerSetup],
    policy: Policy,
    rng,
    expansions: Tuple[str, ...] = ('base',),
    on_turn_extra: Optional[Callable[[int, int, List[Action]], None]] = None,
) -> Tuple[Game, GameRecord]:
    """Play a fresh game with ``policy`` and capture a :class:`GameRecord`.

    ``on_turn_extra`` (if given) is chained after recording each turn — used by the
    match runner to maintain an observation history.
    """
    coins = Ruleset.of(*expansions).starting_coins
    players = [Player(s.name, content.wonder(s.wonder, s.side), coins) for s in setups]
    game = Game(players)
    builder = RecordBuilder(setups, tuple(expansions))
    for age in (1, 2, 3):
        deck = build_age_deck(content, age, game.n, rng)
        builder.set_deck(age, [c.name for c in deck])
        hands = deal_hands(deck, game.n)

        def on_turn(a: int, t: int, actions: List[Action], _builder=builder) -> None:
            _builder.add_turn(a, actions)
            if on_turn_extra is not None:
                on_turn_extra(a, t, actions)

        game.play_age(age, hands, policy, on_turn=on_turn)
    game.phase = Phase.FINISHED
    return game, builder.build()
