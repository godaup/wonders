"""A replayable notation for games (like PGN for chess).

A :class:`GameRecord` captures only what is needed to reproduce a game with zero
ambiguity: the setup (players, wonders, and the *ordered* per-Age decks) plus every
player's decision, turn by turn. Everything else — trades, coin flows, military,
scores — is re-derived by :meth:`GameRecord.replay`, which re-simulates the recorded
decisions through the deterministic engine.

Extra decisions granted by wonder powers (Babylon B's 7th card, Halicarnassus'
discard build) are kept in ``extras``, one ``[turn, kind, seat, code, card]`` entry per
decision in the order the engine asked for them.

A move on which the player chose how to split a purchase between the neighbours (see
:meth:`~wonders.engine.actions.Action.paying`) carries the coins paid as two more numbers,
``[code, card, left, right]``; moves without them paid the cheapest way.

The decks are stored explicitly (not as an RNG seed) so replay does not depend on
the shuffle implementation or engine version — only on the rules staying stable.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from wonders.cards.base import Card
from wonders.cards.content import Content, build_age_deck, deal_hands
from wonders.engine.actions import Action, ActionKind, DecisionKind, DecisionPoint
from wonders.engine.game import AgeSteps, Game, Policy, drive
from wonders.engine.observation import Observation, ObservationBuilder
from wonders.engine.phase import Phase
from wonders.engine.player import Player
from wonders.ruleset import Ruleset

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
    leader_box: List[str] = field(default_factory=list)  # undealt leaders, drawn from the end (Rome B)
    recruits: Dict[int, List[Optional[List[str]]]] = field(default_factory=dict)
    extras: Extras = field(default_factory=dict)
    # Extensions behind the expansions in play: {id: {version, hash}} (empty for bundled content).
    extensions: Dict[str, Dict[str, str]] = field(default_factory=dict)

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
            out['leader_box'] = list(self.leader_box)
            out['recruits'] = {str(a): plays for a, plays in self.recruits.items()}
        if any(self.extras.values()):
            out['extras'] = {str(a): entries for a, entries in self.extras.items()}
        if self.extensions:
            out['extensions'] = self.extensions
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
            players,
            tuple(d['expansions']),
            age_decks,
            moves,
            version=d.get('version', '1'),
            leader_hands=leader_hands,
            leader_box=list(d.get('leader_box', [])),
            recruits=recruits,
            extras=extras,
            extensions=dict(d.get('extensions', {})),
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
                parts = [f'{self.players[seat].name} {move[0]}:{move[1]}' for seat, move in enumerate(turn)]
                lines.append(f'  {t}. ' + ' | '.join(parts))
                for turn_idx, kind, seat, code, name, *_split in self.extras.get(age, []):
                    if turn_idx == t - 1:
                        lines.append(f'     + {self.players[seat].name} {code}:{name} ({kind})')
        return '\n'.join(lines)

    # -- replay ------------------------------------------------------------
    def check_extensions(self) -> List[str]:
        """Differences between the extensions this game was played with and those installed now."""
        from wonders.extensions import ensure_loaded

        installed = ensure_loaded().extensions
        problems = []
        for ext_id, pin in self.extensions.items():
            ext = installed.get(ext_id)
            if ext is None:
                problems.append(f'extension {ext_id!r} (version {pin.get("version")}) is not installed')
            elif ext.digest != pin.get('hash'):
                problems.append(
                    f'extension {ext_id!r} differs: recorded {pin.get("version")} ({pin.get("hash")}), '
                    f'installed {ext.manifest.version} ({ext.digest})'
                )
        return problems

    def replay(self, content: Content) -> Game:
        """Re-simulate the recorded decisions and return the finished game."""
        for problem in self.check_extensions():
            warnings.warn(f'replay may differ: {problem}', stacklevel=2)
        coins = Ruleset.of(*self.expansions).starting_coins
        players = [Player(s.name, content.wonder(s.wonder, s.side), coins) for s in self.players]
        game = Game(players, expansions=self.expansions)
        for seat, names in self.leader_hands.items():
            players[seat].leaders_hand = [content.card(n) for n in names]
        game.leader_box = [content.card(n) for n in self.leader_box]
        for age in (1, 2, 3):
            choose = self._chooser(age)  # one chooser: recruitment extras precede the Age's
            if self.leader_hands:
                drive(game.recruit_steps(age), choose)
            deck = [content.card(name) for name in self.age_decks[age]]
            hands = deal_hands(deck, game.n)
            drive(game.age_steps(age, hands), choose)
        game.phase = Phase.FINISHED
        return game

    def replay_observations(self, content: Content, seat: int) -> Iterator[Tuple[DecisionPoint, Observation, Action]]:
        """Re-simulate the game and yield every decision of ``seat``: the point, its observation, the move.

        The observation is built exactly as during play (the same public history, filtered for
        ``seat``), so an agent or an analysis sees what the player saw. The Leaders draft is not
        part of a record (only the drafted hands are), so it is not yielded; recruitment is.
        """
        game, builder = self._start(content)

        def relay(steps: AgeSteps, choose: Callable[[DecisionPoint], Dict[int, Action]]):
            try:
                point = next(steps)
                while True:
                    actions = choose(point)
                    if seat in point.options:
                        yield point, builder.build(game, seat, point.options[seat]), actions[seat]
                    _log(builder, game, point, actions)
                    point = steps.send(actions)
            except StopIteration:
                return

        for age in (1, 2, 3):
            choose = self._chooser(age)
            if self.leader_hands:
                yield from relay(game.recruit_steps(age), choose)
            deck = [content.card(name) for name in self.age_decks[age]]
            yield from relay(game.age_steps(age, deal_hands(deck, game.n)), choose)
        game.phase = Phase.FINISHED

    def branch(
        self,
        content: Content,
        seat: int,
        index: int,
        action: Action,
        policy: Callable[[Game, int, Observation, List[Card]], Action],
        rng: Optional[Any] = None,
    ) -> Game:
        """Replay up to ``seat``'s decision ``index``, play ``action`` there instead, then let ``policy`` play.

        ``index`` counts the seat's decisions as :meth:`replay_observations` yields them; ``action``
        is matched by kind and card name. The other seats keep their recorded moves of that same
        (simultaneous) turn; from the next decision on ``policy(game, seat, observation, options)``
        plays every seat. ``rng`` reshuffles the decks of the Ages after the branch point (each
        keeps its recorded cards), so repeated branches explore different deals. Returns the
        finished game.
        """
        game, builder = self._start(content)
        mine = 0
        diverged = False

        def relay(steps: AgeSteps, choose: Callable[[DecisionPoint], Dict[int, Action]]) -> None:
            nonlocal mine, diverged
            try:
                point = next(steps)
                while True:
                    if diverged:
                        actions = {
                            i: policy(game, i, builder.build(game, i, opts), opts) for i, opts in point.options.items()
                        }
                    else:
                        actions = choose(point)
                        if seat in point.options:
                            if mine == index:
                                card = next(c for c in point.options[seat] if c.name == action.card.name)
                                actions[seat] = Action(action.kind, card)
                                diverged = True
                            mine += 1
                    _log(builder, game, point, actions)
                    point = steps.send(actions)
            except StopIteration:
                return

        for age in (1, 2, 3):
            choose = self._chooser(age)
            if self.leader_hands:
                relay(game.recruit_steps(age), choose)
            names = list(self.age_decks[age])
            if diverged and rng is not None:
                rng.shuffle(names)
            relay(game.age_steps(age, deal_hands([content.card(n) for n in names], game.n)), choose)
        game.phase = Phase.FINISHED
        if not diverged:
            raise ValueError(f'seat {seat} has no decision #{index} in this record')
        return game

    def _start(self, content: Content) -> Tuple[Game, ObservationBuilder]:
        """A fresh game with the recorded setup (players, leaders, leader box) and an empty history."""
        coins = Ruleset.of(*self.expansions).starting_coins
        players = [Player(s.name, content.wonder(s.wonder, s.side), coins) for s in self.players]
        game = Game(players, expansions=self.expansions)
        for i, names in self.leader_hands.items():
            players[i].leaders_hand = [content.card(n) for n in names]
        game.leader_box = [content.card(n) for n in self.leader_box]
        return game, ObservationBuilder()

    def _chooser(self, age: int) -> Callable[[DecisionPoint], Dict[int, Action]]:
        """Answer the engine's decision points for ``age`` from the recorded moves."""
        extras = iter(self.extras.get(age, []))

        def pick(options: List[Card], move: Sequence[Any]) -> Action:
            code, name, split = move[0], move[1], move[2:]
            card = next((c for c in options if c.name == name), None)
            if card is None:
                raise ReplayError(f'Age {age}: recorded card {name!r} is not among the options')
            action = Action(_KIND[code], card)
            return action.paying(int(split[0]), int(split[1])) if len(split) == 2 else action

        def choose(point: DecisionPoint) -> Dict[int, Action]:
            if point.kind is DecisionKind.PLAY:
                row = self.moves[age][point.turn]
                return {seat: pick(opts, row[seat]) for seat, opts in point.options.items()}
            if point.kind is DecisionKind.RECRUIT:
                plays = self.recruits.get(age, [])
                actions = {}
                for seat, opts in point.options.items():
                    entry = plays[seat] if seat < len(plays) else None
                    if entry is None:
                        raise ReplayError(f'Age {age}: no recorded recruitment for seat {seat}')
                    actions[seat] = pick(opts, entry)
                return actions
            actions = {}
            for seat, opts in point.options.items():
                entry = next(extras, None)
                if entry is None or entry[:3] != [point.turn, point.kind.value, seat]:
                    raise ReplayError(f'Age {age}: no recorded {point.kind.value} decision for seat {seat}')
                actions[seat] = pick(opts, entry[3:])
            return actions

        return choose


def _log(builder: ObservationBuilder, game: Game, point: DecisionPoint, actions: Dict[int, Action]) -> None:
    """Log an answered decision in the public history, as live play does (recruitment is not logged)."""
    if point.kind is DecisionKind.PLAY:
        builder.record(point.age, point.turn, [(a.kind, a.card.name) for _i, a in sorted(actions.items())])
    elif point.kind is not DecisionKind.RECRUIT:  # extras are public events
        moves = [(actions[i].kind, actions[i].card.name) if i in actions else None for i in range(game.n)]
        builder.record(point.age, point.turn, moves)


def _move(action: Action) -> List[Any]:
    """``[code, card]``, plus ``left, right`` coins when the player chose how to split a purchase."""
    move: List[Any] = [_CODE[action.kind], action.card.name]
    if action.payment is not None:
        move += list(action.payment)
    return move


class RecordBuilder:
    """Accumulates decks and moves during a live game into a :class:`GameRecord`."""

    def __init__(self, setups: Sequence[PlayerSetup], expansions: Tuple[str, ...]):
        self.setups = list(setups)
        self.expansions = expansions
        self.age_decks: Dict[int, List[str]] = {1: [], 2: [], 3: []}
        self.moves: Moves = {1: [], 2: [], 3: []}
        self.leader_hands: Dict[int, List[str]] = {}
        self.leader_box: List[str] = []
        self.recruits: Dict[int, List[Optional[List[str]]]] = {}
        self.extras: Extras = {1: [], 2: [], 3: []}

    def set_deck(self, age: int, names: Sequence[str]) -> None:
        self.age_decks[age] = list(names)

    def add_turn(self, age: int, actions: Sequence[Action]) -> None:
        self.moves[age].append([_move(a) for a in actions])

    def add_extra(self, point: DecisionPoint, actions: Dict[int, Action]) -> None:
        for seat in sorted(actions):
            self.extras[point.age].append([point.turn, point.kind.value, seat, *_move(actions[seat])])

    def set_leader_box(self, box: Sequence[Card]) -> None:
        self.leader_box = [c.name for c in box]

    def set_leader_hands(self, hands: Dict[int, List[Card]]) -> None:
        self.leader_hands = {i: [c.name for c in cards] for i, cards in hands.items()}

    def add_recruits(self, age: int, actions: Sequence[Optional[Action]]) -> None:
        self.recruits[age] = [None if a is None else [_CODE[a.kind], a.card.name] for a in actions]

    def build(self) -> GameRecord:
        from wonders.extensions import REGISTRY

        return GameRecord(
            self.setups,
            self.expansions,
            self.age_decks,
            self.moves,
            leader_hands=self.leader_hands,
            leader_box=self.leader_box,
            recruits=self.recruits,
            extras=self.extras,
            extensions=REGISTRY.pins(self.expansions),
        )


def play_and_record(
    content: Content,
    setups: Sequence[PlayerSetup],
    policy: Policy,
    rng,
    expansions: Tuple[str, ...] = ('base',),
    on_turn_extra: Optional[Callable[[int, int, Sequence[Optional[Action]]], None]] = None,
    leader_draft: Optional[List[List[Card]]] = None,
    leader_box: Sequence[Card] = (),
) -> Tuple[Game, GameRecord]:
    """Play a fresh game with ``policy`` and capture a :class:`GameRecord`.

    ``on_turn_extra`` (if given) is chained after recording each turn — used by the
    match runner to maintain an observation history. It also sees the extra decisions
    granted by wonder powers, as a per-seat list with ``None`` for seats not acting.

    ``leader_draft`` (the dealt leader hands, see :func:`~wonders.cards.content.deal_leaders`)
    turns on the Leaders phases: ``policy`` drafts (``game.decision`` is ``DRAFT``) and,
    before each Age, recruits (``RECRUIT``) — the same coroutines the env drives.
    ``leader_box`` holds the undealt leaders (Rome B draws from it).
    """
    coins = Ruleset.of(*expansions).starting_coins
    players = [Player(s.name, content.wonder(s.wonder, s.side), coins) for s in setups]
    game = Game(players, expansions=tuple(expansions))
    builder = RecordBuilder(setups, tuple(expansions))

    def choose(point: DecisionPoint) -> Dict[int, Action]:
        return {seat: policy(game, seat, options) for seat, options in point.options.items()}

    def on_extra(point: DecisionPoint, actions: Dict[int, Action]) -> None:
        builder.add_extra(point, actions)
        if on_turn_extra is not None:
            on_turn_extra(point.age, point.turn, [actions.get(i) for i in range(game.n)])

    if leader_draft is not None:
        game.leader_box = list(leader_box)
        builder.set_leader_box(leader_box)
        drive(game.draft_steps(leader_draft), choose)
        builder.set_leader_hands({i: list(p.leaders_hand) for i, p in enumerate(players)})
    for age in (1, 2, 3):
        if leader_draft is not None:  # recruitment phase precedes the Age's card play
            recruited: Dict[int, Action] = {}

            def recruit(point: DecisionPoint, _recruited=recruited) -> Dict[int, Action]:
                actions = choose(point)
                if point.kind is DecisionKind.RECRUIT:
                    _recruited.update(actions)
                else:  # a power resolved during recruitment (Solomon, Rome B, Courtesans)
                    on_extra(point, actions)
                return actions

            drive(game.recruit_steps(age), recruit)
            builder.add_recruits(age, [recruited.get(i) for i in range(game.n)])
        deck = build_age_deck(content, age, game.n, rng)
        builder.set_deck(age, [c.name for c in deck])
        hands = deal_hands(deck, game.n)

        def on_turn(a: int, t: int, actions: List[Action], _builder=builder) -> None:
            _builder.add_turn(a, actions)
            if on_turn_extra is not None:
                on_turn_extra(a, t, actions)

        game.play_age(age, hands, policy, on_turn=on_turn, on_extra=on_extra)
    game.phase = Phase.FINISHED
    return game, builder.build()
