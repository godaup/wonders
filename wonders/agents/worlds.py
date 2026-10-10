"""Rebuild playable games from one seat's point of view (determinization, for search).

A search agent only gets an :class:`~wonders.engine.observation.Observation`. To look ahead it
needs a full :class:`~wonders.engine.game.Game`, so :func:`sample_world` rebuilds one: every
public part (cities, coins, tokens, wonder stages, history) comes straight from the
observation, and every hidden part is *sampled* consistently with what the seat has seen:

* the opponents' current hands and the cards they discarded or buried this Age, drawn from the
  Age's deck composition minus the cards the seat knows were played (the cards it saw in earlier
  hands, ``seen_cards``, are always placed among them);
* Age III's guild draw and the Cities black-card draw (known cards always included);
* the discard pile's unknown cards, from earlier Ages' cards nobody built;
* the opponents' unrecruited leaders and the undealt leader box.

Because only the observation is read, the sampled world cannot leak hidden information.
Future Ages are dealt freshly by :meth:`World.steps` from the world's own random source.

Known simplifications (kept deliberately; they only blur the sample): seen cards are spread
uniformly over the hidden slots (not traced to the hand they were passed to), opponents' leader
hands have the size the seat's own would have, and free builds from the discard pile count
against an Olympia allowance like free builds from hand.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Generator, List, Optional, Sequence, Tuple

from wonders.cards.base import Card, Color
from wonders.cards.content import (
    GUILDS_OVER_PLAYERS,
    LEADERS_PER_PLAYER,
    Content,
    build_age_deck,
    deal_hands,
    load_content,
)
from wonders.engine.actions import Action, ActionKind, DecisionKind, DecisionPoint
from wonders.engine.game import Game
from wonders.engine.observation import MoveEvent, Observation, ObservationBuilder
from wonders.engine.phase import Phase
from wonders.engine.player import Player
from wonders.ruleset import Ruleset


class ContentIndex:
    """Name lookups and per-Age deck compositions for one expansion set (cached per process)."""

    def __init__(self, content: Content):
        self.content = content
        self.cards: Dict[str, Card] = {}
        self.by_age: Dict[Tuple[str, Optional[int]], Card] = {}  # a few names exist in two Ages
        for card in content.cards:
            self.cards.setdefault(card.name, card)
            self.by_age.setdefault((card.name, card.age), card)
        self.leaders = [c for c in content.cards if c.color is Color.WHITE]
        self.guilds = content.guilds()

    def card(self, name: str, age: Optional[int] = None) -> Card:
        """The card called ``name`` (of ``age`` when that Age has one: Press, Loom ... are in two)."""
        return self.by_age.get((name, age)) or self.cards[name]

    def fixed(self, age: int, n_players: int) -> Counter:
        """The deterministic part of an Age's deck (no guilds / black cards), as name counts."""
        out: Counter = Counter()
        for card in self.content.cards:
            if card.guild or card.color in (Color.WHITE, Color.BLACK) or card.age != age:
                continue
            out[card.name] += card.copies_for(n_players)
        return +out

    def black(self, age: int) -> List[Card]:
        return [c for c in self.content.cards if c.color is Color.BLACK and c.age == age]


@lru_cache(maxsize=16)
def content_index(expansions: Tuple[str, ...]) -> ContentIndex:
    """The :class:`ContentIndex` for the expansions of a game (``Observation.expansions``)."""
    return ContentIndex(load_content(*Ruleset.of(*expansions).content_expansions))


@dataclass
class World:
    """A rebuilt game at the observed decision, with sampled hidden information.

    ``hands`` are every seat's current hand (the viewer's real one, the others sampled);
    ``observations`` replays the public history so observations of any seat can be built.
    Drive :meth:`steps` like :meth:`Game.age_steps`, and feed every answer to :meth:`record`
    so later observations see the moves.
    """

    game: Game
    hands: List[List[Card]]
    observations: ObservationBuilder
    index: ContentIndex
    rng: random.Random

    def steps(self) -> Generator[DecisionPoint, Dict[int, Action], None]:
        """The rest of the game from the observed turn: this Age, then fresh decks for later Ages."""
        game = self.game
        leaders = Ruleset.of(*game.expansions).uses('leaders')
        yield from game.age_steps(game.age, self.hands, first_turn=game.turn)
        for age in range(game.age + 1, 4):
            if leaders:
                yield from game.recruit_steps(age)
            deck = build_age_deck(self.index.content, age, game.n, self.rng, game.rules.extra_guilds)
            yield from game.age_steps(age, deal_hands(deck, game.n))
        game.phase = Phase.FINISHED

    def record(self, point: DecisionPoint, actions: Dict[int, Action]) -> None:
        """Log an answered decision in the public history (recruitment is not logged, as in the env)."""
        if point.kind in (DecisionKind.RECRUIT, DecisionKind.DRAFT):
            return
        moves = [(actions[i].kind, actions[i].card.name) if i in actions else None for i in range(self.game.n)]
        self.observations.record(point.age, point.turn, moves)


def sample_world(obs: Observation, rng: random.Random) -> World:
    """Rebuild the game at a regular-turn (``'play'``) observation, sampling what ``obs.seat`` cannot see."""
    if obs.decision != DecisionKind.PLAY.value:
        raise ValueError(f'worlds are rebuilt at regular turns, not at {obs.decision!r} decisions')
    index = content_index(tuple(obs.expansions))
    n, me, age = obs.n_players, obs.seat, obs.age
    game = _public_game(obs, index)

    # -- this Age's hidden cards: opponents' hands and their hidden plays --------------------
    this_age = [e for e in obs.history if e.age == age]
    hidden = [e for e in this_age if e.card is None]
    hand = [index.card(c.name, age) for c in obs.hand]
    known = Counter(c.name for c in hand) + Counter(e.card for e in this_age if e.card is not None)
    seen = Counter(obs.seen_cards)
    unknown = _age_deck(index, age, n, known + seen, rng, game.rules.extra_guilds) - known
    slots = (n - 1) * len(hand) + len(hidden)
    pick = list((seen & unknown).elements())
    rest = list((unknown - seen).elements())
    rng.shuffle(rest)
    pick += rest[: max(0, slots - len(pick))]
    while len(pick) < slots:  # composition and history disagree (rare edge cases): pad plausibly
        pick.append(rng.choice(list(index.fixed(age, n).elements())))
    pick = pick[:slots]
    rng.shuffle(pick)
    hands: List[List[Card]] = []
    for seat in range(n):
        if seat == me:
            hands.append(hand)
        else:
            hands.append([index.card(pick.pop(), age) for _ in range(len(hand))])
    sampled = {id(e): pick.pop() for e in hidden}

    # -- the discard pile ---------------------------------------------------------------------
    pile = [e.card for e in obs.history if e.seat == me and e.kind == 'discard' and e.card is not None]
    pile += [sampled[id(e)] for e in hidden if e.kind == 'discard']
    missing = obs.discard_count - len(pile)
    if missing > 0:
        pile += _old_discards(obs, index, missing, rng)
    game.discard_pile = [index.card(name) for name in pile[: obs.discard_count]]

    # -- leaders ------------------------------------------------------------------------------
    if Ruleset.of(*obs.expansions).uses('leaders'):
        _sample_leaders(obs, index, game, rng)

    builder = ObservationBuilder()
    builder.history = _raw_history(obs.history, n, sampled)
    # Seed the viewer's card counting so its seen_cards carry over into the world.
    own_plays = [e.card for e in this_age if e.seat == me and e.card is not None]
    builder._hands[(me, age)] = {-1: list(obs.seen_cards) + own_plays}
    return World(game, hands, builder, index, rng)


def _public_game(obs: Observation, index: ContentIndex) -> Game:
    """A game whose cities, coins and tokens match ``obs.table``."""
    players = []
    for city in obs.table:
        player = Player(city.name, index.content.wonder(city.wonder, city.wonder_side), city.coins)
        for name in city.cards:
            player.add_card(index.card(name))
        player.entombed = [index.card(name) for name in city.entombed]
        for name in city.copied_leaders:
            player.add_copied_leader(index.card(name))
        player.tokens = list(city.tokens)
        player.debt = city.debt
        player.diplomacy = city.diplomacy
        players.append(player)
    game = Game(players, expansions=tuple(obs.expansions))  # resolves copied stages (Manneken Pis)
    for player, city in zip(players, obs.table):
        # The buried cards stay hidden: a stage scoring its buried card's colour (Stonehenge B) is rebuilt
        # without it, as its owner's opponents see it.
        built = city.stage_slots or [[k, 0] for k in range(city.stages_built)]
        for slot, variant in built:
            player.add_stage(player.slots[slot][variant], slot, variant=variant)
    game.age, game.turn, game.phase = obs.age, obs.turn, Phase.for_age(obs.age)
    for seat, player in enumerate(players):
        allowance = player.free_builds_left  # nothing used yet
        if seat == obs.seat:
            player.free_builds_used = max(0, allowance - obs.free_builds_left)
        elif allowance:
            used = sum(1 for e in obs.history if e.age == obs.age and e.seat == seat and e.kind == 'free')
            player.free_builds_used = min(allowance, used)
    return game


def _age_deck(
    index: ContentIndex, age: int, n: int, include: Counter, rng: random.Random, extra_guilds: int = GUILDS_OVER_PLAYERS
) -> Counter:
    """A plausible full deck for ``age`` (name counts) whose random draws contain ``include``."""
    deck = index.fixed(age, n)
    if age == 3:
        deck.update(c.name for c in _draw(index.guilds, n + extra_guilds, include, rng))
    deck.update(c.name for c in _draw(index.black(age), n, include, rng))
    return deck


def _draw(pool: Sequence[Card], want: int, include: Counter, rng: random.Random) -> List[Card]:
    """``want`` cards of ``pool``: every card named in ``include`` plus a random rest."""
    if want >= len(pool):
        return list(pool)
    must = [c for c in pool if c.name in include]
    others = [c for c in pool if c.name not in include]
    return must + rng.sample(others, max(0, min(len(others), want - len(must))))


def _old_discards(obs: Observation, index: ContentIndex, count: int, rng: random.Random) -> List[str]:
    """Unknown discards: cards of earlier Ages (fixed deck parts) that nobody built."""
    pool: Counter = Counter()
    for age in range(1, obs.age):
        pool += index.fixed(age, obs.n_players)
    pool -= Counter(name for city in obs.table for name in city.cards)
    names = list(pool.elements()) or list(index.fixed(max(1, obs.age), obs.n_players).elements())
    return [rng.choice(names) for _ in range(count)]


def _sample_leaders(obs: Observation, index: ContentIndex, game: Game, rng: random.Random) -> None:
    """Opponents' unrecruited leaders and the leader box, from the leaders nobody has shown."""
    public = {name for city in obs.table for name in city.cards} | set(obs.leaders_hand)
    pool = [c for c in index.leaders if c.name not in public]
    rng.shuffle(pool)
    size = max(0, LEADERS_PER_PLAYER - obs.age)
    for seat, player in enumerate(game.players):
        if seat == obs.seat:
            player.leaders_hand = [index.card(name) for name in obs.leaders_hand]
        else:
            player.leaders_hand = [pool.pop() for _ in range(min(size, len(pool)))]
    game.leader_box = pool


def _raw_history(events: Sequence[MoveEvent], n: int, sampled: Dict[int, str]) -> List[Tuple[int, int, List]]:
    """Regroup the viewer's filtered events into raw ``(age, turn, moves)`` entries.

    Hidden plays of this Age get their sampled card (so a world's opponent knows what it
    discarded); older hidden plays stay hidden (``None``) for everyone.
    """
    out: List[Tuple[int, int, List]] = []
    moves: List[Optional[Tuple[ActionKind, Optional[str]]]] = []
    key: Optional[Tuple[int, int]] = None
    last_seat = -1
    for e in events:  # an entry lists its seats in order, so a repeated seat starts the next one
        if (e.age, e.turn) != key or e.seat <= last_seat:
            if key is not None:
                out.append((key[0], key[1], moves))
            key, moves = (e.age, e.turn), [None] * n
        moves[e.seat] = (ActionKind(e.kind), e.card if e.card is not None else sampled.get(id(e)))
        last_seat = e.seat
    if key is not None:
        out.append((key[0], key[1], moves))
    return out
