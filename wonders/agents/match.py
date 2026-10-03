"""Run a game between agents, capturing a replayable record.

The match runner is the bridge between the :class:`~wonders.agents.base.Agent`
interface and the engine: each turn it builds every seat's Observation (including a
running public history), asks that seat's agent to decide, and records the decision.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from wonders.agents.base import Agent
from wonders.cards.base import Card
from wonders.cards.content import Content
from wonders.engine.actions import Action
from wonders.engine.game import Game
from wonders.engine.observation import ObservationBuilder
from wonders.engine.record import GameRecord, PlayerSetup, play_and_record
from wonders.ruleset import Ruleset

LEADERS_PER_PLAYER = 4


def _draft_leaders(content: Content, agents: Sequence[Agent], rng, n: int) -> Dict[int, List[Card]]:
    """Deal 4 leaders each and pick-and-pass to build each seat's leader hand."""
    deck = list(content.leaders())
    rng.shuffle(deck)
    hands = [[deck.pop() for _ in range(LEADERS_PER_PLAYER)] for _ in range(n)]
    kept: List[List[Card]] = [[] for _ in range(n)]
    for _ in range(LEADERS_PER_PLAYER):
        picks = [agents[i].draft_leader(hands[i]) for i in range(n)]
        for i, pick in enumerate(picks):
            hands[i].remove(pick)
            kept[i].append(pick)
        hands = [hands[(i - 1) % n] for i in range(n)]  # pass hands to the left
    return {i: kept[i] for i in range(n)}


def play_match(
    content: Content,
    setups: Sequence[PlayerSetup],
    agents: Sequence[Agent],
    rng,
    expansions: Tuple[str, ...] = ('base',),
) -> Tuple[Game, GameRecord]:
    """Play ``agents`` (seat-aligned with ``setups``) and return the game + record."""
    if len(agents) != len(setups):
        raise ValueError('need one agent per seat')
    n = len(setups)

    # Running public history + per-decision shared table, filtered per viewer on demand.
    observations = ObservationBuilder()

    leader_hands: Optional[Dict[int, List[Card]]] = None
    if Ruleset.of(*expansions).uses('leaders'):
        leader_hands = _draft_leaders(content, agents, rng, n)

    def recruit(game: Game, seat: int) -> Optional[Action]:
        leaders = game.players[seat].leaders_hand
        if not leaders:
            return None
        obs = observations.build(game, seat, leaders)
        return agents[seat].recruit(leaders, obs)

    def policy(game: Game, seat: int, hand) -> Action:
        obs = observations.build(game, seat, hand)
        return agents[seat].decide(obs)

    def on_turn_extra(age: int, turn: int, actions: Sequence[Optional[Action]]) -> None:
        observations.record(age, turn, [None if a is None else (a.kind, a.card.name) for a in actions])

    return play_and_record(
        content,
        setups,
        policy,
        rng,
        expansions,
        on_turn_extra,
        leader_hands,
        recruit if leader_hands else None,
    )
