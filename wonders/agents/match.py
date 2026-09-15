"""Run a game between agents, capturing a replayable record.

The match runner is the bridge between the :class:`~wonders.agents.base.Agent`
interface and the engine: each turn it builds every seat's Observation (including a
running public history), asks that seat's agent to decide, and records the decision.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from wonders.agents.base import Agent
from wonders.cards.content import Content
from wonders.engine.actions import Action
from wonders.engine.game import Game
from wonders.engine.observation import build_observation
from wonders.engine.record import GameRecord, PlayerSetup, play_and_record


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

    # Raw public history: list of (age, turn, [(ActionKind, card_name), ...]).
    history: List[Tuple[int, int, List]] = []

    def policy(game: Game, seat: int, hand) -> Action:
        obs = build_observation(game, seat, hand, history)
        return agents[seat].decide(obs)

    def on_turn_extra(age: int, turn: int, actions: List[Action]) -> None:
        history.append((age, turn, [(a.kind, a.card.name) for a in actions]))

    return play_and_record(content, setups, policy, rng, expansions, on_turn_extra)
