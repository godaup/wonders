"""Run a game between agents, capturing a replayable record.

The match runner is the bridge between the :class:`~wonders.agents.base.Agent`
interface and the engine: each turn it builds every seat's Observation (including a
running public history), asks that seat's agent to decide, and records the decision.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from wonders.agents.base import Agent
from wonders.cards.content import Content, deal_leaders
from wonders.engine.actions import Action
from wonders.engine.game import Game
from wonders.engine.observation import ObservationBuilder
from wonders.engine.record import GameRecord, PlayerSetup, play_and_record
from wonders.ruleset import Ruleset


def play_match(
    content: Content,
    setups: Sequence[PlayerSetup],
    agents: Sequence[Agent],
    rng,
    expansions: Tuple[str, ...] = ('base',),
) -> Tuple[Game, GameRecord]:
    """Play ``agents`` (seat-aligned with ``setups``) and return the game + record.

    With the Leaders expansion, the leader draft and each Age's recruitment are routed
    to :meth:`Agent.draft_leader` / :meth:`Agent.recruit` via :meth:`Agent.act`.
    """
    if len(agents) != len(setups):
        raise ValueError('need one agent per seat')

    # Running public history + per-decision shared table, filtered per viewer on demand.
    observations = ObservationBuilder()
    draft = deal_leaders(content, len(setups), rng) if Ruleset.of(*expansions).uses('leaders') else None

    def policy(game: Game, seat: int, hand) -> Action:
        return agents[seat].act(observations.build(game, seat, hand), hand)

    def on_turn_extra(age: int, turn: int, actions: Sequence[Optional[Action]]) -> None:
        observations.record(age, turn, [None if a is None else (a.kind, a.card.name) for a in actions])

    return play_and_record(content, setups, policy, rng, expansions, on_turn_extra, leader_draft=draft)
