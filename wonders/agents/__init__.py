"""Agents and the state representations they consume."""

from wonders.agents.base import Agent
from wonders.agents.baselines import (
    ArchetypeAgent,
    BuilderAgent,
    CivicAgent,
    DiscardAgent,
    EconomistAgent,
    GreedyAgent,
    MilitaristAgent,
    RandomAgent,
    ScientistAgent,
)
from wonders.agents.encoders import DictEncoder, Encoder, FlatEncoder
from wonders.agents.heuristic import HeuristicAgent
from wonders.agents.match import play_match
from wonders.agents.strategies import BalancedAgent, BlockerAgent, OpportunistAgent, TableAgent, default_card_values
