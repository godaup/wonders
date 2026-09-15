"""Agents and the state representations they consume."""

from wonders.agents.base import Agent
from wonders.agents.baselines import GreedyAgent, RandomAgent
from wonders.agents.encoders import DictEncoder, Encoder, FlatEncoder
from wonders.agents.match import play_match
