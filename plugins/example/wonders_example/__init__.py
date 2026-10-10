"""An example plugin: a new effect type, a guild built from it (data) and an agent that wants that guild.

Install it next to the game (``pip install plugins/example``) and it shows up everywhere:
``wonders ext list``, ``wonders play -e base -e tithe -a taxman,heuristic,greedy`` and the GUI.
"""

from importlib import resources
from typing import Any, Dict

from wonders.agents.baselines import GreedyAgent
from wonders.agents.bench import AgentSpec
from wonders.effects.base import Context, Effect
from wonders.engine.actions import Action, ActionKind
from wonders.engine.observation import Observation

MANIFEST = {
    'id': 'example',
    'name': 'Example plugin',
    'author': 'Patrick Godau',
    'license': 'MIT',
    'description': 'The Tithe effect, the Tax Collectors Guild and the Taxman agent.',
    'rules_version': 2,
}
GUILD = 'Tax Collectors Guild'


class Tithe(Effect):
    """1 victory point per ``per`` coins the owner holds at the end of the game."""

    def __init__(self, per: int) -> None:
        self.per = per

    def victory_points(self, ctx: Context) -> int:
        return ctx.player.coins // self.per


def build_tithe(spec: Dict[str, Any]) -> Tithe:
    """Builder for ``{"type": "tithe", "per": 3}`` in card JSON."""
    return Tithe(int(spec.get('per', 3)))


class TaxmanAgent(GreedyAgent):
    """Greedy, but builds the Tax Collectors Guild whenever it can."""

    name = 'taxman'

    def decide(self, obs: Observation) -> Action:
        for action in obs.legal_actions:
            if action.kind is ActionKind.BUILD and action.card.name == GUILD:
                return action
        return super().decide(obs)


def register(registry: Any) -> None:
    """Called by the game once at start-up (entry point in ``pyproject.toml``)."""
    registry.register_effect('tithe', build_tithe)
    cards = resources.files('wonders_example').joinpath('cards.json')
    registry.register_expansion('tithe', 'Tithe', str(cards))
    registry.register_agent(AgentSpec.of('taxman', TaxmanAgent), blurb='Greedy, and loves the Tax Collectors Guild.')
