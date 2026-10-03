"""An :class:`~wonders.agents.base.Agent` driven by a trained :class:`PolicyValueNet`."""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

import torch

from wonders.agents.base import Agent
from wonders.agents.encoders import CardEncoder
from wonders.engine.actions import Action
from wonders.engine.observation import Observation
from wonders.env.action_space import ActionSpace
from wonders.ml.model import PolicyValueNet, load_checkpoint


class NeuralAgent(Agent):
    """Plays the policy of a trained network (greedy by default).

    Build it from a checkpoint path (picklable, so it works in ``run_bench`` worker
    processes) or from an in-memory ``model`` plus the ``card_names`` vocabulary it was
    trained on. The observation must only contain cards from that vocabulary (i.e. the
    same expansions as in training).

    Args:
        checkpoint: Path to a checkpoint from :func:`~wonders.ml.model.save_checkpoint`.
        model: An in-memory network (alternative to ``checkpoint``).
        card_names: The model's card vocabulary (required with ``model``).
        sample: Sample from the policy instead of taking the most likely action.
        rng: Random source for sampling (seeded for reproducibility).
        name: Display name.
    """

    name = 'neural'

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        *,
        model: Optional[PolicyValueNet] = None,
        card_names: Optional[Sequence[str]] = None,
        sample: bool = False,
        rng: Optional[random.Random] = None,
        name: str = 'neural',
    ):
        if checkpoint is not None:
            model, loaded_names, _meta = load_checkpoint(checkpoint)
            card_names = loaded_names
        if model is None or card_names is None:
            raise ValueError('give a checkpoint path, or both model and card_names')
        self.model = model.eval()
        self.action_space = ActionSpace(card_names)
        self.encoder = CardEncoder(self.action_space.card_names)
        if self.model.config['obs_size'] != self.encoder.size or self.model.config['n_actions'] != self.action_space.n:
            raise ValueError('model shape does not match the card vocabulary')
        self.sample = sample
        self.rng = rng or random.Random()
        self.name = name

    def logits(self, obs: Observation) -> torch.Tensor:
        """Masked policy logits for ``obs`` (shape ``[n_actions]``)."""
        x = torch.tensor([self.encoder.encode(obs)], dtype=torch.float32)
        mask = torch.tensor([self.action_space.mask(obs.legal_actions)])
        with torch.inference_mode():
            logits, _value = self.model(x, mask)
        return logits[0]

    def decide(self, obs: Observation) -> Action:
        logits = self.logits(obs)
        if self.sample:
            legal: List[int] = [self.action_space.index(a) for a in obs.legal_actions]
            probs = torch.softmax(logits[legal], dim=0).tolist()
            index = self.rng.choices(legal, weights=probs)[0]
        else:
            index = int(torch.argmax(logits))
        return self.action_space.to_action(index, obs.legal_actions)
