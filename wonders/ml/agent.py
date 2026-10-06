"""An :class:`~wonders.agents.base.Agent` driven by a trained :class:`PolicyValueNet`."""

from __future__ import annotations

import os
import random
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from wonders.agents.base import Agent
from wonders.agents.encoders import DECISIONS
from wonders.cards.base import Card
from wonders.engine.actions import Action
from wonders.engine.observation import Observation
from wonders.env.vocab import Vocab
from wonders.ml.model import PolicyValueNet, load_checkpoint


@lru_cache(maxsize=32)
def _shared_checkpoint(path: str, device: str) -> Tuple[PolicyValueNet, Tuple[str, ...], Dict[str, Any]]:
    """One copy per checkpoint and device per process (agents only run inference, so they can share it)."""
    model, names, meta = load_checkpoint(path)
    return model.to(device).eval(), tuple(names), meta


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
        decisions: The model's decision kinds (stored in checkpoints; defaults to the base kinds).
        vocab: The full vocabulary (incl. encoder version) for an in-memory ``model``.
        sample: Sample from the policy instead of taking the most likely action.
        rng: Random source for sampling (seeded for reproducibility).
        name: Display name.
        device: Torch device for inference (e.g. ``'cuda'`` to take batch-1 inference of large
            networks off the CPU). Checkpoints are loaded once per process and device and shared.
        leader_policy: Let the network make the Leaders draft and recruitment decisions too
            (as during PPO training) when its checkpoint knows those decision kinds. ``False``
            falls back to the :class:`~wonders.agents.base.Agent` hooks (costliest leader),
            which is how leaders were played before; kept for comparisons.
    """

    name = 'neural'

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        *,
        model: Optional[PolicyValueNet] = None,
        card_names: Optional[Sequence[str]] = None,
        decisions: Optional[Sequence[str]] = None,
        vocab: Optional[Vocab] = None,
        sample: bool = False,
        rng: Optional[random.Random] = None,
        name: str = 'neural',
        leader_policy: bool = True,
        device: str = 'cpu',
    ):
        meta: Dict[str, Any] = {'decisions': list(decisions or DECISIONS)}
        if checkpoint is not None:
            model, loaded_names, meta = _shared_checkpoint(os.path.abspath(checkpoint), device)
            card_names = list(loaded_names)
        if model is None or card_names is None:
            raise ValueError('give a checkpoint path, or both model and card_names')
        self.device = torch.device(device)
        self.model = model.eval() if checkpoint is not None else model.eval().to(self.device)
        self.vocab = vocab or Vocab.from_checkpoint(card_names, meta)
        self.action_space = self.vocab.action_space()
        self.encoder = self.vocab.encoder()
        if self.model.config['obs_size'] != self.encoder.size or self.model.config['n_actions'] != self.action_space.n:
            raise ValueError('model shape does not match the card vocabulary')
        self.sample = sample
        self.rng = rng or random.Random()
        self.name = name
        self.leader_policy = leader_policy and {'draft', 'recruit'} <= set(self.vocab.decisions)

    def act(self, obs: Observation, options: List[Card]) -> Action:
        if self.leader_policy:
            return self.decide(obs)  # every decision kind is in the action space
        return super().act(obs, options)

    def logits(self, obs: Observation) -> torch.Tensor:
        """Masked policy logits for ``obs`` (shape ``[n_actions]``)."""
        x = torch.tensor([self.encoder.encode(obs)], dtype=torch.float32, device=self.device)
        mask = torch.tensor([self.action_space.mask(obs.legal_actions)], device=self.device)
        with torch.inference_mode():
            logits, _value = self.model(x, mask)
        return logits[0].cpu()

    def decide(self, obs: Observation) -> Action:
        logits = self.logits(obs)
        if self.sample:
            legal: List[int] = [self.action_space.index(a) for a in obs.legal_actions]
            probs = torch.softmax(logits[legal], dim=0).tolist()
            index = self.rng.choices(legal, weights=probs)[0]
        else:
            index = int(torch.argmax(logits))
        return self.action_space.to_action(index, obs.legal_actions)
