"""Policy/value network over :class:`~wonders.agents.encoders.CardEncoder` features."""

from __future__ import annotations

from importlib import resources
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import nn

CHECKPOINT_VERSION = 1

# Bundled with the package: imitation of HeuristicAgent (3000 games) then 150 PPO
# iterations vs heuristic + greedy, via `wonders train imitate` / `wonders train ppo`.
PRETRAINED = 'ppo_base_3p.pt'


def pretrained_path(name: str = PRETRAINED) -> str:
    """Filesystem path of a checkpoint shipped in ``wonders/ml/pretrained``."""
    return str(resources.files('wonders.ml').joinpath('pretrained', name))


class PolicyValueNet(nn.Module):
    """An MLP with a masked policy head (one logit per action index) and a value head.

    Inputs are standardized with ``mean``/``std`` buffers (fitted from data with
    :meth:`set_normalization`; identity by default), because raw features mix coin
    counts, scores and one-hot flags on very different scales.
    """

    mean: torch.Tensor  # registered buffers (declared for type checkers)
    std: torch.Tensor

    def __init__(self, obs_size: int, n_actions: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.config: Dict[str, int] = {'obs_size': obs_size, 'n_actions': n_actions, 'hidden': hidden, 'layers': layers}
        trunk: List[nn.Module] = []
        width = obs_size
        for _ in range(layers):
            trunk += [nn.Linear(width, hidden), nn.ReLU()]
            width = hidden
        self.trunk = nn.Sequential(*trunk)
        self.policy = nn.Linear(width, n_actions)
        self.value = nn.Linear(width, 1)
        self.register_buffer('mean', torch.zeros(obs_size))
        self.register_buffer('std', torch.ones(obs_size))

    def set_normalization(self, observations: torch.Tensor) -> None:
        """Fit the input standardization to a batch of observations ``[N, obs_size]``."""
        self.mean.copy_(observations.mean(dim=0))
        self.std.copy_(observations.std(dim=0).clamp_min(1e-3))

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(masked logits [B, n_actions], value [B])``; illegal actions get -inf-like logits."""
        h = self.trunk((obs - self.mean) / self.std)
        logits = self.policy(h).masked_fill(~mask, torch.finfo(torch.float32).min)
        return logits, self.value(h).squeeze(-1)


def save_checkpoint(path: str, model: PolicyValueNet, card_names: Sequence[str], **meta: Any) -> None:
    """Save weights, architecture and the card vocabulary the action space was built on."""
    torch.save(
        {
            'version': CHECKPOINT_VERSION,
            'config': model.config,
            'state_dict': model.state_dict(),
            'card_names': list(card_names),
            'meta': meta,
        },
        path,
    )


def load_checkpoint(path: str) -> Tuple[PolicyValueNet, List[str], Dict[str, Any]]:
    """Load a checkpoint saved by :func:`save_checkpoint`: ``(model, card_names, meta)``."""
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data.get('version') != CHECKPOINT_VERSION:
        raise ValueError(f'unsupported checkpoint version {data.get("version")!r} in {path}')
    model = PolicyValueNet(**data['config'])
    model.load_state_dict(data['state_dict'])
    model.eval()
    return model, list(data['card_names']), dict(data.get('meta', {}))
