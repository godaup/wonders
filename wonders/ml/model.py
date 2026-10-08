"""Policy/value network over :class:`~wonders.agents.encoders.CardEncoder` features."""

from __future__ import annotations

from importlib import resources
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

CHECKPOINT_VERSION = 1

# Bundled with the package: the universal model of training Stage 3 (``sweeps/STAGE3.md``) —
# 256x2 over the table-v2 encoder, every expansion and player count; imitation of the adaptive
# agent, then 40M PPO steps against a tiered opponent mix (scripted / recent self / anchors),
# continued for 10M steps with a linearly decaying learning rate (``s5-anneal``, ``sweeps/OBJECTIVES.md``).
PRETRAINED = 'ppo_universal_v2.pt'


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

    def set_normalization(
        self, observations: torch.Tensor, rows: Optional[torch.Tensor] = None, chunk: int = 65536
    ) -> None:
        """Fit the input standardization to observations ``[N, obs_size]`` (or only ``rows`` of them).

        Accumulates float64 sums chunk by chunk, so large datasets are never copied whole.
        """
        index = torch.arange(len(observations)) if rows is None else rows
        total = torch.zeros(observations.shape[1], dtype=torch.float64)
        squares = torch.zeros_like(total)
        for part in index.split(chunk):
            x = observations[part].double()
            total += x.sum(dim=0)
            squares += (x * x).sum(dim=0)
        n = len(index)
        mean = total / n
        var = (squares - n * mean * mean) / max(1, n - 1)  # unbiased, like Tensor.std
        self.mean.copy_(mean.float())
        self.std.copy_(var.clamp_min(0).sqrt().float().clamp_min(1e-3))

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(masked logits [B, n_actions], value [B])``; illegal actions get -inf-like logits."""
        h = self.trunk((obs - self.mean) / self.std)
        logits = self.policy(h).masked_fill(~mask, torch.finfo(torch.float32).min)
        return logits, self.value(h).squeeze(-1)


def save_checkpoint(path: str, model: PolicyValueNet, card_names: Sequence[str], **meta: Any) -> None:
    """Save weights, architecture and the card vocabulary the action space was built on.

    Pass the vocabulary's ``**vocab.meta()`` (decisions, wonders, encoder). Without an ``encoder``
    entry the checkpoint loads as the legacy ``card-v1`` encoder, so that is only allowed for a
    model of exactly that shape — anything else would save a checkpoint that cannot be loaded.

    The checkpoint records the engine's :data:`~wonders.ruleset.RULES_VERSION` (``rules_version``)
    unless ``meta`` gives one, e.g. when re-saving an older model unchanged.
    """
    from wonders.ruleset import RULES_VERSION

    meta.setdefault('rules_version', RULES_VERSION)
    if 'encoder' not in meta:
        from wonders.agents.encoders import DECISIONS, CardEncoder

        legacy = CardEncoder(card_names, meta.get('decisions', DECISIONS)).size
        if model.config['obs_size'] != legacy:
            raise ValueError("save_checkpoint needs the vocabulary's meta (**vocab.meta()) for this model")
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


def rules_version(meta: Dict[str, Any]) -> int:
    """The rules version a checkpoint was trained under (checkpoints without one predate versioning)."""
    from wonders.ruleset import LEGACY_RULES_VERSION

    return int(meta.get('rules_version', LEGACY_RULES_VERSION))


def load_checkpoint(path: str) -> Tuple[PolicyValueNet, List[str], Dict[str, Any]]:
    """Load a checkpoint saved by :func:`save_checkpoint`: ``(model, card_names, meta)``."""
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data.get('version') != CHECKPOINT_VERSION:
        raise ValueError(f'unsupported checkpoint version {data.get("version")!r} in {path}')
    model = PolicyValueNet(**data['config'])
    model.load_state_dict(data['state_dict'])
    model.eval()
    return model, list(data['card_names']), dict(data.get('meta', {}))
