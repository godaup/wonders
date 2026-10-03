"""Learned agents: a policy/value network, imitation learning and PPO.

Optional: requires PyTorch (``pip install 'wonders[ml]'``). The rest of the package
never imports this module, so the core engine stays dependency-free.
"""

try:
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise ImportError("wonders.ml needs PyTorch: pip install 'wonders[ml]'") from exc

from wonders.ml.agent import NeuralAgent
from wonders.ml.model import PolicyValueNet, load_checkpoint, pretrained_path, save_checkpoint
