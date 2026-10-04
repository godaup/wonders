"""Imitation learning: clone a teacher agent's decisions into a :class:`PolicyValueNet`.

:func:`collect` plays seeded games (optionally across processes) and records, for each
decision a *teacher* seat makes, the encoded observation, the legal-action mask, the
teacher's action index and a value target. :func:`train_imitation` fits the policy head
with cross-entropy on the teacher's actions and the value head on that target.

The value target is the seat's *remaining* score margin: final ``(own VP − mean
opponent VP)`` minus the same margin at decision time (computed from the public
table). That is exactly the return of the env's ``'dense'`` reward, so a cloned
network starts PPO with a useful critic. Targets are multiplied by ``VALUE_SCALE``.
"""

from __future__ import annotations

import random
from array import array
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from wonders._mp import get_context, init_worker
from wonders.agents.base import Agent
from wonders.agents.bench import AGENTS, AgentSpec, _content
from wonders.agents.match import play_match
from wonders.cards.base import Card
from wonders.engine.actions import Action
from wonders.engine.observation import Observation
from wonders.engine.record import PlayerSetup
from wonders.env.action_space import ActionSpace
from wonders.env.vocab import VectorEncoder, Vocab, vocab_for
from wonders.ml.model import PolicyValueNet
from wonders.ruleset import Ruleset

VALUE_SCALE = 0.1  # VP margins (~±30) -> roughly unit-scale value targets / rewards

# Default teacher setups: all-heuristic tables plus heuristic against varied baselines.
DEFAULT_LINEUPS: Tuple[Tuple[str, ...], ...] = (
    ('heuristic', 'heuristic', 'heuristic'),
    ('heuristic', 'heuristic', 'heuristic', 'heuristic'),
    ('heuristic', 'greedy', 'builder'),
    ('heuristic', 'heuristic', 'greedy', 'civic', 'scientist'),
)


def _margin(obs: Observation) -> float:
    others = [c.score for c in obs.table if c.seat != obs.seat]
    return obs.me.score - sum(others) / len(others)


class _Recorder(Agent):
    """Wraps a teacher agent and records each decision it makes (draft and recruit included)."""

    def __init__(self, teacher: Agent, encoder: VectorEncoder, space: ActionSpace):
        self.teacher = teacher
        self.encoder = encoder
        self.space = space
        self.name = teacher.name
        self.rows: List[Tuple[List[float], List[bool], int, float]] = []

    def _record(self, obs: Observation, action: Action) -> Action:
        self.rows.append(
            (self.encoder.encode(obs), self.space.mask(obs.legal_actions), self.space.index(action), _margin(obs))
        )
        return action

    def decide(self, obs: Observation) -> Action:
        return self._record(obs, self.teacher.decide(obs))

    def act(self, obs: Observation, options: List[Card]) -> Action:
        return self._record(obs, self.teacher.act(obs, options))


@dataclass
class Dataset:
    """Recorded teacher decisions as tensors."""

    obs: torch.Tensor  # [N, obs_size] float32
    mask: torch.Tensor  # [N, n_actions] bool
    action: torch.Tensor  # [N] int64
    value: torch.Tensor  # [N] float32 (scaled remaining margin)

    def __len__(self) -> int:
        return int(self.action.shape[0])


# One collection table: (expansions, agent names per seat).
Table = Tuple[Tuple[str, ...], Tuple[str, ...]]


def _collect_game(
    job: Tuple[int, int, Tuple[str, ...], Tuple[AgentSpec, ...], Tuple[str, ...], Vocab],
) -> Tuple[bytes, bytes, List[int], List[float]]:
    index, seed, expansions, lineup, teachers, vocab = job
    expansions = Ruleset.of(*expansions).content_expansions
    content = _content(expansions)
    space = vocab.action_space()
    encoder = vocab.encoder()
    rng = random.Random(seed * 1_000_003 + index)
    n = len(lineup)
    seats = [lineup[(i + index) % n] for i in range(n)]
    names = list(dict.fromkeys(w.name for w in content.wonders))
    setups = [
        PlayerSetup(f'{s.name}-{i}', w, rng.choice('AB')) for i, (s, w) in enumerate(zip(seats, rng.sample(names, n)))
    ]
    agents: List[Agent] = []
    recorders: Dict[int, _Recorder] = {}
    for i, spec in enumerate(seats):
        agent = spec.build(random.Random(rng.random()))
        if spec.name in teachers:
            recorders[i] = _Recorder(agent, encoder, space)
            agent = recorders[i]
        agents.append(agent)
    game, _record = play_match(content, setups, agents, rng=rng, expansions=expansions)
    finals = [game.score(i).total for i in range(n)]
    obs_buf, mask_buf = array('f'), bytearray()
    actions: List[int] = []
    values: List[float] = []
    for i, rec in recorders.items():
        final_margin = finals[i] - (sum(finals) - finals[i]) / (n - 1)
        for x, m, a, margin in rec.rows:
            obs_buf.extend(x)
            mask_buf.extend(bytes(m))
            actions.append(a)
            values.append((final_margin - margin) * VALUE_SCALE)
    return obs_buf.tobytes(), bytes(mask_buf), actions, values


def collect(
    n_games: int,
    lineups: Sequence[Sequence[str]] = DEFAULT_LINEUPS,
    teachers: Sequence[str] = ('heuristic',),
    seed: int = 0,
    processes: int = 1,
    tables: Optional[Sequence[Table]] = None,
    vocab: Optional[Vocab] = None,
) -> Dataset:
    """Play ``n_games`` and record every decision a ``teachers`` seat makes.

    Games cycle through ``tables`` — ``(expansions, lineup)`` pairs, any expansion
    combination and player count — or, by default, through base-game ``lineups``.
    Observations are encoded in ``vocab`` (default: the union of the tables' expansions).
    """
    if tables is None:
        tables = [(('base',), tuple(lineup)) for lineup in lineups]
    if vocab is None:
        vocab = vocab_for(tuple(sorted({e for expansions, _ in tables for e in expansions})))
    space = vocab.action_space()
    obs_size = vocab.obs_size
    specs = [(tuple(expansions), tuple(AGENTS[name] for name in lineup)) for expansions, lineup in tables]
    jobs = [(g, seed, *specs[g % len(specs)], tuple(teachers), vocab) for g in range(n_games)]
    if processes > 1:
        with get_context().Pool(processes, initializer=init_worker) as pool:
            parts = pool.map(_collect_game, jobs, chunksize=max(1, n_games // (processes * 8)))
    else:
        parts = [_collect_game(job) for job in jobs]
    obs = torch.frombuffer(bytearray(b''.join(p[0] for p in parts)), dtype=torch.float32).view(-1, obs_size)
    mask = torch.frombuffer(bytearray(b''.join(p[1] for p in parts)), dtype=torch.uint8).view(-1, space.n).bool()
    action = torch.tensor([a for p in parts for a in p[2]], dtype=torch.int64)
    value = torch.tensor([v for p in parts for v in p[3]], dtype=torch.float32)
    return Dataset(obs.clone(), mask, action, value)


def new_model(hidden: int = 256, layers: int = 2, vocab: Optional[Vocab] = None) -> Tuple[PolicyValueNet, List[str]]:
    """A fresh network sized for ``vocab`` (default: the base game), plus its card names."""
    vocab = vocab or vocab_for(('base',))
    model = PolicyValueNet(vocab.obs_size, vocab.n_actions, hidden=hidden, layers=layers)
    return model, list(vocab.card_names)


def train_imitation(
    model: PolicyValueNet,
    data: Dataset,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    value_coef: float = 0.5,
    val_fraction: float = 0.1,
    seed: int = 0,
    log: Optional[Callable[[str], None]] = print,
    device: str = 'auto',
) -> Dict[str, float]:
    """Fit ``model`` to ``data``; returns final train loss and held-out accuracy / value MSE."""
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(data), generator=gen)
    n_val = max(1, int(len(data) * val_fraction))
    val, train = order[:n_val], order[n_val:]
    model.set_normalization(data.obs[train])
    target = torch.device(
        'cuda' if device == 'auto' and torch.cuda.is_available() else ('cpu' if device == 'auto' else device)
    )
    model.to(target)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    metrics: Dict[str, float] = {}
    for epoch in range(epochs):
        model.train()
        perm = train[torch.randperm(len(train), generator=gen)]
        total, batches = 0.0, 0
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            obs, mask = data.obs[idx].to(target), data.mask[idx].to(target)
            action, value_target = data.action[idx].to(target), data.value[idx].to(target)
            logits, value = model(obs, mask)
            loss = ce(logits, action) + value_coef * nn.functional.mse_loss(value, value_target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            batches += 1
        metrics = {'train_loss': total / max(1, batches), **evaluate(model, data, val, target)}
        if log is not None:
            log(
                f'epoch {epoch + 1}/{epochs}: loss {metrics["train_loss"]:.3f}  '
                f'val acc {metrics["val_accuracy"]:.3f}  val value mse {metrics["val_value_mse"]:.3f}'
            )
    model.cpu()
    model.eval()
    return metrics


def evaluate(
    model: PolicyValueNet, data: Dataset, idx: torch.Tensor, device: Optional[torch.device] = None
) -> Dict[str, float]:
    """Accuracy of the greedy policy vs the teacher, and value MSE, on rows ``idx``."""
    model.eval()
    device = device or next(model.parameters()).device
    correct, squared, count = 0.0, 0.0, 0
    with torch.inference_mode():
        for chunk in idx.split(8192):
            logits, value = model(data.obs[chunk].to(device), data.mask[chunk].to(device))
            correct += float((logits.argmax(dim=1).cpu() == data.action[chunk]).sum())
            squared += float(((value.cpu() - data.value[chunk]) ** 2).sum())
            count += len(chunk)
    return {'val_accuracy': correct / max(1, count), 'val_value_mse': squared / max(1, count)}
