"""Expert iteration: distil network-guided search back into the network.

:class:`~wonders.ml.search.SearchAgent` plays clearly better than the network it searches with
(see ``ROADMAP.md``), so it can serve as a teacher:

1. :func:`collect_search` plays games in which one seat (rotating) searches and the others play
   the plain network. For every regular turn of the searching seat it records the observation and
   a *soft* policy target over the searched candidates — ``softmax(value / temperature)`` of their
   values in VP — so the network learns how much better each candidate was, not only which one
   won. Other decisions (Leaders draft and recruitment, power picks) are recorded as the
   network's own choice. The value target is the remaining score margin, as in
   :mod:`~wonders.ml.imitation`.
2. :func:`distill` fine-tunes the network on those targets (cross-entropy against the soft
   targets plus value regression), keeping its input normalisation.

Repeat with the distilled network as the new searcher while head-to-head keeps improving.
Targets are stored sparsely (candidate indices and probabilities), so datasets stay small.
"""

from __future__ import annotations

import math
import os
import random
from array import array
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn

from wonders._mp import get_context, init_worker
from wonders.agents.base import Agent
from wonders.agents.bench import AGENTS, _content
from wonders.agents.match import play_match
from wonders.cards.base import Card
from wonders.engine.actions import Action
from wonders.engine.observation import Observation
from wonders.engine.record import PlayerSetup
from wonders.ml.agent import NeuralAgent
from wonders.ml.imitation import VALUE_SCALE, _margin
from wonders.ml.model import PolicyValueNet
from wonders.ml.search import SearchAgent
from wonders.ruleset import Ruleset

# One collection cell: (expansions, player count).
Cell = Tuple[Tuple[str, ...], int]


@dataclass
class SearchDataset:
    """Search decisions as tensors; ``targets`` holds up to ``K`` candidates per row (``-1`` = unused)."""

    obs: torch.Tensor  # [N, obs_size] float32
    mask: torch.Tensor  # [N, n_actions] bool
    targets: torch.Tensor  # [N, K] int64 action indices
    probs: torch.Tensor  # [N, K] float32 target probabilities (0 where unused)
    value: torch.Tensor  # [N] float32 (scaled remaining margin)
    # [N, K] float32 the search's candidate values in VP (NaN where unused; one 0 for unsearched rows),
    # so the target temperature can be changed after collection. None for older datasets.
    scores: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return int(self.value.shape[0])

    def with_temperature(self, temperature: float) -> 'SearchDataset':
        """The same decisions with targets ``softmax(scores / temperature)`` (needs ``scores``)."""
        if self.scores is None:
            raise ValueError('this dataset has no candidate scores; collect it again to change the temperature')
        used = ~torch.isnan(self.scores)
        logits = torch.where(used, self.scores / max(1e-6, temperature), torch.full_like(self.scores, -torch.inf))
        probs = torch.softmax(logits, dim=1)
        return SearchDataset(self.obs, self.mask, self.targets, probs, self.value, self.scores)

    def save(self, path: str) -> None:
        torch.save(self.__dict__, path)

    @classmethod
    def load(cls, path: str) -> 'SearchDataset':
        return cls(**torch.load(path, weights_only=True))


def soft_targets(values: Sequence[float], temperature: float) -> List[float]:
    """``softmax(values / temperature)``: candidate values in VP to target probabilities."""
    top = max(values)
    weights = [math.exp((v - top) / max(1e-6, temperature)) for v in values]
    total = sum(weights)
    return [w / total for w in weights]


class _SearchRecorder(Agent):
    """Plays a :class:`SearchAgent` and records every decision with its (soft) target."""

    def __init__(self, search: SearchAgent, temperature: float, width: int):
        self.search = search
        self.temperature = temperature
        self.width = width
        self.name = search.name
        self.rows: List[Tuple[List[float], List[bool], List[int], List[float], List[float], float]] = []

    def act(self, obs: Observation, options: List[Card]) -> Action:
        action = self.search.act(obs, options)
        net = self.search.net
        space = net.action_space
        if self.search.last is not None:
            candidates, values = self.search.last
            indices = [space.index(a) for a in candidates]
            probs = soft_targets(values, self.temperature)
            scores = list(values)
        else:
            indices, probs, scores = [space.index(action)], [1.0], [0.0]
        pad = self.width - len(indices)
        self.rows.append(
            (
                net.encoder.encode(obs),
                space.mask(obs.legal_actions),
                indices + [-1] * pad,
                probs + [0.0] * pad,
                scores + [math.nan] * pad,
                _margin(obs),
            )
        )
        return action

    def decide(self, obs: Observation) -> Action:
        return self.act(obs, [])


def _collect_game(
    job: Tuple[Any, ...],
) -> Tuple[bytes, bytes, List[List[int]], List[List[float]], List[List[float]], List[float]]:
    index, seed, expansions, n, checkpoint, search_kwargs, temperature, opponents = job
    content = _content(Ruleset.of(*expansions).content_expansions)
    rng = random.Random(seed * 1_000_003 + index)
    names = list(dict.fromkeys(w.name for w in content.wonders))
    setups = [PlayerSetup(f'P{i}', w, rng.choice('AB')) for i, w in enumerate(rng.sample(names, n))]
    me = index % n  # the searching seat rotates
    search = SearchAgent(checkpoint, rng=random.Random(rng.random()), name='search', **search_kwargs)
    recorder = _SearchRecorder(search, temperature, int(search_kwargs.get('candidates', 4)))
    agents: List[Agent] = []
    for seat in range(n):
        if seat == me:
            agents.append(recorder)
        else:
            name = opponents[(index + seat) % len(opponents)]
            agents.append(
                NeuralAgent(checkpoint) if name == 'neural' else AGENTS[name].build(random.Random(rng.random()))
            )
    game, _record = play_match(content, setups, agents, rng=rng, expansions=expansions)
    finals = [game.score(i).total for i in range(n)]
    final_margin = finals[me] - (sum(finals) - finals[me]) / (n - 1)
    obs_buf, mask_buf = array('f'), bytearray()
    targets: List[List[int]] = []
    probs: List[List[float]] = []
    scores: List[List[float]] = []
    values: List[float] = []
    for x, m, t, p, s, margin in recorder.rows:
        obs_buf.extend(x)
        mask_buf.extend(bytes(m))
        targets.append(t)
        probs.append(p)
        scores.append(s)
        values.append((final_margin - margin) * VALUE_SCALE)
    return obs_buf.tobytes(), bytes(mask_buf), targets, probs, scores, values


def collect_search(
    checkpoint: str,
    n_games: int,
    cells: Sequence[Cell],
    depth: int = 3,
    worlds: int = 16,
    candidates: int = 4,
    temperature: float = 1.0,
    opponents: Sequence[str] = ('neural',),
    seed: int = 0,
    processes: int = 1,
    log: Optional[Callable[[str], None]] = None,
) -> SearchDataset:
    """Play ``n_games`` (cycling through ``cells``) with one searching seat and record its decisions.

    ``opponents`` fill the other seats: ``'neural'`` is the plain network of ``checkpoint``, any
    other name a built-in agent. ``temperature`` (VP) turns candidate values into target
    probabilities: low values approach the search's argmax, high values a uniform target.
    """
    checkpoint = os.path.abspath(checkpoint)
    search_kwargs = {'depth': depth, 'worlds': worlds, 'candidates': candidates}
    jobs = [
        (
            g,
            seed,
            tuple(cells[g % len(cells)][0]),
            cells[g % len(cells)][1],
            checkpoint,
            search_kwargs,
            temperature,
            tuple(opponents),
        )
        for g in range(n_games)
    ]
    vocab = NeuralAgent(checkpoint).vocab
    obs_parts: List[bytes] = []
    masks: List[bytes] = []
    targets: List[List[int]] = []
    probs: List[List[float]] = []
    scores: List[List[float]] = []
    values: List[float] = []

    def consume(
        parts: Iterable[Tuple[bytes, bytes, List[List[int]], List[List[float]], List[List[float]], List[float]]],
    ) -> None:
        for done, (o, m, t, p, s, v) in enumerate(parts, 1):
            obs_parts.append(o)
            masks.append(m)
            targets.extend(t)
            probs.extend(p)
            scores.extend(s)
            values.extend(v)
            if log is not None and done % max(1, n_games // 20) == 0:
                log(f'collected {done}/{n_games} games, {len(values)} decisions')

    if processes > 1:
        with get_context().Pool(processes, initializer=init_worker) as pool:
            consume(pool.imap_unordered(_collect_game, jobs, chunksize=1))
    else:
        consume(map(_collect_game, jobs))
    obs = torch.frombuffer(bytearray(b''.join(obs_parts)), dtype=torch.float32).view(-1, vocab.obs_size)
    mask = torch.frombuffer(bytearray(b''.join(masks)), dtype=torch.uint8).view(-1, vocab.n_actions).bool()
    return SearchDataset(
        obs.clone(),
        mask,
        torch.tensor(targets, dtype=torch.int64),
        torch.tensor(probs, dtype=torch.float32),
        torch.tensor(values, dtype=torch.float32),
        torch.tensor(scores, dtype=torch.float32),
    )


def _soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """``-sum_k p_k log softmax(logits)[target_k]`` per row, averaged (unused slots have p = 0)."""
    log_p = torch.log_softmax(logits, dim=1)
    picked = log_p.gather(1, targets.clamp(min=0))
    return -(probs * picked).sum(dim=1).mean()


def distill(
    model: PolicyValueNet,
    data: SearchDataset,
    epochs: int = 3,
    batch_size: int = 512,
    lr: float = 1e-4,
    value_coef: float = 0.5,
    val_fraction: float = 0.1,
    seed: int = 0,
    log: Optional[Callable[[str], None]] = print,
    device: str = 'auto',
    temperature: Optional[float] = None,
) -> Dict[str, float]:
    """Fine-tune ``model`` towards the search's targets; returns held-out agreement and losses.

    ``temperature`` re-derives the targets from the stored candidate scores (default: the targets as
    collected).

    The model keeps its input normalisation (it is a fine-tune, not a fresh fit), and the learning
    rate is low by default so the policy moves towards the search without forgetting.
    """
    if temperature is not None:
        data = data.with_temperature(temperature)
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(data), generator=gen)
    n_val = max(1, int(len(data) * val_fraction))
    val, train = order[:n_val], order[n_val:]
    target = torch.device(
        'cuda' if device == 'auto' and torch.cuda.is_available() else ('cpu' if device == 'auto' else device)
    )
    model.to(target)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    metrics: Dict[str, float] = evaluate(model, data, val, target)
    if log is not None:
        log(f'before: agreement {metrics["val_agreement"]:.3f}  value mse {metrics["val_value_mse"]:.3f}')
    for epoch in range(epochs):
        model.train()
        perm = train[torch.randperm(len(train), generator=gen)]
        total, batches = 0.0, 0
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            logits, value = model(data.obs[idx].to(target), data.mask[idx].to(target))
            loss = _soft_cross_entropy(logits, data.targets[idx].to(target), data.probs[idx].to(target))
            loss = loss + value_coef * nn.functional.mse_loss(value, data.value[idx].to(target))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            batches += 1
        metrics = {'train_loss': total / max(1, batches), **evaluate(model, data, val, target)}
        if log is not None:
            log(
                f'epoch {epoch + 1}/{epochs}: loss {metrics["train_loss"]:.3f}  '
                f'agreement {metrics["val_agreement"]:.3f}  value mse {metrics["val_value_mse"]:.3f}'
            )
    model.cpu()
    model.eval()
    return metrics


def evaluate(
    model: PolicyValueNet, data: SearchDataset, idx: torch.Tensor, device: Optional[torch.device] = None
) -> Dict[str, float]:
    """How often the network's greedy move is the search's preferred one, and value MSE, on ``idx``."""
    model.eval()
    device = device or next(model.parameters()).device
    agree, squared, count = 0.0, 0.0, 0
    with torch.inference_mode():
        for chunk in idx.split(8192):
            logits, value = model(data.obs[chunk].to(device), data.mask[chunk].to(device))
            best = data.targets[chunk].gather(1, data.probs[chunk].argmax(dim=1, keepdim=True)).squeeze(1)
            agree += float((logits.argmax(dim=1).cpu() == best).sum())
            squared += float(((value.cpu() - data.value[chunk]) ** 2).sum())
            count += len(chunk)
    return {'val_agreement': agree / max(1, count), 'val_value_mse': squared / max(1, count)}
