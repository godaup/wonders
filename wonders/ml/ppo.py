"""Proximal Policy Optimization against baseline opponents on vectorized envs.

The learner plays one seat of every table (rotating across envs so it sees every
seat). By default every env is the same base-game table with exactly ``opponents``;
with ``tables`` the envs mix expansion combinations and player counts, each drawing its
opponents from ``opponent_pool`` (built-in names or ``neural:<checkpoint>``). With
``league`` the learner's own snapshots join that pool as it trains (self-play). Rewards
are the env's ``'dense'`` reward — the step change in own VP minus the opponents'
mean — scaled by :data:`~wonders.ml.imitation.VALUE_SCALE`, so returns equal the final
margin (the same target the imitation value head learned). Games are short (~20
learner decisions), so ``gamma`` defaults to 1.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.distributions import Categorical

from wonders.agents.ablation import leader_ablation
from wonders.agents.base import Agent
from wonders.agents.bench import AGENTS, AgentSpec
from wonders.env.environment import EnvObs
from wonders.env.vector import SingleAgentEnvFactory, SubprocVectorEnv, SyncVectorEnv, np
from wonders.env.vocab import Vocab, universal_vocab, vocab_for
from wonders.ml.agent import NeuralAgent
from wonders.ml.imitation import VALUE_SCALE
from wonders.ml.model import PolicyValueNet, save_checkpoint


@dataclass(frozen=True)
class _SeededOpponent:
    """Picklable zero-argument factory: build ``spec`` with a fixed seed."""

    spec: AgentSpec
    seed: int

    def __call__(self) -> Agent:
        return self.spec.build(random.Random(self.seed))


# A training table: (expansions, number of players).
TableSpec = Tuple[Tuple[str, ...], int]


@dataclass
class PPOConfig:
    opponents: Sequence[str] = ('heuristic', 'greedy')  # legacy: exact base-game lineup (size = len + 1)
    tables: Sequence[TableSpec] = ()  # mixed tables; env i plays tables[i % len(tables)]
    opponent_pool: Sequence[str] = ()  # names / 'neural:<path>' to draw opponents from (default: opponents)
    universal: bool = False  # use the all-expansions vocabulary (else the tables' union)
    snapshot_every: int = 0  # every N iterations: resumable state (on_snapshot) + league snapshot (0 = never)
    league: bool = False  # add snapshots to the opponent pool (self-play league)
    league_size: int = 4  # most recent snapshots kept in the pool
    league_dir: str = ''  # where snapshots are written
    device: str = 'auto'  # torch device for the learner: 'auto' (CUDA if available), 'cuda', 'cpu'
    encoder: str = 'card-v1'  # observation encoder: 'card-v1' or 'table-v2'
    pipeline: int = 2  # env groups stepped in turn: one simulates while the learner acts for another
    n_envs: int = 24
    envs_per_worker: int = 4  # 0 = run in-process (SyncVectorEnv)
    rollout_steps: int = 64  # learner steps per env per iteration
    iterations: int = 50
    lr: float = 1e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 1024
    value_coef: float = 0.5
    entropy_coef: float = 0.003
    max_grad_norm: float = 0.5
    seed: int = 0


def config_vocab(config: PPOConfig) -> Vocab:
    """The vocabulary a model trained with ``config`` must have."""
    if config.universal:
        return universal_vocab(config.encoder)
    expansions = {e for exp, _n in config.tables for e in exp} or {'base'}
    return vocab_for(tuple(sorted(expansions)), config.encoder)


def opponent_spec(entry: str) -> AgentSpec:
    """A built-in agent name, ``name[mode]`` or ``neural:<checkpoint>`` as a picklable spec.

    ``name[mode]`` is the built-in agent with its Leaders decisions replaced
    (:func:`~wonders.agents.ablation.leader_ablation`; modes ``cost``, ``random``,
    ``random-draft``) — e.g. ``board[cost]`` plays leaders like the agents did before they had
    data-driven leader values.
    """
    if entry.startswith('neural:'):
        path = entry[len('neural:') :]
        label = f'neural:{os.path.splitext(os.path.basename(path))[0]}'
        return AgentSpec.of(label, NeuralAgent, checkpoint=os.path.abspath(path), name=label)
    if entry.endswith(']') and '[' in entry:
        name, _, mode = entry[:-1].partition('[')
        return leader_ablation(AGENTS[name], mode)
    return AGENTS[entry]


def env_tables(config: PPOConfig) -> List[TableSpec]:
    """The table each env plays (legacy configs: the base game at ``len(opponents) + 1``)."""
    tables = list(config.tables) or [(('base',), len(config.opponents) + 1)]
    return [tables[i % len(tables)] for i in range(config.n_envs)]


def make_vector_env(
    config: PPOConfig,
    pool: Optional[Sequence[str]] = None,
    epoch: int = 0,
    envs: Optional[Sequence[int]] = None,
) -> Union[SyncVectorEnv, SubprocVectorEnv]:
    """Vector env over env indices ``envs`` (default: all ``n_envs``)."""
    vocab = config_vocab(config)
    pool = list(pool if pool is not None else (config.opponent_pool or config.opponents))
    tables = env_tables(config)
    fns = []
    for i in envs if envs is not None else range(config.n_envs):
        expansions, n_players = tables[i]
        seed = config.seed * 10_007 + epoch * 1_000_003 + i * 31
        if config.tables:  # draw this table's opponents from the pool
            names = random.Random(seed).choices(pool, k=n_players - 1)
        else:
            names = list(config.opponents)
        opponents = [_SeededOpponent(opponent_spec(name), seed + k) for k, name in enumerate(names)]
        env_kwargs = {
            'reward': 'dense',
            'expansions': tuple(expansions),
            'card_names': vocab.card_names,
            'decisions': vocab.decisions,
            'encoder': vocab.encoder(),
        }
        fns.append(SingleAgentEnvFactory(opponents, learner=i % n_players, env_kwargs=env_kwargs))
    if config.envs_per_worker <= 0:
        return SyncVectorEnv(fns, compact=True)
    return SubprocVectorEnv(fns, envs_per_worker=config.envs_per_worker, compact=True)


def env_groups(config: PPOConfig) -> List[List[int]]:
    """Env indices per pipeline group (contiguous halves for ``pipeline=2``)."""
    k = max(1, min(config.pipeline, config.n_envs)) if config.envs_per_worker > 0 else 1
    bounds = [round(g * config.n_envs / k) for g in range(k + 1)]
    return [list(range(bounds[g], bounds[g + 1])) for g in range(k)]


def make_vector_envs(
    config: PPOConfig, pool: Optional[Sequence[str]] = None, epoch: int = 0
) -> List[Union[SyncVectorEnv, SubprocVectorEnv]]:
    return [make_vector_env(config, pool, epoch, group) for group in env_groups(config)]


def _table_key(table: TableSpec) -> str:
    expansions, n_players = table
    extras = [e for e in expansions if e != 'base']
    return f"{'+'.join(extras) or 'base'}/{n_players}p"


def train_ppo(
    model: PolicyValueNet,
    config: PPOConfig,
    log: Optional[Callable[[str], None]] = print,
    on_iteration: Optional[Callable[[int, Dict[str, float]], None]] = None,
    on_snapshot: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    state: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, float]]:
    """Improve ``model`` in place with PPO; returns per-iteration metrics.

    Metrics: ``win_rate`` / ``margin`` (learner, over games finished this iteration),
    ``win_rel`` (win rate relative to the fair share ``1/players``, so mixed table sizes
    are comparable; 1.0 = fair), per-table ``win_rel/<table>``, ``games``, the PPO losses,
    ``approx_kl`` and ``steps_per_sec``.

    Every ``snapshot_every`` iterations a checkpoint is written to ``league_dir`` (and, with
    ``league``, added to the opponent pool) and ``on_snapshot(iteration, state)`` receives
    the resumable training state; pass that ``state`` back to continue a run.
    """
    vocab = config_vocab(config)
    if model.config['obs_size'] != vocab.obs_size or model.config['n_actions'] != vocab.n_actions:
        raise ValueError('model shape does not match the configured vocabulary')
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    model.to(device)
    tables = env_tables(config)
    learner_seat = [i % n for i, (_exp, n) in enumerate(tables)]
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-5)
    start_iteration = 1
    steps_done = 0  # cumulative learner steps (survives changes of n_envs / rollout_steps on resume)
    pool = list(config.opponent_pool or config.opponents)
    if state:
        opt.load_state_dict(state['optimizer'])
        start_iteration = int(state['iteration']) + 1
        steps_done = int(state.get('steps', 0))
        pool = list(state.get('pool', pool))
    history: List[Dict[str, float]] = []
    rollouts = _Rollouts(config, pool, start_iteration, device, tables, learner_seat)
    try:
        for iteration in range(start_iteration, config.iterations + 1):
            start = time.time()
            model.eval()
            batch, next_value, stats = rollouts.collect(model)
            advantages = _gae(
                batch['reward'], batch['value'], batch['done'], next_value, config.gamma, config.gae_lambda
            )
            returns = advantages + torch.stack(batch['value'])
            losses = _update(
                model,
                opt,
                config,
                torch.cat(batch['x']),
                torch.cat(batch['mask']),
                torch.cat(batch['action']),
                torch.cat(batch['logp']),
                advantages.flatten(),
                returns.flatten(),
            )
            games = stats['games']
            metrics = {
                'iteration': float(iteration),
                'games': games,
                'win_rate': stats['wins'] / games if games else 0.0,
                'margin': stats['margins'] / games if games else 0.0,
                'win_rel': stats['rel'] / games if games else 0.0,
                'steps': float(steps_done + config.rollout_steps * config.n_envs),
                'steps_per_sec': config.rollout_steps * config.n_envs / (time.time() - start),
                **losses,
                **{f'win_rel/{key}': r / g for key, (r, g) in sorted(rollouts.per_table.items())},
            }
            steps_done += config.rollout_steps * config.n_envs
            history.append(metrics)
            if log is not None:
                log(
                    f'iter {iteration}/{config.iterations}: win {metrics["win_rate"]:.3f} '
                    f'margin {metrics["margin"]:+.2f} over {int(games)} games | '
                    f'pi {losses["policy_loss"]:.3f} v {losses["value_loss"]:.3f} '
                    f'H {losses["entropy"]:.3f} kl {losses["approx_kl"]:.4f} | {metrics["steps_per_sec"]:.0f} steps/s'
                )
            if on_iteration is not None:
                on_iteration(iteration, metrics)
            if config.snapshot_every and iteration % config.snapshot_every == 0:
                pool = _snapshot(model, config, vocab, iteration, pool)
                if on_snapshot is not None:
                    on_snapshot(
                        iteration,
                        {'optimizer': opt.state_dict(), 'iteration': iteration, 'pool': pool, 'steps': steps_done},
                    )
                if config.league and iteration < config.iterations:  # new opponents: rebuild the envs
                    rollouts.close()
                    rollouts = _Rollouts(config, pool, iteration, device, tables, learner_seat)
    finally:
        rollouts.close()
        model.cpu()  # checkpoints and callers always see a CPU model
    model.eval()
    return history


class _Rollouts:
    """Collects rollouts from env groups in a pipeline.

    With ``pipeline=2`` the envs form two groups of worker processes: while one group
    simulates its step, the learner computes the other group's actions, so neither side
    waits for the other. Buffers are reassembled in env order, so GAE and metrics see the
    same layout as a single lockstep vector env.
    """

    def __init__(
        self,
        config: PPOConfig,
        pool: Sequence[str],
        epoch: int,
        device: torch.device,
        tables: Sequence[TableSpec],
        learner_seat: Sequence[int],
    ):
        self.config, self.device, self.tables, self.learner_seat = config, device, tables, learner_seat
        self.groups = env_groups(config)
        self.vecs = make_vector_envs(config, pool, epoch)
        self.obs = [vec.reset(config.seed + epoch + 7919 * g)[0] for g, vec in enumerate(self.vecs)]
        self.per_table: Dict[str, List[float]] = {}

    def close(self) -> None:
        for vec in self.vecs:
            vec.close()

    def collect(self, model: PolicyValueNet) -> Tuple[Dict[str, List[torch.Tensor]], torch.Tensor, Dict[str, float]]:
        """One rollout: per-step tensors (env order), the bootstrap value and game stats."""
        steps, n_groups = self.config.rollout_steps, len(self.groups)
        buf: List[List[Dict[str, torch.Tensor]]] = [[] for _ in range(n_groups)]
        stats = {'wins': 0.0, 'margins': 0.0, 'games': 0.0, 'rel': 0.0}
        self.per_table = {}
        for t in range(steps):
            for g, vec in enumerate(self.vecs):
                if t > 0:
                    self._receive(g, buf[g][-1], stats)  # simulated while the learner acted for others
                x, m = _batch(self.obs[g], self.device)
                with torch.no_grad():
                    logits, value = model(x, m)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                buf[g].append({'x': x, 'mask': m, 'action': action, 'logp': dist.log_prob(action), 'value': value})
                vec.step_async(action.tolist())
        for g in range(n_groups):
            self._receive(g, buf[g][-1], stats)
        batch = {
            key: [torch.cat([buf[g][t][key] for g in range(n_groups)]) for t in range(steps)]
            for key in ('x', 'mask', 'action', 'logp', 'value', 'reward', 'done')
        }
        with torch.no_grad():
            next_value = torch.cat([model(*_batch(self.obs[g], self.device))[1] for g in range(n_groups)])
        return batch, next_value, stats

    def _receive(self, g: int, step: Dict[str, torch.Tensor], stats: Dict[str, float]) -> None:
        """Collect group ``g``'s pending step into ``step``; update the game stats."""
        obs, reward, terminated, truncated, infos = self.vecs[g].step_wait()
        self.obs[g] = obs
        done = [t or u for t, u in zip(terminated, truncated)]
        step['reward'] = torch.tensor(reward, dtype=torch.float32, device=self.device) * VALUE_SCALE
        step['done'] = torch.tensor(done, dtype=torch.float32, device=self.device)
        for local, (finished, info) in enumerate(zip(done, infos)):
            if finished:
                self._record_game(self.groups[g][local], info['final_info']['scores'], stats)

    def _record_game(self, env: int, scores: Sequence[int], stats: Dict[str, float]) -> None:
        me = self.learner_seat[env]
        best_other = max(s for j, s in enumerate(scores) if j != me)
        won = float(scores[me] > best_other)
        stats['wins'] += won
        stats['margins'] += scores[me] - best_other
        stats['games'] += 1
        stats['rel'] += won * len(scores)
        bucket = self.per_table.setdefault(_table_key(self.tables[env]), [0.0, 0.0])
        bucket[0] += won * len(scores)
        bucket[1] += 1


def resolve_device(name: str) -> torch.device:
    """``'auto'`` → CUDA when available, else CPU."""
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def _batch(observations: Sequence[EnvObs], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack per-env observations / masks (NumPy arrays or lists) into tensors on ``device``."""
    first = observations[0]['observation']
    if np is not None and isinstance(first, np.ndarray):
        x = torch.from_numpy(np.stack([o['observation'] for o in observations]))
        m = torch.from_numpy(np.stack([o['action_mask'] for o in observations]))
    else:
        x = torch.tensor([o['observation'] for o in observations], dtype=torch.float32)
        m = torch.tensor([o['action_mask'] for o in observations])
    return x.to(device, non_blocking=True), m.to(device, non_blocking=True)


def _snapshot(model: PolicyValueNet, config: PPOConfig, vocab: Vocab, iteration: int, pool: List[str]) -> List[str]:
    """With ``league``, write a snapshot checkpoint and put it in the pool (keeping the newest few).

    Without a league nothing is written here (``on_snapshot`` gets the resumable state).
    """
    if not config.league:
        return pool
    directory = config.league_dir or '.'
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f'snapshot-{iteration:06d}.pt')
    save_checkpoint(path, model, vocab.card_names, iteration=iteration, **vocab.meta())
    scripted = [p for p in pool if not p.startswith('neural:')]
    neural = [p for p in pool if p.startswith('neural:')] + [f'neural:{path}']
    return scripted + neural[-config.league_size :]


def _gae(
    rewards: List[torch.Tensor],
    values: List[torch.Tensor],
    dones: List[torch.Tensor],
    next_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Generalized advantage estimates ``[T, n_envs]``; ``dones[t]`` ends the episode after step t."""
    advantages = torch.zeros(len(rewards), next_value.shape[0], device=next_value.device)
    running = torch.zeros_like(next_value)
    for t in reversed(range(len(rewards))):
        following = next_value if t == len(rewards) - 1 else values[t + 1]
        alive = 1.0 - dones[t]
        delta = rewards[t] + gamma * following * alive - values[t]
        running = delta + gamma * lam * alive * running
        advantages[t] = running
    return advantages


def _update(
    model: PolicyValueNet,
    opt: torch.optim.Optimizer,
    config: PPOConfig,
    x: torch.Tensor,
    mask: torch.Tensor,
    action: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
) -> Dict[str, float]:
    model.train()
    n = x.shape[0]
    totals = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'approx_kl': 0.0}
    batches = 0
    for _ in range(config.epochs):
        for idx in torch.randperm(n, device=x.device).split(config.minibatch):
            logits, value = model(x[idx], mask[idx])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(action[idx])
            adv = advantages[idx]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ratio = (logp - old_logp[idx]).exp()
            policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - config.clip, 1 + config.clip) * adv).mean()
            value_loss = nn.functional.mse_loss(value, returns[idx])
            entropy = dist.entropy().mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            opt.step()
            with torch.no_grad():
                totals['approx_kl'] += (old_logp[idx] - logp).mean().item()
            totals['policy_loss'] += policy_loss.item()
            totals['value_loss'] += value_loss.item()
            totals['entropy'] += entropy.item()
            batches += 1
    return {k: v / max(1, batches) for k, v in totals.items()}
