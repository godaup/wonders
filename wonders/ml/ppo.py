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

from wonders.agents.base import Agent
from wonders.agents.bench import AGENTS, AgentSpec
from wonders.env.vector import SingleAgentEnvFactory, SubprocVectorEnv, SyncVectorEnv
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
        return universal_vocab()
    expansions = {e for exp, _n in config.tables for e in exp} or {'base'}
    return vocab_for(tuple(sorted(expansions)))


def opponent_spec(entry: str) -> AgentSpec:
    """A built-in agent name or ``neural:<checkpoint>`` as a picklable spec."""
    if entry.startswith('neural:'):
        path = entry[len('neural:') :]
        label = f'neural:{os.path.splitext(os.path.basename(path))[0]}'
        return AgentSpec.of(label, NeuralAgent, checkpoint=os.path.abspath(path), name=label)
    return AGENTS[entry]


def env_tables(config: PPOConfig) -> List[TableSpec]:
    """The table each env plays (legacy configs: the base game at ``len(opponents) + 1``)."""
    tables = list(config.tables) or [(('base',), len(config.opponents) + 1)]
    return [tables[i % len(tables)] for i in range(config.n_envs)]


def make_vector_env(
    config: PPOConfig, pool: Optional[Sequence[str]] = None, epoch: int = 0
) -> Union[SyncVectorEnv, SubprocVectorEnv]:
    vocab = config_vocab(config)
    pool = list(pool if pool is not None else (config.opponent_pool or config.opponents))
    fns = []
    for i, (expansions, n_players) in enumerate(env_tables(config)):
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
        }
        fns.append(SingleAgentEnvFactory(opponents, learner=i % n_players, env_kwargs=env_kwargs))
    if config.envs_per_worker <= 0:
        return SyncVectorEnv(fns)
    return SubprocVectorEnv(fns, envs_per_worker=config.envs_per_worker)


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
    tables = env_tables(config)
    learner_seat = [i % n for i, (_exp, n) in enumerate(tables)]
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-5)
    start_iteration = 1
    pool = list(config.opponent_pool or config.opponents)
    if state:
        opt.load_state_dict(state['optimizer'])
        start_iteration = int(state['iteration']) + 1
        pool = list(state.get('pool', pool))
    history: List[Dict[str, float]] = []
    vec = make_vector_env(config, pool, epoch=start_iteration)
    try:
        observations, _infos = vec.reset(config.seed + start_iteration)
        for iteration in range(start_iteration, config.iterations + 1):
            start = time.time()
            model.eval()
            xs, masks, actions, logps, values, rewards, dones = [], [], [], [], [], [], []
            wins = margins = games = rel = 0.0
            per_table: Dict[str, List[float]] = {}
            for _ in range(config.rollout_steps):
                x = torch.tensor([o['observation'] for o in observations], dtype=torch.float32)
                m = torch.tensor([o['action_mask'] for o in observations])
                with torch.no_grad():
                    logits, value = model(x, m)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                observations, reward, terminated, truncated, infos = vec.step(action.tolist())
                done = [t or u for t, u in zip(terminated, truncated)]
                for env, (finished, info) in enumerate(zip(done, infos)):
                    if finished:
                        scores = info['final_info']['scores']
                        me = learner_seat[env]
                        best_other = max(s for j, s in enumerate(scores) if j != me)
                        won = float(scores[me] > best_other)
                        wins += won
                        margins += scores[me] - best_other
                        games += 1
                        rel += won * len(scores)
                        bucket = per_table.setdefault(_table_key(tables[env]), [0.0, 0.0])
                        bucket[0] += won * len(scores)
                        bucket[1] += 1
                xs.append(x)
                masks.append(m)
                actions.append(action)
                logps.append(dist.log_prob(action))
                values.append(value)
                rewards.append(torch.tensor(reward, dtype=torch.float32) * VALUE_SCALE)
                dones.append(torch.tensor(done, dtype=torch.float32))
            with torch.no_grad():
                x = torch.tensor([o['observation'] for o in observations], dtype=torch.float32)
                m = torch.tensor([o['action_mask'] for o in observations])
                _logits, next_value = model(x, m)
            advantages = _gae(rewards, values, dones, next_value, config.gamma, config.gae_lambda)
            returns = advantages + torch.stack(values)
            losses = _update(
                model,
                opt,
                config,
                torch.cat(xs),
                torch.cat(masks),
                torch.cat(actions),
                torch.cat(logps),
                advantages.flatten(),
                returns.flatten(),
            )
            steps = config.rollout_steps * config.n_envs
            metrics = {
                'iteration': float(iteration),
                'games': games,
                'win_rate': wins / games if games else 0.0,
                'margin': margins / games if games else 0.0,
                'win_rel': rel / games if games else 0.0,
                'steps_per_sec': steps / (time.time() - start),
                **losses,
                **{f'win_rel/{key}': r / g for key, (r, g) in sorted(per_table.items())},
            }
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
                    on_snapshot(iteration, {'optimizer': opt.state_dict(), 'iteration': iteration, 'pool': pool})
                if config.league and iteration < config.iterations:  # new opponents: rebuild the envs
                    vec.close()
                    vec = make_vector_env(config, pool, epoch=iteration)
                    observations, _infos = vec.reset(config.seed + iteration)
    finally:
        vec.close()
    model.eval()
    return history


def _snapshot(model: PolicyValueNet, config: PPOConfig, vocab: Vocab, iteration: int, pool: List[str]) -> List[str]:
    """With ``league``, write a snapshot checkpoint and put it in the pool (keeping the newest few).

    Without a league nothing is written here (``on_snapshot`` gets the resumable state).
    """
    if not config.league:
        return pool
    directory = config.league_dir or '.'
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f'snapshot-{iteration:06d}.pt')
    save_checkpoint(path, model, vocab.card_names, decisions=list(vocab.decisions), iteration=iteration)
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
    advantages = torch.zeros(len(rewards), next_value.shape[0])
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
        for idx in torch.randperm(n).split(config.minibatch):
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
