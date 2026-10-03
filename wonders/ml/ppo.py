"""Proximal Policy Optimization against baseline opponents on vectorized envs.

The learner plays one seat of every table (rotating across envs so it sees every
seat); opponents are built from :data:`~wonders.agents.bench.AGENTS` names. Rewards
are the env's ``'dense'`` reward — the step change in own VP minus the opponents'
mean — scaled by :data:`~wonders.ml.imitation.VALUE_SCALE`, so returns equal the final
margin (the same target the imitation value head learned). Games are short (~20
learner decisions), so ``gamma`` defaults to 1.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
from torch import nn
from torch.distributions import Categorical

from wonders.agents.base import Agent
from wonders.agents.bench import AGENTS, AgentSpec
from wonders.env.vector import SingleAgentEnvFactory, SubprocVectorEnv, SyncVectorEnv
from wonders.ml.imitation import VALUE_SCALE
from wonders.ml.model import PolicyValueNet


@dataclass(frozen=True)
class _SeededOpponent:
    """Picklable zero-argument factory: build ``spec`` with a fixed seed."""

    spec: AgentSpec
    seed: int

    def __call__(self) -> Agent:
        return self.spec.build(random.Random(self.seed))


@dataclass
class PPOConfig:
    opponents: Sequence[str] = ('heuristic', 'greedy')  # AGENTS names; table size = len + 1
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


def make_vector_env(config: PPOConfig) -> Union[SyncVectorEnv, SubprocVectorEnv]:
    n_players = len(config.opponents) + 1
    fns = []
    for i in range(config.n_envs):
        opponents = [
            _SeededOpponent(AGENTS[name], config.seed * 10_007 + i * 31 + k) for k, name in enumerate(config.opponents)
        ]
        fns.append(SingleAgentEnvFactory(opponents, learner=i % n_players, env_kwargs={'reward': 'dense'}))
    if config.envs_per_worker <= 0:
        return SyncVectorEnv(fns)
    return SubprocVectorEnv(fns, envs_per_worker=config.envs_per_worker)


def train_ppo(
    model: PolicyValueNet,
    config: PPOConfig,
    log: Optional[Callable[[str], None]] = print,
    on_iteration: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> List[Dict[str, float]]:
    """Improve ``model`` in place with PPO; returns per-iteration metrics.

    Metrics: ``win_rate`` / ``margin`` (learner, over games finished this iteration),
    ``games``, the PPO losses, ``approx_kl`` and ``steps_per_sec``.
    """
    torch.manual_seed(config.seed)
    n_players = len(config.opponents) + 1
    learner_seat = [i % n_players for i in range(config.n_envs)]
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-5)
    history: List[Dict[str, float]] = []
    vec = make_vector_env(config)
    try:
        observations, _infos = vec.reset(config.seed)
        for iteration in range(1, config.iterations + 1):
            start = time.time()
            model.eval()
            xs, masks, actions, logps, values, rewards, dones = [], [], [], [], [], [], []
            wins = margins = games = 0.0
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
                        wins += float(scores[me] > best_other)
                        margins += scores[me] - best_other
                        games += 1
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
                'steps_per_sec': steps / (time.time() - start),
                **losses,
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
    finally:
        vec.close()
    model.eval()
    return history


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
