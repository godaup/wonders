"""Vectorized environments: many :class:`SingleAgentEnv` games stepped as a batch.

The engine is pure Python and single-threaded, so throughput scales by running
games in parallel *processes*. :class:`SubprocVectorEnv` hosts one env per worker
process; :class:`SyncVectorEnv` has the identical API in-process (for debugging,
tests, or when per-step IPC would cost more than the game).

Both follow the Gymnasium vector-env conventions with plain lists (no numpy
dependency): ``reset(seed) -> (observations, infos)`` and ``step(actions) ->
(observations, rewards, terminated, truncated, infos)``, one entry per env. An
env whose game ends is **reset automatically**: its observation is the new game's
first one and its info carries the finished game under ``'final_info'``
(``scores`` and the replayable ``record``). With a seed, env ``i`` starts from
``seed + i`` and draws the seeds of later games from its own RNG, so a batch run
is reproducible regardless of timing.
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import random
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from wonders.agents.base import Agent
from wonders.env.environment import EnvObs, SingleAgentEnv

EnvFn = Callable[[], SingleAgentEnv]
VecStep = Tuple[List[EnvObs], List[float], List[bool], List[bool], List[Dict[str, Any]]]


@dataclass
class SingleAgentEnvFactory:
    """A picklable recipe for a :class:`SingleAgentEnv` with *fresh* opponents.

    ``opponents`` are zero-argument agent factories (agent classes, or
    ``functools.partial(RandomAgent, random.Random(0))``). Each env deep-copies them
    before calling, so state bound into a factory (like that ``Random``) is never shared
    between envs: every env starts from the same state, in-process or in a worker.
    Being a plain dataclass, it pickles under any multiprocessing start method (unlike
    a lambda).
    """

    opponents: Sequence[Callable[[], Agent]]
    learner: int = 0
    env_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __call__(self) -> SingleAgentEnv:
        opponents = [copy.deepcopy(make)() for make in self.opponents]
        return SingleAgentEnv(opponents, learner=self.learner, **copy.deepcopy(self.env_kwargs))


class _AutoResetEnv:
    """One env plus the auto-reset/seeding/info-trimming shared by both vector envs."""

    def __init__(self, env: SingleAgentEnv, keep_obs: bool):
        self.env = env
        self.keep_obs = keep_obs
        self.rng: Optional[random.Random] = None

    def _trim(self, info: Dict[str, Any]) -> Dict[str, Any]:
        if self.keep_obs:
            return info
        return {k: v for k, v in info.items() if k != 'obs'}

    def reset(self, seed: Optional[int]) -> Tuple[EnvObs, Dict[str, Any]]:
        self.rng = random.Random(seed) if seed is not None else None
        obs, info = self.env.reset(seed)
        return obs, self._trim(info)

    def step(self, action: int) -> Tuple[EnvObs, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            seed = self.rng.randrange(2**31) if self.rng is not None else None
            obs, reset_info = self.env.reset(seed)
            info = {**self._trim(reset_info), 'final_info': info}
        else:
            info = self._trim(info)
        return obs, reward, terminated, truncated, info


def _transpose(results: Sequence[Tuple]) -> VecStep:
    obs, rewards, terminated, truncated, infos = zip(*results)
    return list(obs), list(rewards), list(terminated), list(truncated), list(infos)


class SyncVectorEnv:
    """Steps ``len(env_fns)`` envs one after another in this process.

    Args:
        env_fns: One zero-argument env factory per env (e.g. :class:`SingleAgentEnvFactory`).
        keep_obs: Keep the raw ``Observation`` in each info (``'obs'``); off by default
            because the encoded ``'observation'`` is what a learner consumes.
    """

    def __init__(self, env_fns: Sequence[EnvFn], keep_obs: bool = False):
        self._envs = [_AutoResetEnv(fn(), keep_obs) for fn in env_fns]
        self.num_envs = len(self._envs)
        self.action_space = self._envs[0].env.action_space

    def reset(self, seed: Optional[int] = None) -> Tuple[List[EnvObs], List[Dict[str, Any]]]:
        pairs = [env.reset(None if seed is None else seed + i) for i, env in enumerate(self._envs)]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    def step(self, actions: Sequence[int]) -> VecStep:
        if len(actions) != self.num_envs:
            raise ValueError(f'expected {self.num_envs} actions, got {len(actions)}')
        return _transpose([env.step(a) for env, a in zip(self._envs, actions)])

    def close(self) -> None:
        self._envs = []


def _worker(remote: Connection, parent_remote: Connection, env_fns: Sequence[EnvFn], keep_obs: bool) -> None:
    """Host a group of envs; each command carries one argument per hosted env."""
    parent_remote.close()
    envs = [_AutoResetEnv(fn(), keep_obs) for fn in env_fns]
    try:
        while True:
            cmd, args = remote.recv()
            if cmd == 'step':
                remote.send([env.step(a) for env, a in zip(envs, args)])
            elif cmd == 'reset':
                remote.send([env.reset(seed) for env, seed in zip(envs, args)])
            elif cmd == 'action_space':
                remote.send(envs[0].env.action_space)
            elif cmd == 'close':
                break
    except (EOFError, KeyboardInterrupt):  # parent went away
        pass
    finally:
        remote.close()


class SubprocVectorEnv:
    """Runs the envs in worker processes and steps them in parallel.

    Every step is a lockstep round: each worker steps its envs, and the round lasts
    as long as the slowest worker. Hosting several envs per worker
    (``envs_per_worker``) amortizes the round-trip and evens out per-step latency
    (a step that ends an Age or resets a game is much slower than a plain turn); use
    about one worker per physical core.

    Args:
        env_fns: One zero-argument env factory per env. They are sent to the workers,
            so they must be picklable under the chosen start method — use
            :class:`SingleAgentEnvFactory` or a ``functools.partial`` rather than a lambda.
        keep_obs: See :class:`SyncVectorEnv`.
        context: Multiprocessing start method (``'fork'``, ``'spawn'``,
            ``'forkserver'``); defaults to the platform default.
        envs_per_worker: How many consecutive envs each worker process hosts.

    Use as a context manager (or call :meth:`close`) to shut the workers down.
    """

    def __init__(
        self,
        env_fns: Sequence[EnvFn],
        keep_obs: bool = False,
        context: Optional[str] = None,
        envs_per_worker: int = 1,
    ):
        if envs_per_worker < 1:
            raise ValueError('envs_per_worker must be at least 1')
        ctx = mp.get_context(context)
        self.num_envs = len(env_fns)
        groups = [list(env_fns[k : k + envs_per_worker]) for k in range(0, self.num_envs, envs_per_worker)]
        self._sizes = [len(g) for g in groups]
        self._remotes: List[Connection] = []
        self._processes: List[Any] = []
        for group in groups:
            remote, child = ctx.Pipe()
            process = ctx.Process(  # type: ignore[attr-defined]  # typeshed types get_context() loosely
                target=_worker, args=(child, remote, group, keep_obs), daemon=True
            )
            process.start()
            child.close()
            self._remotes.append(remote)
            self._processes.append(process)
        self._closed = False
        self._remotes[0].send(('action_space', None))
        self.action_space = self._remotes[0].recv()

    def _split(self, values: Sequence[Any]) -> List[List[Any]]:
        out, k = [], 0
        for size in self._sizes:
            out.append(list(values[k : k + size]))
            k += size
        return out

    def _round(self, cmd: str, values: Sequence[Any]) -> List[Any]:
        for remote, chunk in zip(self._remotes, self._split(values)):
            remote.send((cmd, chunk))
        return [result for remote in self._remotes for result in remote.recv()]

    def reset(self, seed: Optional[int] = None) -> Tuple[List[EnvObs], List[Dict[str, Any]]]:
        pairs = self._round('reset', [None if seed is None else seed + i for i in range(self.num_envs)])
        return [p[0] for p in pairs], [p[1] for p in pairs]

    def step(self, actions: Sequence[int]) -> VecStep:
        if len(actions) != self.num_envs:
            raise ValueError(f'expected {self.num_envs} actions, got {len(actions)}')
        return _transpose(self._round('step', actions))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for remote in self._remotes:
            try:
                remote.send(('close', None))
            except (BrokenPipeError, OSError):
                pass
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        for remote in self._remotes:
            remote.close()

    def __enter__(self) -> 'SubprocVectorEnv':
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # pragma: no cover - interpreter shutdown
            pass
