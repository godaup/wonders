"""A resumable experiment runner for training sweeps.

A sweep is a JSON spec::

    {"out": "runs/pilot",
     "runs": [
       {"name": "imit-base3", "kind": "imitate", "games": 3000,
        "tables": [[["base"], ["board", "board", "board"]]], "teachers": ["board"]},
       {"name": "ppo-base3", "kind": "ppo", "init": "imit-base3", "steps": 500000,
        "ppo": {"tables": [[["base"], 3]], "opponent_pool": ["heuristic", "board"]},
        "grid": {"lr": [3e-5, 1e-4], "entropy_coef": [0.001, 0.003]},
        "eval": {"cells": "base/3p", "games": 600}}
     ]}

``grid`` expands a run into one run per combination (``ppo-base3-lr0.0001-...``);
``init`` names an earlier run whose ``final.pt`` the model starts from; ``steps`` sets
the iteration count from the rollout size. Each run gets ``<out>/<name>/`` holding
``config.json``, ``metrics.jsonl`` (one line per PPO iteration), ``snapshots/``,
``state.pt`` (resumable training state), ``final.pt``, ``eval.json`` / ``eval.md`` and a
``done`` marker: re-running a sweep skips finished runs and resumes interrupted ones.
"""

from __future__ import annotations

import itertools
import json
import os
import time
from dataclasses import fields
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from wonders.env.vocab import universal_vocab, vocab_for
from wonders.ml.evaluate import evaluate, parse_cells, summary_table
from wonders.ml.imitation import collect, new_model, train_imitation
from wonders.ml.model import PolicyValueNet, load_checkpoint, save_checkpoint
from wonders.ml.ppo import PPOConfig, config_vocab, train_ppo

Log = Callable[[str], None]


def expand(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand ``grid`` entries into one run per combination of values."""
    out: List[Dict[str, Any]] = []
    for run in runs:
        grid = run.get('grid')
        if not grid:
            out.append(run)
            continue
        keys = sorted(grid)
        for values in itertools.product(*(grid[k] for k in keys)):
            child = json.loads(json.dumps({k: v for k, v in run.items() if k != 'grid'}))
            child.setdefault('ppo', {})
            for key, value in zip(keys, values):
                child['ppo'][key] = value
            child['name'] = run['name'] + ''.join(f'-{k}{v}' for k, v in zip(keys, values))
            out.append(child)
    return out


def _tables(raw: Sequence[Any]) -> List[Any]:
    """JSON lists back to the tuples the trainers expect."""
    return [(tuple(exp), (tuple(x) if isinstance(x, list) else x)) for exp, x in raw]


def ppo_config(run: Dict[str, Any], run_dir: str) -> PPOConfig:
    params = dict(run.get('ppo', {}))
    allowed = {f.name for f in fields(PPOConfig)}
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"run {run['name']}: unknown PPO settings {sorted(unknown)}")
    if 'tables' in params:
        params['tables'] = _tables(params['tables'])
    for key in ('opponents', 'opponent_pool'):
        if key in params:
            params[key] = tuple(params[key])
    params.setdefault('snapshot_every', 25)  # resumable state every 25 iterations
    params.setdefault('league_dir', os.path.join(run_dir, 'snapshots'))
    config = PPOConfig(**params)
    if 'steps' in run:
        config.iterations = max(1, int(run['steps']) // (config.n_envs * config.rollout_steps))
    return config


def _imitate(run: Dict[str, Any], run_dir: str, processes: int, log: Log) -> str:
    tables = _tables(run['tables'])
    encoder = run.get('encoder', 'card-v1')  # 'card-v1' or 'table-v2'
    expansions = tuple(sorted({e for exp, _ in tables for e in exp}))
    vocab = universal_vocab(encoder) if run.get('universal') else vocab_for(expansions, encoder)
    data = collect(
        int(run.get('games', 3000)),
        tables=tables,
        teachers=tuple(run.get('teachers', ('board',))),
        seed=int(run.get('seed', 0)),
        processes=processes,
        vocab=vocab,
    )
    log(f"{run['name']}: {len(data)} teacher decisions")
    model, names = new_model(int(run.get('hidden', 256)), int(run.get('layers', 2)), vocab)
    metrics = train_imitation(model, data, epochs=int(run.get('epochs', 12)), seed=int(run.get('seed', 0)), log=log)
    final = os.path.join(run_dir, 'final.pt')
    save_checkpoint(final, model, names, source='imitation', **vocab.meta(), **metrics)
    return final


def _ppo(run: Dict[str, Any], run_dir: str, out: str, log: Log, previous: Optional[Dict[str, Any]] = None) -> str:
    config = ppo_config(run, run_dir)
    vocab = config_vocab(config)
    state_path = os.path.join(run_dir, 'state.pt')
    state: Optional[Dict[str, Any]] = None
    if os.path.exists(state_path):  # resume an interrupted run
        saved = torch.load(state_path, map_location='cpu', weights_only=True)
        model = PolicyValueNet(**saved['config'])
        model.load_state_dict(saved['model'])
        iteration = int(saved['iteration'])
        steps_done = int(saved.get('steps', iteration * _batch_size(previous or run)))
        state = {'optimizer': saved['optimizer'], 'iteration': iteration, 'pool': saved['pool'], 'steps': steps_done}
        if 'steps' in run:  # finish the step budget even if the rollout size changed
            batch = config.n_envs * config.rollout_steps
            config.iterations = iteration + max(0, -(-(int(run['steps']) - steps_done) // batch))
        log(f"{run['name']}: resuming after iteration {iteration} ({steps_done:,} steps done)")
    elif run.get('init'):
        model, names, meta = load_checkpoint(os.path.join(out, run['init'], 'final.pt'))
        if tuple(names) != vocab.card_names or meta.get('encoder', 'card-v1') != vocab.encoder_kind:
            raise ValueError(f"run {run['name']}: init checkpoint vocabulary / encoder differs from the PPO config")
    else:
        model, _names = new_model(int(run.get('hidden', 256)), int(run.get('layers', 2)), vocab)
    metrics_path = os.path.join(run_dir, 'metrics.jsonl')

    def on_iteration(iteration: int, metrics: Dict[str, float]) -> None:
        with open(metrics_path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps({'time': time.time(), **metrics}) + '\n')

    def on_snapshot(iteration: int, snapshot: Dict[str, Any]) -> None:
        tmp = state_path + '.tmp'
        torch.save({'config': model.config, 'model': model.state_dict(), **snapshot}, tmp)
        os.replace(tmp, state_path)

    train_ppo(model, config, log=log, on_iteration=on_iteration, on_snapshot=on_snapshot, state=state)
    final = os.path.join(run_dir, 'final.pt')
    save_checkpoint(final, model, vocab.card_names, source='ppo', run=run['name'], **vocab.meta())
    return final


def _batch_size(run: Dict[str, Any]) -> int:
    """Learner steps per PPO iteration of a run spec."""
    params = run.get('ppo', {})
    defaults = PPOConfig()
    return int(params.get('n_envs', defaults.n_envs)) * int(params.get('rollout_steps', defaults.rollout_steps))


def run_one(run: Dict[str, Any], out: str, processes: int = 6, log: Log = print) -> Optional[str]:
    """Execute one run (skipped if already done); returns its final checkpoint."""
    run_dir = os.path.join(out, run['name'])
    final = os.path.join(run_dir, 'final.pt')
    if os.path.exists(os.path.join(run_dir, 'done')):
        log(f"{run['name']}: done, skipping")
        return final
    os.makedirs(run_dir, exist_ok=True)
    previous: Optional[Dict[str, Any]] = None  # the config an interrupted run was started with
    config_path = os.path.join(run_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, encoding='utf-8') as fh:
            previous = json.load(fh)
    with open(config_path, 'w', encoding='utf-8') as fh:
        json.dump(run, fh, indent=1)
    started = time.time()
    if not os.path.exists(final):
        if run['kind'] == 'imitate':
            _imitate(run, run_dir, processes, log)
        elif run['kind'] == 'ppo':
            _ppo(run, run_dir, out, log, previous)
        else:
            raise ValueError(f"unknown run kind {run['kind']!r}")
    if run.get('eval'):
        spec = run['eval']
        results = evaluate(
            final,
            cells=parse_cells(spec.get('cells', 'all')),
            lineups=tuple(spec.get('lineups', ('heuristic', 'board'))),
            games=int(spec.get('games', 600)),
            seed=int(spec.get('seed', 4242)),
            processes=processes,
        )
        with open(os.path.join(run_dir, 'eval.json'), 'w', encoding='utf-8') as fh:
            json.dump(results, fh, indent=1)
        with open(os.path.join(run_dir, 'eval.md'), 'w', encoding='utf-8') as fh:
            fh.write(summary_table(results, run['name']) + '\n')
    with open(os.path.join(run_dir, 'done'), 'w', encoding='utf-8') as fh:
        fh.write(f'{time.time() - started:.0f}s\n')
    log(f"{run['name']}: finished in {time.time() - started:.0f}s")
    return final


def run_sweep(
    spec_path: str,
    only: Sequence[str] = (),
    processes: int = 6,
    dry_run: bool = False,
    log: Log = print,
) -> List[str]:
    """Run every (or every ``only``-matching) run of a sweep spec, in order."""
    with open(spec_path, encoding='utf-8') as fh:
        spec = json.load(fh)
    out = spec['out']
    runs = expand(spec['runs'])
    if only:
        runs = [r for r in runs if any(r['name'].startswith(prefix) for prefix in only)]
    if dry_run:
        for run in runs:
            detail = f" ({run['steps']:,} steps)" if 'steps' in run else ''
            log(f"{run['kind']:8} {run['name']}{detail}")
        return [r['name'] for r in runs]
    os.makedirs(out, exist_ok=True)
    finished = []
    for run in runs:
        run_one(run, out, processes=processes, log=log)
        finished.append(run['name'])
    return finished
