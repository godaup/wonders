"""Diagnostics for trained policies, focused on the Leaders decisions.

- :func:`teacher_agreement` plays teacher-only games and, at every teacher decision, asks the
  network what it would play: agreement per decision kind (``play``, ``draft``, ``recruit``, …).
- :func:`leaders_report` measures what the Leaders decisions are worth: the teacher against
  itself with random leader picks, and the network playing leaders with its own policy
  against the cost proxy and random picks — all at the same tables
  (:func:`~wonders.ml.evaluate.compare_agents`).
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from wonders._mp import get_context, init_worker
from wonders.agents.ablation import leader_ablation
from wonders.agents.base import Agent
from wonders.agents.bench import AGENTS, _content, wilson_interval
from wonders.agents.match import play_match
from wonders.cards.base import Card
from wonders.engine.actions import Action
from wonders.engine.observation import Observation
from wonders.engine.record import PlayerSetup
from wonders.ml.agent import NeuralAgent
from wonders.ml.evaluate import Cell, cell_key, compare_agents, neural_spec

LEADERS_CELLS: Tuple[Cell, ...] = tuple(
    (combo, n) for combo in (('base', 'leaders'), ('base', 'leaders', 'cities')) for n in (3, 4, 5, 6, 7)
)

_MODELS: Dict[str, NeuralAgent] = {}  # per-process cache: checkpoint -> agent


def _model(checkpoint: str) -> NeuralAgent:
    if checkpoint not in _MODELS:
        _MODELS[checkpoint] = NeuralAgent(checkpoint)
    return _MODELS[checkpoint]


class _Probe(Agent):
    """Plays as ``teacher`` and tallies whether ``model``'s policy would have chosen the same."""

    def __init__(self, teacher: Agent, model: NeuralAgent, tally: Counter):
        self.teacher, self.model, self.tally = teacher, model, tally
        self.name = teacher.name

    def decide(self, obs: Observation) -> Action:
        return self.teacher.decide(obs)

    def act(self, obs: Observation, options: List[Card]) -> Action:
        chosen = self.teacher.act(obs, options)
        guess = self.model.decide(obs)
        self.tally[(obs.decision, 'n')] += 1
        self.tally[(obs.decision, 'agree')] += (guess.kind, guess.card.name) == (chosen.kind, chosen.card.name)
        return chosen


def _agreement_game(job: Tuple[int, int, str, str, Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
    index, seed, checkpoint, teacher, expansions, n = job
    content = _content(expansions)
    rng = random.Random(seed * 1_000_003 + index)
    names = list(dict.fromkeys(w.name for w in content.wonders))
    setups = [PlayerSetup(f'{teacher}-{i}', w, rng.choice('AB')) for i, w in enumerate(rng.sample(names, n))]
    tally: Counter = Counter()
    model = _model(checkpoint)
    agents: List[Agent] = [_Probe(AGENTS[teacher].build(random.Random(rng.random())), model, tally) for _ in range(n)]
    play_match(content, setups, agents, rng=rng, expansions=expansions)
    return dict(tally)


def teacher_agreement(
    checkpoint: str,
    cells: Sequence[Cell] = LEADERS_CELLS,
    teacher: str = 'board',
    games: int = 50,
    seed: int = 777,
    processes: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """Share of ``teacher`` decisions the network's greedy policy reproduces, per decision kind."""
    jobs = [(g, seed, checkpoint, teacher, tuple(exp), n) for exp, n in cells for g in range(games)]
    if processes > 1:
        with get_context().Pool(processes, initializer=init_worker) as pool:
            parts = pool.map(_agreement_game, jobs, chunksize=max(1, len(jobs) // (processes * 8)))
    else:
        parts = [_agreement_game(job) for job in jobs]
    total: Counter = Counter()
    for part in parts:
        total.update(part)
    out: Dict[str, Dict[str, Any]] = {}
    for kind in sorted({k for k, _ in total}):
        n, agree = total[(kind, 'n')], total[(kind, 'agree')]
        lo, hi = wilson_interval(agree, n)
        out[kind] = {'decisions': n, 'agreement': agree / n if n else 0.0, 'ci': [lo, hi]}
    return out


def leaders_report(
    checkpoint: str,
    label: str = 'model',
    cells: Sequence[Cell] = LEADERS_CELLS,
    teacher: str = 'board',
    games: int = 200,
    agreement_games: int = 50,
    seed: int = 4242,
    processes: int = 1,
) -> Dict[str, Any]:
    """Teacher agreement plus Leaders ablations for ``teacher`` and the network."""
    base = AGENTS[teacher]
    model = neural_spec(label, checkpoint)
    model_cost = neural_spec(f'{label}[cost]', checkpoint, leader_policy=False)
    return {
        'checkpoint': checkpoint,
        'cells': [cell_key(c) for c in cells],
        'agreement': teacher_agreement(checkpoint, cells, teacher, agreement_games, seed, processes),
        'teacher_ablation': compare_agents(
            [base, leader_ablation(base, 'random-draft'), leader_ablation(base, 'random')],
            cells,
            games,
            seed,
            processes=processes,
        ),
        'model_ablation': compare_agents(
            [model, model_cost, leader_ablation(model, 'random')], cells, games, seed, processes=processes
        ),
    }


def leaders_report_table(report: Dict[str, Any], title: Optional[str] = None) -> str:
    """Markdown summary of :func:`leaders_report`."""
    from wonders.ml.evaluate import head_to_head_table

    lines = [f'## {title}', ''] if title else []
    lines += ['### Teacher agreement (greedy policy vs teacher, per decision kind)', '']
    lines += ['| decision | decisions | agreement (95% CI) |', '|---|---|---|']
    for kind, a in report['agreement'].items():
        lo, hi = a['ci']
        lines.append(f"| {kind} | {a['decisions']} | {100 * a['agreement']:.1f}% ({100 * lo:.1f}–{100 * hi:.1f}) |")
    lines += ['', head_to_head_table(report['teacher_ablation'], 'Teacher: own leader play vs random'), '']
    lines += [head_to_head_table(report['model_ablation'], 'Model: policy vs cost proxy vs random leaders')]
    return '\n'.join(lines)
