"""Post-game analysis: rate one player's decisions with network-guided search.

:func:`analyse` replays a :class:`~wonders.engine.record.GameRecord` and, for every regular turn
of one seat, lets :class:`~wonders.ml.search.SearchAgent` evaluate the move that was played next
to the network's best candidates — in the same sampled worlds, built only from what the player
could see, so the verdict is about the decision, not about hidden cards. Each move gets:

* its **loss**: the best candidate's expected final margin minus the played move's (VP), with a
  paired standard error (the candidates share their sampled worlds);
* a **label**: ``best`` / ``good`` (within noise or under 1 VP) / ``inaccuracy`` (≥ 1 VP) /
  ``mistake`` (≥ 2.5) / ``blunder`` (≥ 5) — only when the loss also clearly exceeds the noise;
* a short **reason**: the score categories in which the better move comes out ahead.

The report is a JSON-ready dict: per-decision reviews, a summary, the key moments and the
expected margin over the game (for a graph). Leaders draft and recruitment, and wonder-power
picks, are listed but not rated.
"""

from __future__ import annotations

import math
import os
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from wonders._mp import get_context, init_worker
from wonders.agents.bench import _content
from wonders.engine.actions import Action, DecisionKind
from wonders.engine.record import GameRecord
from wonders.ruleset import Ruleset

LABELS: Tuple[Tuple[float, str], ...] = ((5.0, 'blunder'), (2.5, 'mistake'), (1.0, 'inaccuracy'))
CATEGORY_NAMES = {
    'military': 'military',
    'treasury': 'coins',
    'wonders': 'wonder',
    'civilian': 'civic buildings',
    'commercial': 'commerce',
    'guilds': 'guilds',
    'science': 'science',
    'leaders': 'leaders',
    'cities': 'city cards',
}


def label(loss: float, se: float, best: bool) -> str:
    """The verdict for a move that lost ``loss`` VP (standard error ``se``) against the best candidate."""
    if best:
        return 'best'
    if loss < max(1.0, 2.0 * se):
        return 'good'
    return next(name for threshold, name in LABELS + ((0.0, 'inaccuracy'),) if loss >= threshold)


def reason(best: Dict[str, float], played: Dict[str, float], limit: int = 2) -> str:
    """Why the better move is better: the score categories it gains and gives up a few turns later.

    The categories are measured where the lookahead stops; whatever they do not show comes from the
    network's estimate of the rest of the game (a better long-term outlook).
    """
    deltas = {k: best.get(k, 0.0) - played.get(k, 0.0) for k in set(best) | set(played)}
    gains = sorted(((d, k) for k, d in deltas.items() if d >= 0.5), reverse=True)[:limit]
    costs = sorted((d, k) for k, d in deltas.items() if d <= -0.5)[:limit]
    parts = []
    if gains:
        parts.append('gains ' + ', '.join(f'{d:.1f} {CATEGORY_NAMES.get(k, k)}' for d, k in gains))
    if costs:
        parts.append('gives up ' + ', '.join(f'{-d:.1f} {CATEGORY_NAMES.get(k, k)}' for d, k in costs) + ' now')
    if not gains:
        parts.append('the gain is in the longer-term outlook')
    return '; '.join(parts)


def _move(action: Action) -> Dict[str, str]:
    return {'kind': action.kind.value, 'card': action.card.name}


def _review(job: Tuple[str, str, int, int, Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate decision ``index`` of ``seat`` in the recorded game (runs in a worker process)."""
    from wonders.ml.search import SearchAgent

    record_json, checkpoint, seat, index, settings = job
    record = GameRecord.from_json(record_json)
    content = _content(Ruleset.of(*record.expansions).content_expansions)
    point, obs, played = next(d for k, d in enumerate(record.replay_observations(content, seat)) if k == index)
    search = SearchAgent(
        checkpoint,
        worlds=settings['worlds'],
        candidates=settings['candidates'],
        depth=settings['depth'],
        rng=random.Random(settings['seed'] * 1_000_003 + index),
    )
    net = search.net
    logits = net.logits(obs)
    space = net.action_space
    ranked = sorted(obs.legal_actions, key=lambda a: -float(logits[space.index(a)]))
    candidates = ranked[: settings['candidates']]
    if played not in candidates:
        candidates.append(next(a for a in obs.legal_actions if a == played))
    evals = search.evaluate_detailed(obs, candidates)
    best = max(evals, key=lambda e: e.mean)
    mine = next(e for e in evals if e.action == played)
    diffs = [b - p for b, p in zip(best.values, mine.values)]
    loss = max(0.0, sum(diffs) / len(diffs))
    var = sum((d - loss) ** 2 for d in diffs) / max(1, len(diffs) - 1)
    se = math.sqrt(var / len(diffs))
    is_best = best is mine or best.mean - mine.mean < 1e-9
    return {
        'index': index,
        'age': point.age,
        'turn': point.turn + 1,
        'decision': point.kind.value,
        'hand': [c.name for c in point.options[seat]],
        'played': _move(played),
        'best': _move(best.action),
        'value_played': mine.mean,
        'value_best': best.mean,
        'loss': 0.0 if is_best else loss,
        'loss_se': se,
        'label': label(loss, se, is_best),
        'reason': '' if is_best else reason(best.categories, mine.categories),
        'candidates': [{**_move(e.action), 'value': e.mean, 'prior_rank': ranked.index(e.action) + 1} for e in evals],
    }


def analyse(
    record: GameRecord,
    seat: int,
    checkpoint: Optional[str] = None,
    depth: int = 3,
    worlds: int = 16,
    candidates: int = 4,
    seed: int = 0,
    processes: int = 1,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """Rate every regular-turn decision of ``seat`` in ``record`` (see the module docstring)."""
    from wonders.ml import pretrained_path

    checkpoint = os.path.abspath(checkpoint or pretrained_path())
    content = _content(Ruleset.of(*record.expansions).content_expansions)
    decisions = list(record.replay_observations(content, seat))
    settings = {'depth': depth, 'worlds': worlds, 'candidates': candidates, 'seed': seed}
    rated = [k for k, (point, _obs, _a) in enumerate(decisions) if point.kind is DecisionKind.PLAY]
    record_json = record.to_json()
    jobs = [(record_json, checkpoint, seat, k, settings) for k in rated]
    reviews: Dict[int, Dict[str, Any]] = {}
    done = 0

    def collect(results) -> None:
        nonlocal done
        for review in results:
            reviews[review['index']] = review
            done += 1
            if progress is not None:
                progress(done, len(jobs))

    if processes > 1:
        with get_context().Pool(processes, initializer=init_worker) as pool:
            collect(pool.imap_unordered(_review, jobs))
    else:
        collect(map(_review, jobs))

    entries: List[Dict[str, Any]] = []
    for k, (point, _obs, action) in enumerate(decisions):
        if k in reviews:
            entries.append(reviews[k])
        else:
            entries.append(
                {
                    'index': k,
                    'age': point.age,
                    'turn': point.turn + 1,
                    'decision': point.kind.value,
                    'played': _move(action),
                    'label': 'not rated',
                }
            )
    counts: Dict[str, int] = {}
    for e in entries:
        counts[e['label']] = counts.get(e['label'], 0) + 1
    rated_entries = [e for e in entries if 'loss' in e]
    key = sorted(
        (e for e in rated_entries if e['label'] in ('inaccuracy', 'mistake', 'blunder')), key=lambda e: -e['loss']
    )[:5]
    setup = record.players[seat]
    return {
        'seat': seat,
        'player': setup.name,
        'wonder': f'{setup.wonder} ({setup.side})',
        'expansions': list(record.expansions),
        'settings': {**settings, 'checkpoint': os.path.basename(checkpoint)},
        'summary': {
            'counts': counts,
            'rated': len(rated_entries),
            'total_loss': round(sum(e['loss'] for e in rated_entries), 2),
            'mean_loss': round(sum(e['loss'] for e in rated_entries) / max(1, len(rated_entries)), 2),
            'accuracy': round(
                sum(1 for e in rated_entries if e['label'] in ('best', 'good')) / max(1, len(rated_entries)), 3
            ),
        },
        'key_moments': [e['index'] for e in key],
        'graph': [
            {
                'index': e['index'],
                'age': e['age'],
                'turn': e['turn'],
                'played': e['value_played'],
                'best': e['value_best'],
            }
            for e in rated_entries
        ],
        'decisions': entries,
    }


def summary_text(report: Dict[str, Any], limit: int = 5) -> str:
    """A short plain-text summary of :func:`analyse`'s report (for the CLI)."""
    s = report['summary']
    lines = [
        f"{report['player']} — {report['wonder']}: {s['rated']} rated moves, accuracy {100 * s['accuracy']:.0f}%, "
        f"{s['total_loss']:.1f} VP lost in total",
        '  ' + ', '.join(f'{n} {k}' for k, n in sorted(s['counts'].items(), key=lambda kv: -kv[1])),
    ]
    by_index = {e['index']: e for e in report['decisions']}
    if report['key_moments']:
        lines.append('Key moments:')
    for k in report['key_moments'][:limit]:
        e = by_index[k]
        lines.append(
            f"  Age {e['age']} turn {e['turn']}: {e['label']} (−{e['loss']:.1f} VP ± {e['loss_se']:.1f}) — you "
            f"{e['played']['kind']} {e['played']['card']}, better {e['best']['kind']} {e['best']['card']} "
            f"({e['reason']})"
        )
    return '\n'.join(lines)


def analyse_games(records: Sequence[GameRecord], seat: int, **kwargs: Any) -> List[Dict[str, Any]]:
    """:func:`analyse` for several games (e.g. a player's history)."""
    return [analyse(r, seat, **kwargs) for r in records]
