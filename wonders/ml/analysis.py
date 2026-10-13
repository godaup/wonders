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
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from wonders._mp import get_context, init_worker
from wonders.agents.bench import _content
from wonders.cards.base import Color, ScienceSymbol
from wonders.engine.actions import Action, ActionKind, DecisionKind
from wonders.engine.record import GameRecord
from wonders.i18n import Message, Translator, english, msg
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


def _amount(vp: float, category: str) -> Message:
    return msg('{vp:.1f} {category}', vp=vp, category=msg(CATEGORY_NAMES.get(category, category), 'category'))


def reason_message(best: Dict[str, float], played: Dict[str, float], limit: int = 2) -> Message:
    """Why the better move is better: the score categories it gains and gives up a few turns later.

    The categories are measured where the lookahead stops; whatever they do not show comes from the
    network's estimate of the rest of the game (a better long-term outlook).
    """
    deltas = {k: best.get(k, 0.0) - played.get(k, 0.0) for k in set(best) | set(played)}
    gains = sorted(((d, k) for k, d in deltas.items() if d >= 0.5), reverse=True)[:limit]
    costs = sorted((d, k) for k, d in deltas.items() if d <= -0.5)[:limit]
    parts = []
    if gains:
        parts.append(msg('gains {items}', items=[_amount(d, k) for d, k in gains]))
    if costs:
        parts.append(msg('gives up {items} now', items=[_amount(-d, k) for d, k in costs]))
    if not gains:
        parts.append(msg('the gain is in the longer-term outlook'))
    return {'join': '; ', 'items': parts}


def reason(best: Dict[str, float], played: Dict[str, float], limit: int = 2) -> str:
    """:func:`reason_message` in English."""
    return english(reason_message(best, played, limit))


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
        'reason_msg': None if is_best else reason_message(best.categories, mine.categories),
        'candidates': [{**_move(e.action), 'value': e.mean, 'prior_rank': ranked.index(e.action) + 1} for e in evals],
    }


def _rank(game: Any, seat: int) -> int:
    """1-based place of ``seat`` (score, then coins — like :meth:`Game.winner`)."""
    key = [(game.score(i).total, game.players[i].coins) for i in range(game.n)]
    return 1 + sum(1 for i in range(game.n) if i != seat and key[i] > key[seat])


def _what_if(job: Tuple[str, str, int, int, Dict[str, str], Dict[str, str], int]) -> Dict[str, Any]:
    """One paired continuation: the played and the better move from the true position (worker process)."""
    from wonders.ml.agent import NeuralAgent

    record_json, checkpoint, seat, index, played, best, seed = job
    record = GameRecord.from_json(record_json)
    content = _content(Ruleset.of(*record.expansions).content_expansions)
    agent = NeuralAgent(checkpoint)

    def policy(_game: Any, _i: int, obs: Any, options: Any) -> Action:
        return agent.act(obs, options)

    out: Dict[str, Any] = {'index': index}
    for name, move in (('played', played), ('best', best)):
        game = record.branch(
            content,
            seat,
            index,
            Action(ActionKind(move['kind']), content.card(move['card'])),
            policy,
            rng=random.Random(seed),
        )
        scores = [game.score(i).total for i in range(game.n)]
        out[name] = {
            'rank': _rank(game, seat),
            'margin': scores[seat] - max(s for i, s in enumerate(scores) if i != seat),
        }
    return out


def _summarise_what_if(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(runs)
    out: Dict[str, Any] = {'runs': n}
    for name in ('played', 'best'):
        ranks = [r[name]['rank'] for r in runs]
        out[name] = {
            'first': round(sum(1 for x in ranks if x == 1) / n, 3),
            'mean_rank': round(sum(ranks) / n, 2),
            'mean_margin': round(sum(r[name]['margin'] for r in runs) / n, 2),
        }
    return out


AGE_VALUE = {1: 1, 2: 3, 3: 5}  # military victory tokens per Age


Pattern = Dict[str, Any]
Decisions = Sequence[Tuple[Any, Any, Action]]


def _pattern(kind: str, title: Message, detail: Message, vp: float) -> Pattern:
    """A pattern in English, keeping its messages for other languages (see :func:`localize`)."""
    return {
        'kind': kind,
        'title': english(title),
        'detail': english(detail),
        'title_msg': title,
        'detail_msg': detail,
        'vp': round(vp, 1),
    }


def localize(report: Dict[str, Any], t: Translator) -> Dict[str, Any]:
    """A copy of an :func:`analyse` report with its reasons and patterns in ``t``'s language."""
    out = dict(report)
    out['decisions'] = [
        {**d, 'reason': t.render(d['reason_msg'])} if d.get('reason_msg') else d for d in report['decisions']
    ]
    out['patterns'] = [
        {**p, 'title': t.render(p['title_msg']), 'detail': t.render(p['detail_msg'])} if 'title_msg' in p else p
        for p in report.get('patterns', [])
    ]
    return out


def _trade_pattern(decisions: Decisions) -> List[Pattern]:
    """Coins paid to neighbours for resources (cards built from hand)."""
    trade = 0
    for point, obs, action in decisions:
        if point.kind is DecisionKind.PLAY and action.kind is ActionKind.BUILD:
            view = next((c for c in obs.hand if c.name == action.card.name), None)
            trade += view.trade_coins if view is not None else 0
    if trade < 6:
        return []
    return [
        _pattern(
            'trade',
            msg('{n} coins spent on trade', n=trade),
            msg(
                'You paid your neighbours {n} coins for resources — about {vp:.0f} VP, and coins they could '
                'use against you. More of your own brown / grey production would have kept them.',
                n=trade,
                vp=trade / 3,
            ),
            trade / 3,
        )
    ]


def _military_patterns(game: Any, seat: int, decisions: Decisions, names: List[str]) -> List[Pattern]:
    """Conflicts lost or tied by a single shield (shields as they stood when each Age ended)."""
    n = game.n
    shields_at: Dict[int, Tuple[int, int, int]] = {}
    for age in (1, 2):
        first = next((obs for point, obs, _a in decisions if point.age == age + 1), None)
        if first is not None:
            shields_at[age] = (first.me.shields, first.left.shields, first.right.shields)
    me = game.players[seat]
    shields_at[3] = (me.shields(), game.left_of(seat).shields(), game.right_of(seat).shields())
    out = []
    for age, (mine, left, right) in sorted(shields_at.items()):
        for theirs, other in ((left, (seat + 1) % n), (right, (seat - 1) % n)):
            if not 0 <= theirs - mine <= 1:
                continue
            if theirs > mine:  # lost by one: one more shield ties (no defeat token), two more win
                title = msg('Age {age}: lost against {name} by one shield', age=age, name=names[other])
                detail = msg(
                    'You had {mine} shields to their {theirs}: one more would have avoided the −1, two more would '
                    'have won {vp} VP.',
                    mine=mine,
                    theirs=theirs,
                    vp=AGE_VALUE[age],
                )
                stake = 1 + AGE_VALUE[age] / 2
            else:  # tied: one more shield wins
                title = msg('Age {age}: tied with {name}', age=age, name=names[other])
                detail = msg('You both had {n} shields: one more would have won {vp} VP.', n=mine, vp=AGE_VALUE[age])
                stake = float(AGE_VALUE[age])
            out.append(_pattern('military', title, detail, stake))
    return out


def _science_pattern(game: Any, seat: int) -> List[Pattern]:
    """What one more science symbol would have been worth."""
    me = game.players[seat]
    symbols, wilds = me.science(), me.science_wilds()
    if sum(symbols.values()) + wilds < 2:
        return []
    now = game._science(seat, symbols, wilds)
    gains = {s: game._science(seat, symbols + Counter({s: 1}), wilds) - now for s in ScienceSymbol}
    symbol, gain = max(gains.items(), key=lambda kv: kv[1])
    if gain < 4:
        return []
    have = [msg('{symbol} {n}', symbol=msg(k.value, 'science'), n=v) for k, v in symbols.items() if v]
    if wilds:
        have.append(msg('{n} of your choice', n=wilds))
    name = msg(symbol.value, 'science')
    return [
        _pattern(
            'science',
            msg('One more {symbol} was worth {vp} VP', symbol=name, vp=gain),
            msg(
                'Your science ({have}) scores {now} VP; one more {symbol} would have added {vp}.',
                have=have,
                now=now,
                symbol=name,
                vp=gain,
            ),
            gain / 2,
        )
    ]


def _wonder_and_coin_patterns(game: Any, seat: int) -> List[Pattern]:
    """Wonder stages left unbuilt, and coins left over (one VP per three)."""
    me = game.players[seat]
    out = []
    built, total = me.wonder_stages_built, len(me.wonder.stages)
    if built < total:
        ctx = game.context(seat)
        vp = sum(e.victory_points(ctx) for stage in me.wonder.stages[built:] for e in stage.effects)
        out.append(
            _pattern(
                'wonder',
                msg('{n} of {total} wonder stages unbuilt', n=total - built, total=total),
                msg('The remaining stages were worth at least {vp} VP (plus their other effects).', vp=vp),
                vp * 0.5,
            )
        )
    if me.coins >= 8:
        out.append(
            _pattern(
                'coins',
                msg('Finished with {n} coins', n=me.coins),
                msg(
                    'They scored only {vp} VP; spent earlier on cards or wonder stages they are usually worth more.',
                    vp=me.coins // 3,
                ),
                me.coins / 6,
            )
        )
    return out


def _valuable(card: Any, ctx: Any) -> bool:
    """Cards worth denying a neighbour: guilds, science, military and civic cards of 5+ VP."""
    if card.guild or card.color in (Color.GREEN, Color.RED):
        return True
    return card.color is Color.BLUE and sum(e.victory_points(ctx) for e in card.effects) >= 5


def _passing_and_discard_patterns(
    record: GameRecord, seat: int, decisions: Decisions, names: List[str], game: Any
) -> List[Pattern]:
    """Valuable cards you could afford, passed to the neighbour who built them next; discards for coins."""
    content = _content(Ruleset.of(*record.expansions).content_expansions)
    n = len(record.players)
    passed = []
    discards = 0
    for point, obs, action in decisions:
        if point.kind is not DecisionKind.PLAY:
            continue
        discards += action.kind is ActionKind.DISCARD
        down = (seat + 1) % n if point.age in (1, 3) else (seat - 1) % n
        turns = record.moves.get(point.age, [])
        if point.turn + 1 < len(turns):
            code, card = turns[point.turn + 1][down][:2]
            view = next((c for c in obs.hand if c.name == card), None)
            if (
                code == 'B'
                and view is not None
                and view.can_build
                and card != action.card.name
                and _valuable(content.card(card), game.context(down))
            ):
                passed.append((card, names[down]))
    out = []
    if passed:
        shown = [msg('{card} (to {name})', card=msg(c), name=who) for c, who in passed[:3]]
        title = (
            '{n} cards you passed were built next door'
            if len(passed) > 1
            else '{n} card you passed were built next door'
        )
        out.append(
            _pattern(
                'passing',
                msg(title, n=len(passed)),
                msg(
                    '{cards} — cards you could have built yourself. Taking or burying a card your neighbour needs '
                    'can be worth more than a small gain of your own.',
                    cards=shown,
                ),
                2 * len(passed),
            )
        )
    if discards >= 3:
        out.append(
            _pattern(
                'discards',
                msg('{n} cards discarded for coins', n=discards),
                msg('Each discard gives 3 coins (1 VP); a wonder stage from the same card is usually worth more.'),
                discards,
            )
        )
    return out


def _timing_pattern(entries: Sequence[Dict[str, Any]]) -> List[Pattern]:
    """The Age in which most points were lost."""
    by_age: Dict[int, float] = {}
    for e in entries:
        if 'loss' in e:
            by_age[e['age']] = by_age.get(e['age'], 0.0) + e['loss']
    if not by_age:
        return []
    age, lost = max(by_age.items(), key=lambda kv: kv[1])
    if lost < 3:
        return []
    return [
        _pattern(
            'timing',
            msg('Most points lost in Age {age}', age=age),
            msg("{vp:.1f} VP over that Age's moves (of {total:.1f} in total).", vp=lost, total=sum(by_age.values())),
            lost,
        )
    ]


GOOD = ('best', 'good')
KINDS = ('build', 'wonder', 'discard', 'free')


def _rated_stats(entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Move count, accuracy and VP lost of a set of rated decisions."""
    rated = [e for e in entries if 'loss' in e]
    return {
        'moves': len(rated),
        'accuracy': round(sum(1 for e in rated if e['label'] in GOOD) / len(rated), 3) if rated else None,
        'loss': round(sum(e['loss'] for e in rated), 2),
    }


def _ranked(values: List[float], seat: int) -> int:
    """1-based rank of ``values[seat]`` among all seats (higher is better; ties share the better rank)."""
    return 1 + sum(1 for i, v in enumerate(values) if i != seat and v > values[seat])


def aspects(record: GameRecord, seat: int, decisions: Decisions, entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Statistics of one player's game for charts, compared with the rest of the table where useful.

    Covers the score categories, accuracy by Age and by move type, the city's card colours, military
    results per Age and neighbour, the economy and the wonder.
    """
    content = _content(Ruleset.of(*record.expansions).content_expansions)
    game = record.replay(content)
    n = game.n
    rows = [game.score(i).as_dict() for i in range(n)]
    categories = []
    for key in list(rows[seat]):
        values = [float(r[key]) for r in rows]
        others = [v for i, v in enumerate(values) if i != seat]
        if not any(values):
            continue
        categories.append(
            {
                'key': key,
                'mine': values[seat],
                'average': round(sum(others) / len(others), 2),
                'best_other': max(others),
                'rank': _ranked(values, seat),
            }
        )
    rated = [e for e in entries if 'loss' in e]
    by_age = [{'age': age, **_rated_stats([e for e in rated if e['age'] == age])} for age in (1, 2, 3)]
    by_kind = {k: _rated_stats([e for e in rated if e['played']['kind'] == k]) for k in KINDS}
    matrix = {k: {b: 0 for b in KINDS} for k in KINDS}
    for e in rated:
        matrix[e['played']['kind']][e['best']['kind']] += 1
    colors = sorted({c.color.value for p in game.players for c in p.cards})
    cards = {
        col: {
            'mine': sum(1 for c in game.players[seat].cards if c.color.value == col),
            'average': round(
                sum(sum(1 for c in game.players[i].cards if c.color.value == col) for i in range(n) if i != seat)
                / (n - 1),
                2,
            ),
        }
        for col in colors
    }
    trade = chained = 0
    wonder_ages: List[int] = []
    for point, obs, action in decisions:
        if point.kind is not DecisionKind.PLAY:
            continue
        view = next((c for c in obs.hand if c.name == action.card.name), None)
        if action.kind is ActionKind.BUILD and view is not None:
            trade += view.trade_coins
            chained += bool(view.chain_free)
        if action.kind is ActionKind.WONDER:
            wonder_ages.append(point.age)
    me = game.players[seat]
    military = []
    names = [s.name for s in record.players]
    shields: Dict[int, List[int]] = {}
    for age in (1, 2):
        first = next((obs for point, obs, _a in decisions if point.age == age + 1), None)
        if first is not None:
            shields[age] = [c.shields for c in first.table]
    shields[3] = [p.shields() for p in game.players]
    for age, row in sorted(shields.items()):
        for side, other in (('left', (seat + 1) % n), ('right', (seat - 1) % n)):
            diff = row[seat] - row[other]
            military.append(
                {
                    'age': age,
                    'side': side,
                    'opponent': names[other],
                    'mine': row[seat],
                    'theirs': row[other],
                    'result': 'win' if diff > 0 else 'tie' if diff == 0 else 'loss',
                }
            )
    coins = [p.coins for p in game.players]
    return {
        'categories': categories,
        'by_age': by_age,
        'by_kind': {k: v for k, v in by_kind.items() if v['moves']},
        'kind_matrix': {k: {b: c for b, c in row.items() if c} for k, row in matrix.items() if any(row.values())},
        'cards': cards,
        'military': military,
        'economy': {
            'trade_paid': trade,
            'chain_builds': chained,
            'discards': sum(1 for e in rated if e['played']['kind'] == 'discard'),
            'final_coins': me.coins,
            'coins_rank': _ranked([float(c) for c in coins], seat),
        },
        'wonder': {'built': me.wonder_stages_built, 'total': len(me.wonder.stages), 'ages': wonder_ages},
        'players': n,
    }


def patterns(
    record: GameRecord, seat: int, decisions: Decisions, entries: Sequence[Dict[str, Any]], limit: int = 6
) -> List[Pattern]:
    """Recurring themes of one player's game, each with an estimate of the VP at stake (largest first)."""
    content = _content(Ruleset.of(*record.expansions).content_expansions)
    game = record.replay(content)
    names = [s.name for s in record.players]
    found = (
        _trade_pattern(decisions)
        + _military_patterns(game, seat, decisions, names)
        + _science_pattern(game, seat)
        + _wonder_and_coin_patterns(game, seat)
        + _passing_and_discard_patterns(record, seat, decisions, names, game)
        + _timing_pattern(entries)
    )
    found.sort(key=lambda f: -f['vp'])
    return found[:limit]


class UnknownContent(ValueError):
    """The network cannot rate a game whose cards or boards it was never trained on."""

    def __init__(self, message: Message):
        super().__init__(english(message))
        self.message = message


def unknown_content(record: GameRecord, checkpoint: Optional[str] = None) -> Optional[Message]:
    """Why ``checkpoint``'s network cannot analyse ``record`` (cards or boards it never saw), or ``None``.

    Checks what the game actually used: the dealt Age decks, the leaders and the players' boards.
    """
    from wonders.cards.content import read_expansion
    from wonders.extensions import REGISTRY
    from wonders.ml import pretrained_path
    from wonders.ml.model import load_checkpoint
    from wonders.ruleset import EXPANSIONS

    _model, card_names, meta = load_checkpoint(os.path.abspath(checkpoint or pretrained_path()))
    used = {name for deck in record.age_decks.values() for name in deck}
    used |= {name for hand in record.leader_hands.values() for name in hand} | set(record.leader_box)
    missing = sorted(used - set(card_names))
    known_boards = set(meta.get('wonders', ()))
    if known_boards:  # older checkpoints do not list boards
        missing += sorted({p.wonder for p in record.players} - known_boards)
    if not missing:
        return None
    owners = []
    for exp in Ruleset.of(*record.expansions).expansions:
        raw = REGISTRY.expansion_raw(exp) or read_expansion(exp)
        names = {str(item.get('name')) for item in raw.get('cards', []) + raw.get('wonders', [])}
        if names & set(missing):
            owners.append(msg(EXPANSIONS[exp].name if exp in EXPANSIONS else exp, 'expansion'))
    text = (
        'The analysis network was not trained with {expansions}: it does not know {names}. '
        if len(missing) <= 3
        else 'The analysis network was not trained with {expansions}: it does not know {names} and {more} more. '
    )
    return msg(
        text + 'It can analyse games with the official expansions.',
        expansions={'join': ' + ', 'items': owners or ['?']},
        names=[msg(name) for name in missing[:3]],  # names in the reader's language
        more=len(missing) - 3,
    )


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
    what_if: int = 3,
    what_if_runs: int = 24,
) -> Dict[str, Any]:
    """Rate every regular-turn decision of ``seat`` in ``record`` (see the module docstring).

    The ``what_if`` costliest key moments are also replayed from the true position with the played
    and with the better move (``what_if_runs`` paired continuations each, later Ages reshuffled),
    every seat then playing the network.
    """
    from wonders.ml import pretrained_path

    checkpoint = os.path.abspath(checkpoint or pretrained_path())
    reason = unknown_content(record, checkpoint)
    if reason is not None:
        raise UnknownContent(reason)
    content = _content(Ruleset.of(*record.expansions).content_expansions)
    decisions = list(record.replay_observations(content, seat))
    settings = {'depth': depth, 'worlds': worlds, 'candidates': candidates, 'seed': seed}
    rated = [k for k, (point, _obs, _a) in enumerate(decisions) if point.kind is DecisionKind.PLAY]
    record_json = record.to_json()
    jobs = [(record_json, checkpoint, seat, k, settings) for k in rated]
    reviews: Dict[int, Dict[str, Any]] = {}
    continuations: Dict[int, List[Dict[str, Any]]] = {}
    total = len(jobs) + what_if * what_if_runs
    done = 0

    def tick() -> None:
        nonlocal done
        done += 1
        if progress is not None:
            progress(done, total)

    def what_if_jobs() -> List[Tuple[str, str, int, int, Dict[str, str], Dict[str, str], int]]:
        costly = sorted(
            (r for r in reviews.values() if r['label'] in ('inaccuracy', 'mistake', 'blunder')),
            key=lambda r: -r['loss'],
        )[:what_if]
        return [
            (record_json, checkpoint, seat, r['index'], r['played'], r['best'], seed * 1_000_003 + k)
            for r in costly
            for k in range(what_if_runs if r['age'] < 3 else 1)  # Age III: nothing left to reshuffle
        ]

    def run(pool: Any) -> None:
        mapper = pool.imap_unordered if pool is not None else map
        for review in mapper(_review, jobs):
            reviews[review['index']] = review
            tick()
        for result in mapper(_what_if, what_if_jobs()):
            continuations.setdefault(result['index'], []).append(result)
            tick()

    if processes > 1:
        with get_context().Pool(processes, initializer=init_worker) as pool:
            run(pool)
    else:
        run(None)
    for index, runs in continuations.items():
        reviews[index]['what_if'] = _summarise_what_if(runs)
    if progress is not None:
        progress(total, total)

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
        'patterns': patterns(record, seat, decisions, entries),
        'aspects': aspects(record, seat, decisions, entries),
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
        w = e.get('what_if')
        if w:
            gain = w['best']['mean_margin'] - w['played']['mean_margin']
            lines.append(
                f"    what if ({w['runs']} continuation{'s' if w['runs'] > 1 else ''}): {gain:+.1f} VP against the "
                f"leader, rank {w['best']['mean_rank']} instead of {w['played']['mean_rank']}, 1st place "
                f"{100 * w['best']['first']:.0f}% vs {100 * w['played']['first']:.0f}%"
                + (' — the continuations disagree with the verdict' if gain < -1 else '')
            )
    if report.get('patterns'):
        lines.append('Patterns:')
        lines.extend(f"  {p['title']} — {p['detail']}" for p in report['patterns'])
    return '\n'.join(lines)


def analyse_games(records: Sequence[GameRecord], seat: int, **kwargs: Any) -> List[Dict[str, Any]]:
    """:func:`analyse` for several games (e.g. a player's history)."""
    return [analyse(r, seat, **kwargs) for r in records]
