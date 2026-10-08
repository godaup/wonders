"""Network-guided lookahead: determinized rollouts over sampled worlds (evaluation agent).

:class:`SearchAgent` improves a trained :class:`~wonders.ml.model.PolicyValueNet` at play time.
At each regular turn it

1. takes the ``candidates`` most likely legal actions under the network's policy;
2. samples ``worlds`` complete games consistent with what the seat has seen
   (:func:`~wonders.agents.worlds.sample_world` — opponents' hands, hidden discards, leaders);
3. plays every candidate out in every world, every seat following the network greedily, for
   ``depth`` further turns (``None``: to the end of the game);
4. scores each rollout by the seat's score margin over the opponents' mean — the final margin, or
   at the depth limit the current margin plus the network's value estimate of the rest (the value
   head predicts exactly that remaining margin, scaled by ``VALUE_SCALE``);
5. plays the candidate with the best mean.

All candidates see the same worlds (common random numbers), so their comparison is not swamped
by the sampling noise. All rollouts advance in lockstep and every network call is batched across
them. Other decisions (Leaders draft and recruitment, wonder-power picks) use the network directly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple

import torch

from wonders.agents.base import Agent
from wonders.agents.worlds import World, sample_world
from wonders.cards.base import Card
from wonders.engine.actions import Action, DecisionKind, DecisionPoint
from wonders.engine.observation import Observation
from wonders.ml.agent import NeuralAgent
from wonders.ml.imitation import VALUE_SCALE


@dataclass
class _Rollout:
    world: World
    first: Action  # the candidate the searching seat plays at the root
    steps: Generator[DecisionPoint, Dict[int, Action], None] = field(init=False)
    point: Optional[DecisionPoint] = None
    turns: int = 0  # regular turns completed after the root
    value: Optional[float] = None  # the rollout's result, once finished

    def __post_init__(self) -> None:
        self.steps = self.world.steps()
        self.point = next(self.steps)


class SearchAgent(Agent):
    """Determinized rollout search on top of a trained network (see the module docstring).

    Args:
        checkpoint: The network to search with (a :func:`~wonders.ml.model.save_checkpoint` file).
        worlds: Sampled worlds per decision.
        candidates: Most likely actions (by the policy) that are searched; the rest are never played.
        depth: Regular turns to roll out after the root before the value head takes over;
            ``None`` plays every rollout to the end of the game.
        prior_weight: VP added per nat of the policy's log-probability, so a candidate the network
            dislikes must win its rollouts clearly before it is played (0: rollouts alone decide).
        rng: Random source for the world samples.
        name: Display name.
        device: Torch device for the network.
    """

    name = 'search'

    def __init__(
        self,
        checkpoint: str,
        *,
        worlds: int = 8,
        candidates: int = 4,
        depth: Optional[int] = None,
        prior_weight: float = 0.0,
        rng: Optional[random.Random] = None,
        name: str = 'search',
        device: str = 'cpu',
    ):
        if worlds < 1 or candidates < 1 or (depth is not None and depth < 0):
            raise ValueError('need worlds >= 1, candidates >= 1 and depth >= 0 (or None)')
        self.net = NeuralAgent(checkpoint, device=device)
        self.worlds = worlds
        self.candidates = candidates
        self.depth = depth
        self.prior_weight = prior_weight
        self.rng = rng or random.Random()
        self.name = name

    def act(self, obs: Observation, options: List[Card]) -> Action:
        if obs.decision != DecisionKind.PLAY.value:
            return self.net.act(obs, options)
        return self.decide(obs)

    def decide(self, obs: Observation) -> Action:
        if obs.decision != DecisionKind.PLAY.value:
            return self.net.decide(obs)
        logits = self.net.logits(obs)
        space = self.net.action_space
        ranked = sorted(obs.legal_actions, key=lambda a: -float(logits[space.index(a)]))
        top = ranked[: self.candidates]
        if len(top) == 1:
            return top[0]
        values = self.evaluate(obs, top)
        if self.prior_weight:
            legal = torch.tensor([space.index(a) for a in obs.legal_actions])
            log_prior = torch.log_softmax(logits[legal], dim=0)
            by_index = dict(zip(legal.tolist(), log_prior.tolist()))
            values = [v + self.prior_weight * by_index[space.index(a)] for v, a in zip(values, top)]
        best = max(range(len(top)), key=lambda k: values[k])  # ties go to the higher prior
        return top[best]

    def evaluate(self, obs: Observation, actions: List[Action]) -> List[float]:
        """Mean rollout value of each of ``actions`` (legal at ``obs``) over the sampled worlds."""
        seeds = [self.rng.getrandbits(64) for _ in range(self.worlds)]
        rollouts: List[List[_Rollout]] = []
        for action in actions:
            row = []
            for seed in seeds:  # the same worlds for every candidate
                world = sample_world(obs, random.Random(seed))
                row.append(_Rollout(world, action))
            rollouts.append(row)
        self._run([r for row in rollouts for r in row], obs.seat)
        return [sum(r.value or 0.0 for r in row) / len(row) for row in rollouts]

    # -- lockstep rollouts -------------------------------------------------------------------
    def _run(self, rollouts: List[_Rollout], me: int) -> None:
        live = list(rollouts)
        while live:
            requests: List[Tuple[_Rollout, int, Observation]] = []
            for r in live:
                point = r.point
                assert point is not None
                for seat, options in point.options.items():
                    if _at_root(r, seat, me):
                        continue  # the candidate under evaluation
                    requests.append((r, seat, r.world.observations.build(r.world.game, seat, options)))
            choices, values = self._infer([obs for _r, _s, obs in requests])
            answers: Dict[int, Dict[int, Action]] = {id(r): {} for r in live}
            leaf: Dict[int, float] = {}
            for (r, seat, obs), action, value in zip(requests, choices, values):
                answers[id(r)][seat] = action
                if seat == me and self._at_depth(r):
                    leaf[id(r)] = value
            still = []
            for r in live:
                if id(r) in leaf:
                    r.value = _margin(r, me) + leaf[id(r)] / VALUE_SCALE
                    continue
                point = r.point
                assert point is not None
                actions = answers[id(r)]
                if _at_root(r, me, me):
                    actions[me] = _same(r.first, point.options[me])
                r.world.record(point, actions)
                if point.kind is DecisionKind.PLAY:
                    r.turns += 1
                try:
                    r.point = r.steps.send(actions)
                    still.append(r)
                except StopIteration:
                    r.point = None
                    r.value = _margin(r, me)
            live = still

    def _at_depth(self, r: _Rollout) -> bool:
        point = r.point
        return (
            self.depth is not None
            and r.turns >= max(1, self.depth)
            and point is not None
            and point.kind is DecisionKind.PLAY
        )

    def _infer(self, observations: List[Observation]) -> Tuple[List[Action], List[float]]:
        """Greedy actions and value estimates for a batch of observations."""
        if not observations:
            return [], []
        net = self.net
        x = torch.tensor([net.encoder.encode(o) for o in observations], dtype=torch.float32, device=net.device)
        mask = torch.tensor([net.action_space.mask(o.legal_actions) for o in observations], device=net.device)
        with torch.inference_mode():
            logits, value = net.model(x, mask)
        best = torch.argmax(logits, dim=1).tolist()
        actions = [net.action_space.to_action(i, o.legal_actions) for i, o in zip(best, observations)]
        return actions, value.cpu().tolist()


def _at_root(r: _Rollout, seat: int, me: int) -> bool:
    """Whether ``seat`` is the searching seat at the root turn (it plays the candidate there)."""
    point = r.point
    return seat == me and r.turns == 0 and point is not None and point.kind is DecisionKind.PLAY


def _same(action: Action, options: List[Card]) -> Action:
    """``action`` with the matching card object of a rebuilt world's hand."""
    card = next(c for c in options if c.name == action.card.name)
    return Action(action.kind, card)


def _margin(r: _Rollout, me: int) -> float:
    """The seat's score minus the opponents' mean (what the value head predicts the change of)."""
    game = r.world.game
    scores = [game.score(i).total for i in range(game.n)]
    others = [s for i, s in enumerate(scores) if i != me]
    return scores[me] - sum(others) / len(others)
