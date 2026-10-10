"""Step-based training environments over the engine.

:class:`WondersEnv` is a *parallel* multi-agent environment in the style of
PettingZoo's ``ParallelEnv`` (no dependency on it): every step, each seat in
``env.acting`` submits an action index and the engine advances to the next
decision point. Regular turns are simultaneous (all seats act); the extra
decisions granted by wonder powers (Babylon's 7th card, Halicarnassus' discard
build) involve a single seat.

Each acting seat receives ``{'observation': <encoded>, 'action_mask': [bool, ...]}``
plus the raw :class:`~wonders.engine.observation.Observation` in ``infos[seat]['obs']``
— so scripted :class:`~wonders.agents.base.Agent` opponents can play alongside
learners. Observations are built with the same partial-observability filter as
:func:`~wonders.agents.match.play_match`. Every game is captured as a replayable
:class:`~wonders.engine.record.GameRecord`.

:class:`SingleAgentEnv` wraps it as a Gym-style single-agent env: one learning seat,
the others played by agents.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, Sequence, Set, Tuple, Union

from wonders.agents.base import Agent
from wonders.agents.encoders import DECISIONS, LEADERS_DECISIONS, CardEncoder, Encoder
from wonders.cards.base import Card
from wonders.cards.content import Content, build_age_deck, deal_hands, deal_leaders, load_content
from wonders.engine.actions import Action, DecisionKind, DecisionPoint
from wonders.engine.game import Game
from wonders.engine.observation import Observation, ObservationBuilder
from wonders.engine.phase import Phase
from wonders.engine.player import Player
from wonders.engine.record import GameRecord, PlayerSetup, RecordBuilder
from wonders.env.action_space import ActionSpace
from wonders.ruleset import Ruleset

REWARD_MODES = ('margin', 'win', 'score', 'dense', 'dense_rank')

EnvObs = Dict[str, Any]
StepResult = Tuple[Dict[int, EnvObs], Dict[int, float], Dict[int, bool], Dict[int, bool], Dict[int, Dict[str, Any]]]


class WondersEnv:
    """Parallel multi-agent 7 Wonders environment.

    Args:
        n_players: Seats at the table (3-7 for the base deck).
        expansions: Expansion ids. With Leaders, the draft and each Age's recruitment
            are decision points too (``infos[seat]['decision']`` is ``'draft'`` /
            ``'recruit'``; a draft pick is the ``BUILD`` index of the leader kept).
        content: Pre-loaded content (defaults to ``load_content(*expansions)``).
        encoder: Turns an Observation into the ``'observation'`` entry. Defaults to a
            :class:`CardEncoder` over the action-space vocabulary.
        setups: Fixed wonders per seat; by default each reset deals distinct random wonders.
        sides: ``'A'``, ``'B'`` or ``'random'`` — wonder side when ``setups`` is not given.
        reward: ``'margin'`` (final own VP minus best opponent), ``'win'`` (+1 to the
            winner, ``-1/(n-1)`` to the others), ``'score'`` (final own VP), or ``'dense'``
            (each step: change in own VP minus the opponents' mean; sums to the final margin
            over the mean), or ``'dense_rank'`` (``'dense'`` plus a terminal bonus of
            ``rank_bonus`` VP for first place, falling linearly to ``-rank_bonus`` for last; places
            break ties by coins, like the benchmark).
        rank_bonus: VP-equivalent of finishing first under ``'dense_rank'``.
        card_names: Fix the action-space / encoder vocabulary (e.g. :func:`~wonders.env.vocab.
            universal_vocab`), so one network fits games of any expansion combination.
        decisions: Decision kinds the encoder one-hots (defaults to this game's kinds).
    """

    def __init__(
        self,
        n_players: int = 3,
        expansions: Sequence[str] = ('base',),
        content: Optional[Content] = None,
        encoder: Optional[Encoder] = None,
        setups: Optional[Sequence[PlayerSetup]] = None,
        sides: str = 'random',
        reward: str = 'margin',
        rank_bonus: float = 10.0,
        card_names: Optional[Sequence[str]] = None,
        decisions: Optional[Sequence[str]] = None,
    ):
        self.ruleset = Ruleset.of(*expansions)
        if not 3 <= n_players <= 7:
            raise ValueError(f'the base deck supports 3-7 players, got {n_players}')
        if reward not in REWARD_MODES:
            raise ValueError(f'reward must be one of {REWARD_MODES}, got {reward!r}')
        if sides not in ('A', 'B', 'random'):
            raise ValueError(f"sides must be 'A', 'B' or 'random', got {sides!r}")
        if setups is not None and len(setups) != n_players:
            raise ValueError('need one setup per seat')
        self.expansions = tuple(expansions)
        self.content = content if content is not None else load_content(*self.ruleset.content_expansions)
        if card_names is not None:  # a fixed (e.g. universal) vocabulary, shared across games
            missing = {c.name for c in self.content.cards} - set(card_names)
            if missing:
                raise ValueError(f'card_names lacks {len(missing)} cards of this content, e.g. {sorted(missing)[:3]}')
            self.action_space = ActionSpace(card_names)
        else:
            self.action_space = ActionSpace.from_content(self.content)
        self.leaders = self.ruleset.uses('leaders')
        if decisions is None:
            decisions = LEADERS_DECISIONS if self.leaders else DECISIONS
        self.encoder = encoder if encoder is not None else CardEncoder(self.action_space.card_names, decisions)
        self.n_players = n_players
        self.setups = list(setups) if setups is not None else None
        self.sides = sides
        self.reward_mode = reward
        self.rank_bonus = rank_bonus
        self.possible_agents: List[int] = list(range(n_players))
        self.agents: List[int] = []
        self.game: Optional[Game] = None
        self._point: Optional[DecisionPoint] = None
        self._steps: Optional[Generator[DecisionPoint, Dict[int, Action], None]] = None
        self._observations = ObservationBuilder()
        self._builder: Optional[RecordBuilder] = None
        self._last_margin: List[float] = []
        self._obs_cache: Dict[int, Observation] = {}  # raw observations at the current decision
        # Seats whose observation is encoded in step()/reset() results (None = every acting seat);
        # SingleAgentEnv encodes only the learner — opponents read the raw Observation.
        self.encode_seats: Optional[Set[int]] = None

    # -- public API ----------------------------------------------------------
    @property
    def acting(self) -> List[int]:
        """Seats that must submit an action in the next :meth:`step`."""
        return list(self._point.options) if self._point is not None else []

    @property
    def decision(self) -> Optional[DecisionKind]:
        """Which kind of decision the acting seats face (``None`` once the game is over)."""
        return self._point.kind if self._point is not None else None

    @property
    def done(self) -> bool:
        return self.game is not None and self._point is None

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[int, EnvObs], Dict[int, Dict[str, Any]]]:
        rng = random.Random(seed)
        setups = self.setups or self._random_setups(rng)
        coins = self.ruleset.starting_coins
        players = [Player(s.name, self.content.wonder(s.wonder, s.side), coins) for s in setups]
        self.game = Game(players, expansions=self.expansions)
        self._builder = RecordBuilder(setups, self.expansions)
        self._observations = ObservationBuilder()
        self._last_margin = [0.0] * self.n_players
        self.agents = list(self.possible_agents)
        self._steps = self._run(rng)
        self._point = next(self._steps)
        return self._observe()

    def step(self, actions: Mapping[int, Union[int, Action]]) -> StepResult:
        """Apply one action index per acting seat; return PettingZoo-style dicts.

        A seat may give a legal :class:`Action` instead of its index — e.g. one with a chosen
        split of its purchase between the neighbours (:meth:`Action.paying`).

        Returns ``(observations, rewards, terminations, truncations, infos)``. Rewards,
        terminations and truncations cover every seat; observations and infos cover the
        seats acting next (empty once the game is over).
        """
        if self._point is None or self._steps is None:
            raise RuntimeError('call reset() before step(), and again once the game is over')
        if set(actions) != set(self._point.options):
            raise ValueError(f'expected actions for seats {sorted(self._point.options)}, got {sorted(actions)}')
        game = self._require_game()
        chosen = {seat: self._chosen(game, seat, choice) for seat, choice in actions.items()}
        try:
            self._point = self._steps.send(chosen)
        except StopIteration:
            self._point = None
        terminal = self._point is None
        rewards = self._rewards(terminal)
        if terminal:
            self.agents = []
        observations, infos = self._observe()
        terminations = {seat: terminal for seat in self.possible_agents}
        truncations = {seat: False for seat in self.possible_agents}
        return observations, rewards, terminations, truncations, infos

    def observation(self, seat: int) -> Observation:
        """The raw Observation for an acting ``seat`` at the current decision."""
        if self._point is None or seat not in self._point.options:
            raise ValueError(f'seat {seat} is not acting')
        if seat not in self._obs_cache:
            self._obs_cache[seat] = self._observations.build(self._require_game(), seat, self._point.options[seat])
        return self._obs_cache[seat]

    def options(self, seat: int) -> List[Card]:
        """The cards an acting ``seat`` is choosing from (hand, discard pile or leaders)."""
        if self._point is None or seat not in self._point.options:
            raise ValueError(f'seat {seat} is not acting')
        return list(self._point.options[seat])

    def record(self) -> GameRecord:
        """The replayable record of the game so far."""
        if self._builder is None:
            raise RuntimeError('no game in progress; call reset()')
        return self._builder.build()

    def scores(self) -> List[int]:
        return list(self._require_game().cached_totals())

    # -- internals -----------------------------------------------------------
    def _require_game(self) -> Game:
        if self.game is None:
            raise RuntimeError('call reset() first')
        return self.game

    def _chosen(self, game: Game, seat: int, choice: Union[int, Action]) -> Action:
        assert self._point is not None
        legal = game.legal_actions(seat, self._point.options[seat])
        if isinstance(choice, Action):
            if choice not in legal:
                raise ValueError(f'seat {seat}: {choice.kind.value} {choice.card.name} is not legal')
            return choice
        return self.action_space.to_action(choice, legal)

    def _random_setups(self, rng: random.Random) -> List[PlayerSetup]:
        names = list(dict.fromkeys(w.name for w in self.content.wonders))
        if self.n_players > len(names):
            raise ValueError(f'only {len(names)} wonders for {self.n_players} players')
        chosen = rng.sample(names, self.n_players)
        return [
            PlayerSetup(f'P{i}', name, rng.choice('AB') if self.sides == 'random' else self.sides)
            for i, name in enumerate(chosen)
        ]

    def _run(self, rng: random.Random) -> Generator[DecisionPoint, Dict[int, Action], None]:
        """The whole game as one coroutine: deal each Age and relay its decision points."""
        game = self._require_game()
        builder = self._builder
        assert builder is not None
        if self.leaders:  # draft picks are private; only the resulting hands are recorded
            hands, box = deal_leaders(self.content, game.n, rng)
            game.leader_box = list(box)
            builder.set_leader_box(box)
            yield from self._relay(game.draft_steps(hands))
            builder.set_leader_hands({i: list(p.leaders_hand) for i, p in enumerate(game.players)})
        for age in (1, 2, 3):
            if self.leaders:
                recruited: Dict[int, Action] = {}

                def keep(point: DecisionPoint, actions: Dict[int, Action], _r=recruited) -> None:
                    if point.kind is DecisionKind.RECRUIT:
                        _r.update(actions)
                    else:  # a power resolved during recruitment is recorded like other extras
                        self._record_turn(point, actions)

                yield from self._relay(game.recruit_steps(age), keep)
                builder.add_recruits(age, [recruited.get(i) for i in range(game.n)])
            deck = build_age_deck(self.content, age, game.n, rng, game.rules.extra_guilds)
            builder.set_deck(age, [c.name for c in deck])
            yield from self._relay(game.age_steps(age, deal_hands(deck, game.n)), self._record_turn)
        game.phase = Phase.FINISHED

    def _relay(
        self,
        steps: Generator[DecisionPoint, Dict[int, Action], None],
        record: Optional[Callable[[DecisionPoint, Dict[int, Action]], None]] = None,
    ) -> Generator[DecisionPoint, Dict[int, Action], None]:
        """Pass an engine coroutine's decision points through, calling ``record`` per answer."""
        try:
            point = next(steps)
        except StopIteration:
            return
        while True:
            actions = yield point
            if record is not None:
                record(point, actions)
            try:
                point = steps.send(actions)
            except StopIteration:
                return

    def _record_turn(self, point: DecisionPoint, actions: Dict[int, Action]) -> None:
        """Record a card-play decision in the GameRecord and the public history."""
        builder, n = self._builder, self.n_players
        assert builder is not None
        if point.kind is DecisionKind.PLAY:
            ordered = [actions[i] for i in range(n)]
            builder.add_turn(point.age, ordered)
            self._observations.record(point.age, point.turn, [(a.kind, a.card.name) for a in ordered])
        else:
            builder.add_extra(point, actions)
            moves = [(actions[i].kind, actions[i].card.name) if i in actions else None for i in range(n)]
            self._observations.record(point.age, point.turn, moves)

    def _observe(self) -> Tuple[Dict[int, EnvObs], Dict[int, Dict[str, Any]]]:
        observations: Dict[int, EnvObs] = {}
        infos: Dict[int, Dict[str, Any]] = {}
        self._obs_cache = {}
        for seat in self.acting:
            obs = self.observation(seat)
            if self.encode_seats is None or seat in self.encode_seats:
                observations[seat] = {
                    'observation': self.encoder.encode(obs),
                    'action_mask': self.action_space.mask(obs.legal_actions),
                }
            infos[seat] = {'obs': obs, 'decision': obs.decision}
        return observations, infos

    def _rewards(self, terminal: bool) -> Dict[int, float]:
        scores = self.scores()
        n = self.n_players
        if self.reward_mode in ('dense', 'dense_rank'):
            rewards = {}
            for i in range(n):
                margin = scores[i] - sum(s for j, s in enumerate(scores) if j != i) / (n - 1)
                rewards[i] = margin - self._last_margin[i]
                self._last_margin[i] = margin
            if terminal and self.reward_mode == 'dense_rank':
                for place, i in enumerate(self._placing(scores)):
                    rewards[i] += self.rank_bonus * (1.0 - 2.0 * place / (n - 1))
            return rewards
        if not terminal:
            return {i: 0.0 for i in range(n)}
        if self.reward_mode == 'score':
            return {i: float(scores[i]) for i in range(n)}
        if self.reward_mode == 'margin':
            return {i: float(scores[i] - max(s for j, s in enumerate(scores) if j != i)) for i in range(n)}
        winner = self._require_game().winner()
        return {i: 1.0 if i == winner else -1.0 / (n - 1) for i in range(n)}

    def _placing(self, scores: List[int]) -> List[int]:
        """Seats from first to last place (score, then coins)."""
        game = self._require_game()
        return sorted(range(self.n_players), key=lambda i: (-scores[i], -game.players[i].coins))


class SingleAgentEnv:
    """Gym-style single-agent view: seat ``learner`` learns, ``opponents`` fill the rest.

    ``reset(seed) -> (obs, info)`` and ``step(index) -> (obs, reward, terminated,
    truncated, info)`` where ``obs`` is ``{'observation', 'action_mask'}`` for the
    learner. Decisions that do not involve the learner (e.g. an opponent's discard-pile
    build) are played out automatically; rewards earned meanwhile are accumulated.
    """

    def __init__(self, opponents: Sequence[Agent], learner: int = 0, **env_kwargs: Any):
        n_players = len(opponents) + 1
        if not 0 <= learner < n_players:
            raise ValueError('learner seat out of range')
        self.env = WondersEnv(n_players=n_players, **env_kwargs)
        self.env.encode_seats = {learner}  # opponents act on raw Observations
        self.learner = learner
        others = iter(opponents)
        self.agents: Dict[int, Agent] = {seat: next(others) for seat in range(n_players) if seat != learner}
        self.action_space = self.env.action_space

    def reset(self, seed: Optional[int] = None) -> Tuple[EnvObs, Dict[str, Any]]:
        observations, infos = self.env.reset(seed)
        observations, infos, _reward = self._advance(observations, infos, None)
        return observations[self.learner], infos[self.learner]

    def step(self, index: int) -> Tuple[EnvObs, float, bool, bool, Dict[str, Any]]:
        if self.learner not in self.env.acting:
            raise RuntimeError('the game is over; call reset()')
        observations, infos, reward = self._advance({}, {}, index)
        terminated = self.env.done
        if terminated:
            info = {'scores': self.env.scores(), 'record': self.env.record()}
            return {}, reward, True, False, info
        return observations[self.learner], reward, False, False, infos[self.learner]

    def _advance(
        self, observations: Dict[int, EnvObs], infos: Dict[int, Dict[str, Any]], index: Optional[int]
    ) -> Tuple[Dict[int, EnvObs], Dict[int, Dict[str, Any]], float]:
        """Step until the learner must act (or the game ends), using ``index`` once."""
        reward = 0.0
        while not self.env.done and (index is not None or self.learner not in self.env.acting):
            actions: Dict[int, int] = {}
            for seat in self.env.acting:
                if seat == self.learner:
                    assert index is not None
                    actions[seat] = index
                    index = None
                else:
                    obs = self.env.observation(seat)
                    options = self.env.options(seat)
                    actions[seat] = self.action_space.index(self.agents[seat].act(obs, options))
            observations, rewards, _term, _trunc, infos = self.env.step(actions)
            reward += rewards[self.learner]
        return observations, infos, reward
