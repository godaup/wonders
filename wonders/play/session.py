"""One human seat against agents, stepped one human decision at a time.

:class:`HumanSession` wraps a :class:`~wonders.env.environment.WondersEnv`: agent seats are
played out automatically until the human must decide, and :meth:`HumanSession.state`
returns a JSON-ready view built only from the human's own
:class:`~wonders.engine.observation.Observation`, so the UI sees exactly what the seat may
legally observe (opponents' hands and buried/discarded cards stay hidden). Final scores of
every seat are revealed once the game is over.

Every decision of every seat is kept in :attr:`HumanSession.log`, so :meth:`HumanSession.saved`
gives a JSON game file for :class:`~wonders.storage.Store` and :meth:`HumanSession.restore` rebuilds
the game from it — the same seed deals the same game, and replaying the logged decisions (instead
of asking the agents again) reproduces it even after the agents changed.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from wonders.agents.base import Agent
from wonders.agents.bench import AGENTS, CATEGORIES, MAX_PLAYERS, MIN_PLAYERS, agent_name
from wonders.cards.content import Content, load_content
from wonders.engine.actions import Action, ActionKind, DecisionKind
from wonders.engine.observation import Observation
from wonders.engine.record import PlayerSetup
from wonders.env.environment import WondersEnv
from wonders.i18n import ENGLISH, Translator, languages
from wonders.play.rating import places
from wonders.ruleset import EXPANSIONS, Ruleset
from wonders.storage import new_id, now
from wonders.viz.render import ICONS, RES_COLOR, RES_ICON, cost_text, effect_lines, res_name

HUMAN = 'human'
SIDES = ('A', 'B', 'random')

# Short descriptions for the game menu (agents without an entry show no blurb).
AGENT_BLURBS: Dict[str, str] = {
    'random': 'Plays any legal move.',
    'greedy': 'Takes the most points right now.',
    'hoarder': 'Discards for coins.',
    'militarist': 'Favours red cards.',
    'scientist': 'Favours green cards.',
    'civic': 'Favours blue cards.',
    'economist': 'Favours yellow cards.',
    'builder': 'Rushes its wonder.',
    'heuristic': 'Tuned value function; a solid baseline.',
    'opportunist': 'Switches colour focus every few turns.',
    'blocker': 'Denies its neighbour the best card.',
    'table': 'Best affordable card by a fixed value table.',
    'balanced': 'Fills its scarcest colour.',
    'board': 'Plans around its wonder board; the strongest scripted agent.',
    'board-focus': 'Board-aware, favouring the colours its board points to.',
    'adaptive': 'Board-aware; contests winnable military fronts, adapts to the game phase and the wheel.',
    'rival': 'Adaptive, and plays against whoever is leading.',
}


@dataclass(frozen=True)
class AgentOption:
    """An agent the menu offers: a built-in name or a neural checkpoint."""

    id: str
    label: str
    blurb: str = ''
    checkpoint: Optional[str] = None  # neural agents only


def agent_options(checkpoints: Sequence[str] = (), t: Translator = ENGLISH) -> List[AgentOption]:
    """Built-in agents, plus neural agents when PyTorch is installed (blurbs translated by ``t``).

    ``checkpoints`` adds extra neural agents (``neural:<file stem>``) next to the bundled one.
    """
    from wonders.extensions import ensure_loaded

    registry = ensure_loaded()
    blurbs = {**AGENT_BLURBS, **registry.agent_blurbs}
    options = [AgentOption(name, agent_name(name), t(blurbs[name]) if blurbs.get(name) else '') for name in AGENTS]
    try:
        from wonders.ml import pretrained_path
    except ImportError:  # no PyTorch: scripted agents only
        return options
    paths = [pretrained_path(), *checkpoints]
    for i, path in enumerate(dict.fromkeys(os.path.abspath(p) for p in paths)):
        stem = os.path.splitext(os.path.basename(path))[0]
        label = 'neural' if i == 0 else f'neural:{stem}'
        blurb = t('The bundled trained network.') if i == 0 else t('Trained network from {path}.', path=path)
        options.append(AgentOption(label, agent_name(label), blurb, path))
    for model in registry.models.values():  # networks from extensions
        label = f'neural:{model.name}'
        if all(o.id != label for o in options):
            options.append(
                AgentOption(
                    label,
                    label,
                    t(model.description) if model.description else t('Network from {name}.', name=model.extension),
                    model.path,
                )
            )
    return options


@dataclass
class SeatConfig:
    """One seat: the human or an agent id from :func:`agent_options`."""

    player: str = HUMAN
    name: str = ''


@dataclass
class GameConfig:
    """Everything the game menu chooses.

    Attributes:
        expansions: Expansion ids (``base`` is implied).
        seats: One entry per seat, exactly one of them ``human``.
        wonder: The human's wonder name, or ``'random'``.
        side: The human's wonder side (``'A'``, ``'B'`` or ``'random'``).
        opponent_sides: Wonder side for the agent seats.
        seed: Seeds decks, wonders and stochastic agents (``None`` = random).
    """

    expansions: List[str] = field(default_factory=lambda: ['base'])
    seats: List[SeatConfig] = field(
        default_factory=lambda: [SeatConfig(), SeatConfig('heuristic'), SeatConfig('board')]
    )
    wonder: str = 'random'
    side: str = 'random'
    opponent_sides: str = 'random'
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameConfig':
        seats = [SeatConfig(str(s.get('player', HUMAN)), str(s.get('name', '') or '')) for s in data.get('seats', [])]
        seed = data.get('seed')
        return cls(
            expansions=[str(e) for e in data.get('expansions', ['base'])],
            seats=seats or cls().seats,
            wonder=str(data.get('wonder', 'random')),
            side=str(data.get('side', 'random')),
            opponent_sides=str(data.get('opponent_sides', 'random')),
            seed=None if seed is None or seed == '' else int(seed),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'expansions': list(self.expansions),
            'seats': [{'player': s.player, 'name': s.name} for s in self.seats],
            'wonder': self.wonder,
            'side': self.side,
            'opponent_sides': self.opponent_sides,
            'seed': self.seed,
        }


def _asset_version() -> str:
    """A short hash of the card renderer and the card data, so browsers refetch changed images."""
    import hashlib

    from wonders.viz import render

    digest = hashlib.sha1(Path(render.__file__).read_bytes())
    for exp in EXPANSIONS.values():
        if exp.implemented:
            digest.update(repr([(c.name, c.effects) for c in load_content(exp.id).cards]).encode())
    return digest.hexdigest()[:10]


def menu_options(checkpoints: Sequence[str] = (), t: Translator = ENGLISH) -> Dict[str, Any]:
    """What the game menu can offer: expansions, agents, wonders per expansion, seat limits, languages."""
    from wonders.extensions import ensure_loaded

    ensure_loaded()  # extension expansions join the menu
    expansions = []
    for exp in EXPANSIONS.values():
        wonders: List[str] = []
        if exp.implemented:
            content = load_content(exp.id)
            wonders = list(dict.fromkeys(w.name for w in content.wonders))
        expansions.append(
            {
                'id': exp.id,
                'name': t(exp.name, 'expansion'),
                'implemented': exp.implemented,
                'note': t(exp.note) if exp.note else '',
                'wonders': wonders,
            }
        )
    return {
        'expansions': expansions,
        'agents': [{'id': a.id, 'label': a.label, 'blurb': a.blurb} for a in agent_options(checkpoints, t)],
        'min_players': MIN_PLAYERS,
        'max_players': MAX_PLAYERS,
        'sides': list(SIDES),
        'defaults': GameConfig().to_dict(),
        # The card renderer's icons (24x24, currentColor), so the page draws the same symbols.
        'icons': dict(ICONS),
        'asset_version': _asset_version(),  # cache-busts card / board images when the renderer changes
        'resources': {r.value: {'icon': RES_ICON[r], 'color': RES_COLOR[r], 'name': res_name(r, t)} for r in RES_ICON},
        'languages': languages(),
        'lang': t.lang,
    }


class ConfigError(ValueError):
    """The chosen game configuration cannot be played."""


def _neural_agent(option: AgentOption, content: Content, ruleset: Ruleset, rng: random.Random) -> Agent:
    from wonders.ml import NeuralAgent

    agent = NeuralAgent(option.checkpoint, rng=rng, name=option.id)
    missing = {c.name for c in content.cards} - set(agent.vocab.card_names)
    if missing:
        raise ConfigError(f'{option.label} was not trained with these expansions (unknown cards, e.g. {min(missing)})')
    if ruleset.uses('leaders') and 'draft' not in agent.vocab.decisions:
        raise ConfigError(f'{option.label} was not trained for the Leaders draft')
    if agent.vocab.wonders:
        unknown = {w.name for w in content.wonders} - set(agent.vocab.wonders)
        if unknown:
            raise ConfigError(f'{option.label} does not know the wonder {min(unknown)}')
    return agent


class HumanSession:
    """A game with exactly one human seat; the other seats are agents.

    Args:
        config: The menu choices.
        checkpoints: Extra neural checkpoints the config may refer to (``neural:<stem>``).
        on_progress: Called after every decision the agents play out (for live UIs).
        t: Translates the default seat names (``You``).
        game_id: The id it is saved under (a new one by default).
        log: Decisions to replay before play continues (see :meth:`restore`).
    """

    def __init__(
        self,
        config: GameConfig,
        checkpoints: Sequence[str] = (),
        on_progress: Optional[Callable[[], None]] = None,
        t: Translator = ENGLISH,
        game_id: Optional[str] = None,
        log: Sequence[Dict[str, Any]] = (),
    ):
        self.config = config
        self.t = t
        self.id = game_id or new_id()
        self.created = now()
        self.finished_at: Optional[str] = None
        self.traded = 0  # coins the human paid to its neighbours
        self.log: List[Dict[str, Any]] = []  # every step: {'decision': kind, 'moves': {seat: move}}
        seats = config.seats
        humans = [i for i, s in enumerate(seats) if s.player == HUMAN]
        if len(humans) != 1:
            raise ConfigError('choose exactly one human seat')
        if not MIN_PLAYERS <= len(seats) <= MAX_PLAYERS:
            raise ConfigError(f'need {MIN_PLAYERS}-{MAX_PLAYERS} players, got {len(seats)}')
        for value in (config.side, config.opponent_sides):
            if value not in SIDES:
                raise ConfigError(f'unknown wonder side {value!r}')
        try:
            self.ruleset = Ruleset.of(*config.expansions)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        if self.ruleset.unimplemented:
            raise ConfigError(f'not playable yet: {", ".join(self.ruleset.unimplemented)}')
        self.human = humans[0]
        self.seed = config.seed if config.seed is not None else random.randrange(2**31)
        rng = random.Random(self.seed)
        self.content = load_content(*self.ruleset.content_expansions)
        self.cards = {c.name: c for c in self.content.cards}
        self.setups = self._setups(rng)
        options = {a.id: a for a in agent_options(checkpoints)}
        self.agents: Dict[int, Agent] = {}
        for i, seat in enumerate(seats):
            if i == self.human:
                continue
            option = options.get(seat.player)
            if option is None:
                raise ConfigError(f'unknown agent {seat.player!r}')
            agent_rng = random.Random(rng.random())
            if option.checkpoint is not None:
                self.agents[i] = _neural_agent(option, self.content, self.ruleset, agent_rng)
            else:
                self.agents[i] = AGENTS[option.id].build(agent_rng)
        self.env = WondersEnv(
            n_players=len(seats), expansions=self.ruleset.expansions, content=self.content, setups=self.setups
        )
        self.env.encode_seats = set()  # nobody needs tensors here; agents read raw Observations
        self.on_progress = on_progress
        self.env.reset(rng.randrange(2**31))
        for step in log:
            self._replay(step)
        self._advance()

    # -- setup -----------------------------------------------------------------
    def _setups(self, rng: random.Random) -> List[PlayerSetup]:
        config, n = self.config, len(self.config.seats)
        names = list(dict.fromkeys(w.name for w in self.content.wonders))
        if config.wonder != 'random' and config.wonder not in names:
            raise ConfigError(f'unknown wonder {config.wonder!r} for these expansions')
        if n > len(names):
            raise ConfigError(f'only {len(names)} wonders for {n} players')
        pool = [w for w in names if w != config.wonder]
        rng.shuffle(pool)
        human_wonder = config.wonder if config.wonder != 'random' else pool.pop()
        setups = []
        for i, seat in enumerate(config.seats):
            if i == self.human:
                wonder, side = human_wonder, config.side
            else:
                wonder, side = pool.pop(), config.opponent_sides
            side = rng.choice('AB') if side == 'random' else side
            twins = sum(s.player == seat.player for s in config.seats) > 1  # number agents that sit twice
            default = agent_name(seat.player) + (f' {i + 1}' if twins else '')
            name = seat.name or (self.t('You') if i == self.human else default)
            setups.append(PlayerSetup(name, wonder, side))
        return setups

    # -- play ------------------------------------------------------------------
    @property
    def done(self) -> bool:
        return self.env.done

    def _advance(self) -> None:
        """Let the agents play until the human must decide or the game ends."""
        env = self.env
        while not env.done and self.human not in env.acting:
            self._step({seat: self._choose(seat) for seat in env.acting})
            if self.on_progress is not None:
                self.on_progress()

    def _choose(self, seat: int) -> Action:
        """The agent's move for an acting seat."""
        env = self.env
        return self.agents[seat].act(env.observation(seat), env.options(seat))

    def _step(self, actions: Dict[int, Action]) -> None:
        """Play one step and log it (and count the coins the human pays its neighbours)."""
        decision = self.env.decision
        assert decision is not None
        self.log.append({'decision': decision.value, 'moves': {str(s): _move(a) for s, a in sorted(actions.items())}})
        mine = actions.get(self.human)
        if mine is not None and mine.kind in (ActionKind.BUILD, ActionKind.WONDER):
            options = self._payments(mine)
            chosen = [p for p in options if mine.payment is not None and (p.left, p.right) == mine.payment]
            if options:
                self.traded += (chosen or options)[0].traded
        self.env.step(actions)

    def _replay(self, step: Dict[str, Any]) -> None:
        """Play a logged step again (restoring a saved game)."""
        env = self.env
        moves = {int(seat): move for seat, move in step['moves'].items()}
        if env.done or set(moves) != set(env.acting) or env.decision != DecisionKind(step['decision']):
            raise ConfigError('the saved game no longer matches the rules: it cannot be restored')
        actions = {}
        for seat, move in moves.items():
            kind, card = ActionKind(move[0]), move[1]
            found = next(
                (a for a in env.observation(seat).legal_actions if a.kind is kind and a.card.name == card), None
            )
            if found is None:
                raise ConfigError(f'the saved game no longer matches the rules ({kind.value} {card} is not legal)')
            actions[seat] = found.paying(int(move[2]), int(move[3])) if len(move) > 3 else found
        self._step(actions)

    def observation(self) -> Optional[Observation]:
        """The human's current Observation (``None`` once the game is over)."""
        return None if self.done else self.env.observation(self.human)

    def act(self, index: int, payment: Optional[Sequence[int]] = None) -> None:
        """Play the human's ``index``-th legal action (as listed in :meth:`state`).

        ``payment`` optionally chooses how to split the purchase between the neighbours, as
        ``(left, right)`` coins — one of the action's ``payments`` in :meth:`state`.
        """
        obs = self.observation()
        if obs is None:
            raise ConfigError('the game is over')
        legal = obs.legal_actions
        if not 0 <= index < len(legal):
            raise ConfigError(f'no legal action #{index}')
        action = legal[index]
        if payment is not None:
            split = (int(payment[0]), int(payment[1]))
            options = self._payments(action)
            if split not in {(p.left, p.right) for p in options}:
                raise ConfigError(f'cannot pay {split[0]} left / {split[1]} right for {action.card.name}')
            if (options[0].left, options[0].right) != split:  # the cheapest is the default anyway
                action = action.paying(*split)
        choices = {seat: self._choose(seat) for seat in self.env.acting if seat != self.human}
        self._step({**choices, self.human: action})
        self._advance()
        if self.done and self.finished_at is None:
            self.finished_at = now()

    def _payments(self, action: Action):
        game = self.env.game
        assert game is not None
        return game.payment_options(self.human, action)

    # -- views -----------------------------------------------------------------
    def card_info(self, name: str, t: Translator = ENGLISH) -> Dict[str, Any]:
        """Colour, Age and tooltip lines of a card (``title`` is its translated name)."""
        card = self.cards.get(name)
        if card is None:
            return {'name': name, 'title': t.name(name), 'color': 'unknown'}
        text = [t('Cost: {cost}', cost=cost_text(card.cost.coins, card.cost.resources, t))]
        text += effect_lines(card.effects, t)
        if card.chain_requires:
            text.append(t('Free if you own {card}', card=t.name(card.chain_requires)))
        if card.chain_enables:
            text.append(t('Unlocks {cards}', cards=', '.join(t.name(c) for c in card.chain_enables)))
        return {'name': name, 'title': t.name(name), 'color': card.color.value, 'age': card.age, 'text': text}

    def boards(self, built: Sequence[int], t: Translator = ENGLISH) -> List[Dict[str, Any]]:
        """Every seat's wonder board (public): starting resource and stages, ``built[seat]`` built."""
        out = []
        for seat, setup in enumerate(self.setups):
            wonder = self.content.wonder(setup.wonder, setup.side)
            res = wonder.starting_resource
            out.append(
                {
                    'seat': seat,
                    'title': t.name(setup.wonder),
                    'produces': res_name(res, t) if res is not None else t('nothing'),
                    'powers': effect_lines(wonder.effects, t),
                    'stages': [
                        {
                            'index': k + 1,
                            'cost': cost_text(stage.cost.coins, stage.cost.resources, t),
                            'effects': effect_lines(stage.effects, t),
                            'built': k < built[seat],
                        }
                        for k, stage in enumerate(wonder.stages)
                    ],
                }
            )
        return out

    def _actions(self, obs: Observation) -> List[Dict[str, Any]]:
        views = {c.name: c for c in obs.hand}
        out = []
        for i, action in enumerate(obs.legal_actions):
            view = views.get(action.card.name)
            entry: Dict[str, Any] = {'index': i, 'kind': action.kind.value, 'card': action.card.name, 'coins': 0}
            if action.kind.value == 'build' and view is not None and view.build_coin_cost is not None:
                entry.update(coins=view.build_coin_cost, bank=view.bank_coins, trade=view.trade_coins)
                entry['chain'] = view.chain_free
            if action.kind.value in ('build', 'wonder') and obs.decision == 'play':
                payments = self._payments(action)
                if action.kind.value == 'wonder' and payments and obs.next_wonder_stage is not None:
                    stage_coins = int(obs.next_wonder_stage.cost.get('coins', 0) or 0)  # type: ignore[call-overload]
                    entry.update(coins=stage_coins + payments[0].total, trade=payments[0].traded)
                if len(payments) > 1:  # a real choice of whom to buy from
                    entry['payments'] = [{'left': p.left, 'right': p.right, 'total': p.total} for p in payments]
            out.append(entry)
        return out

    def state(self, t: Translator = ENGLISH) -> Dict[str, Any]:
        """A JSON-ready view for the human seat (card, board and wonder texts translated by ``t``)."""
        game = self.env.game
        assert game is not None
        state: Dict[str, Any] = {
            'id': self.id,
            'config': self.config.to_dict(),
            'seed': self.seed,
            'human': self.human,
            'expansions': list(self.ruleset.expansions),
            'players': [
                {
                    'seat': i,
                    'name': s.name,
                    'wonder': s.wonder,
                    'wonder_title': t.name(s.wonder),
                    'side': s.side,
                    'agent': self.config.seats[i].player,
                }
                for i, s in enumerate(self.setups)
            ],
            'done': self.done,
        }
        obs = self.observation()
        if obs is not None:
            view = obs.to_dict()
            view['actions'] = self._actions(obs)
            state['observation'] = view
            state['boards'] = self.boards([city.stages_built for city in obs.table], t)
            cards = {c.name for c in obs.hand}
            cards.update(n for city in obs.table for n in city.cards)
            cards.update(e.card for e in obs.history if e.card)
            cards.update(obs.leaders_hand)
            state['cards'] = {n: self.card_info(n, t) for n in sorted(cards)}
        else:
            scores = [game.score(i).as_dict() for i in range(game.n)]
            winner = game.winner()
            state['result'] = {'scores': scores, 'categories': list(CATEGORIES), 'winner': winner}
            state['cards'] = {n: self.card_info(n, t) for p in game.players for n in (c.name for c in p.cards)}
            state['cities'] = [
                {
                    'seat': i,
                    'cards': [c.name for c in p.cards],
                    'coins': p.coins,
                    'stages_built': p.wonder_stages_built,
                    'tokens': list(p.tokens),
                }
                for i, p in enumerate(game.players)
            ]
        return state

    def record_json(self) -> str:
        """The replayable :class:`~wonders.engine.record.GameRecord` so far, as JSON."""
        return self.env.record().to_json(indent=1)

    # -- saving ----------------------------------------------------------------
    def result(self) -> Optional[Dict[str, Any]]:
        """Final facts of every seat (``None`` while the game runs): what ratings and achievements read."""
        game = self.env.game
        if not self.done or game is None:
            return None
        totals = [game.score(i).total for i in range(game.n)]
        coins = [p.coins for p in game.players]
        builds, discards = [0] * game.n, [0] * game.n
        for step in self.log:
            for seat, move in step['moves'].items():
                if step['decision'] == DecisionKind.DISCARD_BUILD.value:
                    builds[int(seat)] += move[0] == ActionKind.BUILD.value
                discards[int(seat)] += move[0] == ActionKind.DISCARD.value
        return {
            'totals': totals,
            'coins': coins,
            'places': places(totals, coins),
            'winner': game.winner(),
            'scores': [game.score(i).as_dict() for i in range(game.n)],
            'colors': [_colors(p.cards) for p in game.players],
            'cards': [len(p.cards) for p in game.players],
            'chains': [_chained(p.cards) for p in game.players],
            'stages': [p.wonder_stages_built for p in game.players],
            'stages_total': [len(p.wonder.stages) for p in game.players],
            'tokens': [list(p.tokens) for p in game.players],
            'debt': [p.debt for p in game.players],
            'science': [{s.value: n for s, n in p.science().items()} for p in game.players],
            'discards': discards,
            'discard_builds': builds,
            'traded': self.traded,  # the human's coins paid to neighbours (agents' payments are not tracked)
        }

    def progress(self) -> Optional[Dict[str, Any]]:
        """Where an unfinished game stands (Age, turn, decision); ``None`` once it is over."""
        obs = self.observation()
        return None if obs is None else {'age': obs.age, 'turn': obs.turn, 'decision': obs.decision}

    def saved(self) -> Dict[str, Any]:
        """The game as a :class:`~wonders.storage.Store` file: config, players, log, record, result."""
        return {
            'id': self.id,
            'created': self.created,
            'finished': self.done,
            'finished_at': self.finished_at,
            'config': {**self.config.to_dict(), 'seed': self.seed},
            'human': self.human,
            'expansions': list(self.ruleset.expansions),
            'players': [
                {'name': s.name, 'wonder': s.wonder, 'side': s.side, 'agent': self.config.seats[i].player}
                for i, s in enumerate(self.setups)
            ],
            'progress': self.progress(),
            'result': self.result(),
            'log': self.log,
            'record': self.env.record().to_dict(),
        }

    @classmethod
    def restore(
        cls,
        saved: Dict[str, Any],
        checkpoints: Sequence[str] = (),
        on_progress: Optional[Callable[[], None]] = None,
        t: Translator = ENGLISH,
    ) -> 'HumanSession':
        """Rebuild a game from :meth:`saved`: the same deal, every logged decision replayed."""
        session = cls(
            GameConfig.from_dict(saved['config']),
            checkpoints,
            on_progress,
            t,
            game_id=str(saved['id']),
            log=saved.get('log', []),
        )
        session.created = str(saved.get('created') or session.created)
        session.finished_at = saved.get('finished_at') or (session.created if session.done else None)
        return session


def _move(action: Action) -> List[Any]:
    """A logged move: ``[kind, card]``, plus ``left, right`` coins when the payment was chosen."""
    move: List[Any] = [action.kind.value, action.card.name]
    if action.payment is not None:
        move += list(action.payment)
    return move


def _chained(cards: Sequence[Any]) -> int:
    """Cards built for free through a chain (their prerequisite is in the same city)."""
    names = {card.name for card in cards}
    return sum(card.chain_requires in names for card in cards if card.chain_requires)


def _colors(cards: Sequence[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for card in cards:
        counts[card.color.value] = counts.get(card.color.value, 0) + 1
    return counts


def legal_action(session: HumanSession, kind: str, card: str) -> int:
    """Index of the human's legal action ``kind`` on ``card`` (convenience for tests/scripts)."""
    obs = session.observation()
    if obs is None:
        raise ConfigError('the game is over')
    for i, action in enumerate(obs.legal_actions):
        if action.kind.value == kind and action.card.name == card:
            return i
    raise ConfigError(f'{kind} {card} is not legal now')


__all__ = [
    'AgentOption',
    'ConfigError',
    'GameConfig',
    'HUMAN',
    'HumanSession',
    'SeatConfig',
    'agent_options',
    'legal_action',
    'menu_options',
]
