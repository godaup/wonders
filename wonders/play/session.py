"""One human seat against agents, stepped one human decision at a time.

:class:`HumanSession` wraps a :class:`~wonders.env.environment.WondersEnv`: agent seats are
played out automatically until the human must decide, and :meth:`HumanSession.state`
returns a JSON-ready view built only from the human's own
:class:`~wonders.engine.observation.Observation`, so the UI sees exactly what the seat may
legally observe (opponents' hands and buried/discarded cards stay hidden). Final scores of
every seat are revealed once the game is over.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from wonders.agents.base import Agent
from wonders.agents.bench import AGENTS, CATEGORIES, MAX_PLAYERS, MIN_PLAYERS
from wonders.cards.content import Content, load_content
from wonders.engine.actions import Action
from wonders.engine.observation import Observation
from wonders.engine.record import PlayerSetup
from wonders.env.environment import WondersEnv
from wonders.ruleset import EXPANSIONS, Ruleset

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


def agent_options(checkpoints: Sequence[str] = ()) -> List[AgentOption]:
    """Built-in agents, plus neural agents when PyTorch is installed.

    ``checkpoints`` adds extra neural agents (``neural:<file stem>``) next to the bundled one.
    """
    options = [AgentOption(name, name, AGENT_BLURBS.get(name, '')) for name in AGENTS]
    try:
        from wonders.ml import pretrained_path
    except ImportError:  # no PyTorch: scripted agents only
        return options
    paths = [pretrained_path(), *checkpoints]
    for i, path in enumerate(dict.fromkeys(os.path.abspath(p) for p in paths)):
        stem = os.path.splitext(os.path.basename(path))[0]
        label = 'neural' if i == 0 else f'neural:{stem}'
        blurb = 'The bundled trained network.' if i == 0 else f'Trained network from {path}.'
        options.append(AgentOption(label, label, blurb, path))
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


def menu_options(checkpoints: Sequence[str] = ()) -> Dict[str, Any]:
    """What the game menu can offer: expansions, agents, wonders per expansion, seat limits."""
    expansions = []
    for exp in EXPANSIONS.values():
        wonders: List[str] = []
        if exp.implemented:
            content = load_content(exp.id)
            wonders = list(dict.fromkeys(w.name for w in content.wonders))
        expansions.append(
            {'id': exp.id, 'name': exp.name, 'implemented': exp.implemented, 'note': exp.note, 'wonders': wonders}
        )
    return {
        'expansions': expansions,
        'agents': [{'id': a.id, 'label': a.label, 'blurb': a.blurb} for a in agent_options(checkpoints)],
        'min_players': MIN_PLAYERS,
        'max_players': MAX_PLAYERS,
        'sides': list(SIDES),
        'defaults': GameConfig().to_dict(),
    }


class ConfigError(ValueError):
    """The chosen game configuration cannot be played."""


def _neural_agent(option: AgentOption, content: Content, ruleset: Ruleset, rng: random.Random) -> Agent:
    from wonders.ml import NeuralAgent

    agent = NeuralAgent(option.checkpoint, rng=rng, name=option.label)
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
    """

    def __init__(
        self,
        config: GameConfig,
        checkpoints: Sequence[str] = (),
        on_progress: Optional[Callable[[], None]] = None,
    ):
        self.config = config
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
            name = seat.name or ('You' if i == self.human else f'{seat.player.capitalize()} {i}')
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
            env.step({seat: self._choose(seat) for seat in env.acting})
            if self.on_progress is not None:
                self.on_progress()

    def _choose(self, seat: int, human: Optional[Action] = None) -> int:
        """Action index for an acting seat: the human's ``human`` move, or the agent's choice."""
        env = self.env
        action = human if seat == self.human else self.agents[seat].act(env.observation(seat), env.options(seat))
        assert action is not None
        return env.action_space.index(action)

    def observation(self) -> Optional[Observation]:
        """The human's current Observation (``None`` once the game is over)."""
        return None if self.done else self.env.observation(self.human)

    def act(self, index: int) -> None:
        """Play the human's ``index``-th legal action (as listed in :meth:`state`)."""
        obs = self.observation()
        if obs is None:
            raise ConfigError('the game is over')
        legal = obs.legal_actions
        if not 0 <= index < len(legal):
            raise ConfigError(f'no legal action #{index}')
        self.env.step({seat: self._choose(seat, legal[index]) for seat in self.env.acting})
        self._advance()

    # -- views -----------------------------------------------------------------
    def card_info(self, name: str) -> Dict[str, Any]:
        card = self.cards.get(name)
        if card is None:
            return {'name': name, 'color': 'unknown'}
        return {'name': name, 'color': card.color.value, 'age': card.age}

    def _actions(self, obs: Observation) -> List[Dict[str, Any]]:
        views = {c.name: c for c in obs.hand}
        out = []
        for i, action in enumerate(obs.legal_actions):
            view = views.get(action.card.name)
            entry: Dict[str, Any] = {'index': i, 'kind': action.kind.value, 'card': action.card.name, 'coins': 0}
            if action.kind.value == 'build' and view is not None and view.build_coin_cost is not None:
                entry.update(coins=view.build_coin_cost, bank=view.bank_coins, trade=view.trade_coins)
                entry['chain'] = view.chain_free
            out.append(entry)
        return out

    def state(self) -> Dict[str, Any]:
        """A JSON-ready view for the human seat."""
        game = self.env.game
        assert game is not None
        state: Dict[str, Any] = {
            'config': self.config.to_dict(),
            'seed': self.seed,
            'human': self.human,
            'expansions': list(self.ruleset.expansions),
            'players': [
                {'seat': i, 'name': s.name, 'wonder': s.wonder, 'side': s.side, 'agent': self.config.seats[i].player}
                for i, s in enumerate(self.setups)
            ],
            'done': self.done,
        }
        obs = self.observation()
        if obs is not None:
            view = obs.to_dict()
            view['actions'] = self._actions(obs)
            state['observation'] = view
            cards = {c.name for c in obs.hand}
            cards.update(n for city in obs.table for n in city.cards)
            cards.update(e.card for e in obs.history if e.card)
            state['cards'] = {n: self.card_info(n) for n in sorted(cards)}
        else:
            scores = [game.score(i).as_dict() for i in range(game.n)]
            winner = game.winner()
            state['result'] = {'scores': scores, 'categories': list(CATEGORIES), 'winner': winner}
            state['cards'] = {n: self.card_info(n) for p in game.players for n in (c.name for c in p.cards)}
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
