"""Achievements: feats a player earns, derived from the saved games.

Most achievements are *feats* — a predicate over one finished saved game (see
:meth:`~wonders.play.session.HumanSession.saved`) that reads only its ``result`` facts (scores per
category, card colours, military tokens, chains, discards, …), the players and the expansions.
*Career* achievements look at all finished games in order (and the rating after each) and report
progress towards a goal, e.g. wins with four different wonders.

Nothing is stored separately: :func:`achievements` checks every finished game, so a newly added
achievement also counts the games played before it existed. Each has a tier worth points
(:data:`POINTS`), so collecting them is a score too. To add one, append to :data:`ACHIEVEMENTS`
and translate its name and description.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

Game = Dict[str, Any]
# (progress, goal, index of the game that reached the goal or None)
Progress = Tuple[int, int, Optional[int]]

BASE_CATEGORIES = ('military', 'treasury', 'wonders', 'civilian', 'commercial', 'guilds', 'science')
BASE_COLORS = ('brown', 'grey', 'yellow', 'red', 'blue', 'green', 'purple')
TIERS = ('bronze', 'silver', 'gold', 'legendary')
POINTS = {'bronze': 10, 'silver': 25, 'gold': 50, 'legendary': 100}
GROUPS = ('general', 'leaders', 'cities', 'opponents', 'career')


@dataclass(frozen=True)
class Achievement:
    """A feat (``check(game, seat)``) or a career goal (``career(games, ratings)``)."""

    id: str
    name: str
    description: str
    icon: str
    tier: str
    group: str = 'general'
    check: Optional[Callable[[Game, int], bool]] = None
    career: Optional[Callable[[Sequence[Game], Sequence[float]], Progress]] = None


# -- facts ---------------------------------------------------------------------------------------
def _r(game: Game, key: str, seat: int, default: Any = 0) -> Any:
    values = game['result'].get(key)
    return default if values is None else values[seat]


def _won(game: Game, seat: int) -> bool:
    return bool(_r(game, 'places', seat) == 1)


def _margin(game: Game, seat: int) -> int:
    """Points ahead of the best opponent (negative when behind)."""
    totals = game['result']['totals']
    return int(totals[seat] - max(t for i, t in enumerate(totals) if i != seat))


def _color(game: Game, seat: int, color: str) -> int:
    return int(_r(game, 'colors', seat, {}).get(color, 0))


def _uses(game: Game, *expansions: str) -> bool:
    return all(e in game.get('expansions', ()) for e in expansions)


def _agents(game: Game, seat: int) -> List[str]:
    return [p['agent'] for i, p in enumerate(game['players']) if i != seat]


def _above(game: Game, seat: int) -> List[str]:
    """Agents the human finished above."""
    places = game['result']['places']
    return [p['agent'] for i, p in enumerate(game['players']) if i != seat and places[seat] < places[i]]


def _below(game: Game, seat: int) -> List[str]:
    places = game['result']['places']
    return [p['agent'] for i, p in enumerate(game['players']) if i != seat and places[seat] > places[i]]


# -- feats ---------------------------------------------------------------------------------------
def _wonderwall(g: Game, s: int) -> bool:
    return bool(_r(g, 'stages_total', s)) and _r(g, 'stages', s) >= _r(g, 'stages_total', s)


def _sparta(g: Game, s: int) -> bool:
    tokens = _r(g, 'tokens', s, [])
    return _won(g, s) and not any(t < 0 for t in tokens) and sum(t > 0 for t in tokens) >= 6


def _glass_jaw(g: Game, s: int) -> bool:
    return _won(g, s) and sum(t < 0 for t in _r(g, 'tokens', s, [])) >= 5


def _photo_finish(g: Game, s: int) -> bool:
    return _won(g, s) and _margin(g, s) <= 1


def _renaissance(g: Game, s: int) -> bool:
    scores = _r(g, 'scores', s, {})
    return all(scores.get(category, 0) >= 5 for category in BASE_CATEGORIES)


def _hall_of_fame(g: Game, s: int) -> bool:
    return _uses(g, 'leaders') and _color(g, s, 'white') >= 4


def _self_made(g: Game, s: int) -> bool:
    return _uses(g, 'leaders') and _won(g, s) and _color(g, s, 'white') == 0


def _last(g: Game, s: int) -> bool:
    places = g['result']['places']
    return len(places) >= 3 and places[s] == max(places) and places.count(places[s]) == 1


def _philosopher_king(g: Game, s: int) -> bool:
    science = _r(g, 'science', s, {})
    return _won(g, s) and min(science.get(k, 0) for k in ('tablet', 'gear', 'compass')) >= 3


def _boss_rush(g: Game, s: int) -> bool:
    return _won(g, s) and {'neural', 'adaptive', 'rival'} <= set(_agents(g, s))


FEATS: Sequence[Achievement] = (
    # general
    Achievement(
        'first_steps', 'Hello, World Wonder', 'Finish your first game.', '👋', 'bronze', check=lambda g, s: True
    ),
    Achievement('wonderwall', 'Wonderwall', 'Build every stage of your wonder.', '🏛', 'bronze', check=_wonderwall),
    Achievement(
        'full_spectrum',
        'Full Spectrum',
        'Build a brown, grey, yellow, red, blue, green and purple card in one game.',
        '🌈',
        'bronze',
        check=lambda g, s: all(_color(g, s, c) for c in BASE_COLORS),
    ),
    Achievement(
        'chain_reaction',
        'Chain Reaction',
        'Build four cards for free through chains in one game.',
        '⛓',
        'bronze',
        check=lambda g, s: _r(g, 'chains', s) >= 4,
    ),
    Achievement(
        'landslide',
        'Landslide',
        'Win by at least 15 points.',
        '⛰',
        'silver',
        check=lambda g, s: _won(g, s) and _margin(g, s) >= 15,
    ),
    Achievement(
        'dragons_hoard',
        "Dragon's Hoard",
        'Finish a game with at least 25 coins.',
        '🐉',
        'bronze',
        check=lambda g, s: _r(g, 'coins', s) >= 25,
    ),
    Achievement(
        'eureka',
        'Eureka!',
        'Score at least 30 points from science.',
        '🛁',
        'silver',
        check=lambda g, s: _r(g, 'scores', s, {}).get('science', 0) >= 30,
    ),
    Achievement(
        'union_boss',
        'Union Boss',
        'Build three guilds in one game.',
        '🤝',
        'silver',
        check=lambda g, s: _color(g, s, 'purple') >= 3,
    ),
    Achievement(
        'last_drachma',
        'Every Last Drachma',
        'Win with no coins left.',
        '🪙',
        'gold',
        check=lambda g, s: _won(g, s) and _r(g, 'coins', s) == 0,
    ),
    Achievement(
        'pacifist',
        'Make Wonders, Not War',
        'Win a game without building a single military (red) card.',
        '🕊',
        'gold',
        check=lambda g, s: _won(g, s) and _color(g, s, 'red') == 0,
    ),
    Achievement(
        'sparta',
        'This Is Sparta!',
        'Win the game and every military conflict in it (at least six victories, no defeat).',
        '🛡',
        'gold',
        check=_sparta,
    ),
    Achievement(
        'glass_jaw',
        'Glass Jaw, Golden Crown',
        'Win a game despite losing at least five military conflicts.',
        '🥊',
        'gold',
        check=_glass_jaw,
    ),
    Achievement(
        'autarky',
        'Autarky',
        'Win without paying your neighbours a single coin.',
        '🧺',
        'gold',
        check=lambda g, s: _won(g, s) and g['result'].get('traded') == 0,
    ),
    Achievement(
        'less_is_more',
        'Less Is More',
        'Win with at most 14 cards in your city.',
        '🍃',
        'gold',
        check=lambda g, s: _won(g, s) and 0 < _r(g, 'cards', s) <= 14,
    ),
    Achievement(
        'recycling',
        'Recycling Champion',
        'Discard at least four cards in one game and still win.',
        '♻',
        'gold',
        check=lambda g, s: _won(g, s) and _r(g, 'discards', s) >= 4,
    ),
    Achievement(
        'dumpster',
        'Dumpster Diver',
        'Build three cards from the discard pile in one game.',
        '🗑',
        'gold',
        check=lambda g, s: _r(g, 'discard_builds', s) >= 3,
    ),
    Achievement(
        'photo_finish',
        'Photo Finish',
        'Win by a single point, or on coins in a tie.',
        '📸',
        'silver',
        check=_photo_finish,
    ),
    Achievement(
        'seventy',
        'Seventy Wonders',
        'Score at least 70 points.',
        '🎆',
        'silver',
        check=lambda g, s: _r(g, 'totals', s) >= 70,
    ),
    Achievement(
        'renaissance',
        'Jack of All Trades, Master of Seven',
        'Score at least 5 points in each of military, treasury, wonders, civic, commercial, guilds and science '
        'in one game.',
        '🎭',
        'legendary',
        check=_renaissance,
    ),
    Achievement(
        'nerd_alert',
        'Nerd Alert',
        'Build three science (green) cards in one game.',
        '🤓',
        'bronze',
        check=lambda g, s: _color(g, s, 'green') >= 3,
    ),
    Achievement(
        'good_neighbour',
        'Good Neighbour',
        'Pay your neighbours at least 10 coins in one game. They will name a street after you.',
        '🎁',
        'bronze',
        check=lambda g, s: g['result'].get('traded', 0) >= 10,
    ),
    Achievement(
        'participation',
        'Participation Trophy',
        'Finish dead last. Somebody has to.',
        '🧸',
        'bronze',
        check=_last,
    ),
    Achievement(
        'philosopher_king',
        'Philosopher King',
        'Win with three complete science sets (three tablets, gears and compasses each).',
        '📜',
        'legendary',
        check=_philosopher_king,
    ),
    Achievement(
        'full_monty',
        'The Full Monty',
        'Win the game and every military conflict, and build every stage of your wonder.',
        '💯',
        'legendary',
        check=lambda g, s: _sparta(g, s) and _wonderwall(g, s),
    ),
    # Leaders
    Achievement(
        'name_dropper',
        'Name Dropper',
        'Recruit three leaders in one game.',
        '📇',
        'bronze',
        'leaders',
        check=lambda g, s: _uses(g, 'leaders') and _color(g, s, 'white') >= 3,
    ),
    Achievement(
        'hall_of_fame',
        'Hall of Fame',
        'Have four leaders in your city (only Rome makes room for a fourth).',
        '🏅',
        'silver',
        'leaders',
        check=_hall_of_fame,
    ),
    Achievement(
        'self_made',
        'Who Needs Leaders?',
        'Win a Leaders game without recruiting a single leader.',
        '🧍',
        'gold',
        'leaders',
        check=_self_made,
    ),
    # Cities
    Achievement(
        'black_sheep',
        'Black Sheep',
        'Build a black card. Nobody needs to know where the money came from.',
        '🐑',
        'bronze',
        'cities',
        check=lambda g, s: _color(g, s, 'black') >= 1,
    ),
    Achievement(
        'shady_dealings',
        'Shady Dealings',
        'Build four black cards in one game.',
        '🕶',
        'silver',
        'cities',
        check=lambda g, s: _color(g, s, 'black') >= 4,
    ),
    Achievement(
        'too_big_to_fail',
        'Too Big to Fail',
        'Win a game with at least three debt tokens.',
        '💸',
        'gold',
        'cities',
        check=lambda g, s: _won(g, s) and _r(g, 'debt', s) >= 3,
    ),
    Achievement(
        'packed_house',
        'Packed House',
        'Win a seven-player game with Leaders and Cities.',
        '🎪',
        'legendary',
        'opponents',
        check=lambda g, s: _won(g, s) and _uses(g, 'leaders', 'cities') and len(g['players']) >= 7,
    ),
    # opponents
    Achievement(
        'fortunes_fool',
        "Fortune's Fool",
        'Finish behind Fortuna, who plays completely at random. It happens to the best of us.',
        '🎲',
        'bronze',
        'opponents',
        check=lambda g, s: 'random' in _below(g, s),
    ),
    Achievement(
        'homeward_bound',
        'Homeward Bound',
        'Finish above Odysseus. His journey home just got a little longer.',
        '⛵',
        'silver',
        'opponents',
        check=lambda g, s: 'heuristic' in _above(g, s),
    ),
    Achievement(
        'out_forged',
        'Out-Forged',
        'Finish above Hephaestus, the best scripted builder.',
        '🔨',
        'silver',
        'opponents',
        check=lambda g, s: 'board' in _above(g, s),
    ),
    Achievement(
        'giant_slayer',
        'Giant Slayer',
        'Finish above Pythia, the trained network.',
        '🗡',
        'gold',
        'opponents',
        check=lambda g, s: 'neural' in _above(g, s),
    ),
    Achievement(
        'boss_rush',
        'Boss Rush',
        'Win a game against Pythia, Metis and Eris at the same table.',
        '👑',
        'legendary',
        'opponents',
        check=_boss_rush,
    ),
)


# -- career --------------------------------------------------------------------------------------
def _distinct(goal: int, items: Callable[[Game, int], Sequence[str]]) -> Callable[..., Progress]:
    """Progress towards ``goal`` distinct items collected over the games."""

    def career(games: Sequence[Game], ratings: Sequence[float]) -> Progress:
        seen: set = set()
        for index, game in enumerate(games):
            seen.update(items(game, int(game['human'])))
            if len(seen) >= goal:
                return goal, goal, index
        return len(seen), goal, None

    return career


def _count(goal: int) -> Callable[..., Progress]:
    def career(games: Sequence[Game], ratings: Sequence[float]) -> Progress:
        return min(len(games), goal), goal, goal - 1 if len(games) >= goal else None

    return career


def _rating(goal: int) -> Callable[..., Progress]:
    def career(games: Sequence[Game], ratings: Sequence[float]) -> Progress:
        for index, value in enumerate(ratings):
            if value >= goal:
                return goal, goal, index
        return int(max(ratings, default=0)), goal, None

    return career


def _win_wonder(g: Game, s: int) -> List[str]:
    return [g['players'][s]['wonder']] if _won(g, s) else []


CAREER: Sequence[Achievement] = (
    Achievement(
        'season_ticket',
        'Season Ticket',
        'Finish five games.',
        '🎟',
        'bronze',
        'career',
        career=_count(5),
    ),
    Achievement(
        'full_house',
        'Full House',
        'Finish a game at every table size from three to seven players.',
        '🪑',
        'silver',
        'career',
        career=_distinct(5, lambda g, s: [str(len(g['players']))]),
    ),
    Achievement(
        'globetrotter',
        'Globetrotter',
        'Win with four different wonders.',
        '🧭',
        'silver',
        'career',
        career=_distinct(4, _win_wonder),
    ),
    Achievement(
        'rogues_gallery',
        "Rogues' Gallery",
        'Finish above eight different opponents (over any number of games).',
        '🖼',
        'silver',
        'career',
        career=_distinct(8, _above),
    ),
    Achievement(
        'rising_star',
        'Rising Star',
        'Reach an Elo rating of 1750, past Odysseus.',
        '⭐',
        'gold',
        'career',
        career=_rating(1750),
    ),
    Achievement(
        'olympian',
        'Olympian',
        'Reach an Elo rating of 2000.',
        '🏆',
        'legendary',
        'career',
        career=_rating(2000),
    ),
)

# Grouped, then from bronze to legendary (the order every list shows them in).
ACHIEVEMENTS: Sequence[Achievement] = tuple(
    sorted((*FEATS, *CAREER), key=lambda a: (GROUPS.index(a.group), TIERS.index(a.tier)))
)


def earned(game: Game) -> List[str]:
    """Ids of the feats the human earned in a finished saved ``game`` (career goals are not per game)."""
    if not game.get('finished') or not game.get('result'):
        return []
    seat = int(game['human'])
    return [a.id for a in ACHIEVEMENTS if a.check is not None and a.check(game, seat)]


def achievements(games: Sequence[Game], ratings: Sequence[float] = ()) -> List[Dict[str, Any]]:
    """Every achievement with its progress.

    ``games`` are the finished games in the order they were played, ``ratings`` the human's rating
    after each of them.
    """
    games = [g for g in games if g.get('finished') and g.get('result')]
    by_id: Dict[str, List[Game]] = {a.id: [] for a in FEATS}
    for game in games:
        for a_id in earned(game):
            by_id[a_id].append(game)
    out = []
    for a in ACHIEVEMENTS:
        item: Dict[str, Any] = {
            'id': a.id,
            'name': a.name,
            'description': a.description,
            'icon': a.icon,
            'tier': a.tier,
            'points': POINTS[a.tier],
            'group': a.group,
        }
        if a.career is not None:
            progress, goal, index = a.career(games, ratings)
            done = [games[index]] if index is not None else []
            item.update(progress=progress, goal=goal)
        else:
            done = by_id[a.id]
        item.update(
            count=len(done),
            first=(done[0].get('finished_at') or done[0].get('updated')) if done else None,
            games=[g['id'] for g in done],
        )
        out.append(item)
    return out


__all__ = ['ACHIEVEMENTS', 'Achievement', 'CAREER', 'FEATS', 'GROUPS', 'POINTS', 'TIERS', 'achievements', 'earned']
