"""A representative slice of base-game content for demos and tests.

This is NOT the full 7 Wonders deck — transcribing every card and both wonder
sides is data-entry left for later. It covers every *mechanic* (production,
choice-production, trade discounts, chaining, military, science, civilian VP,
commercial coin/VP-per-card, guild scoring) with authentic card names and costs so
the engine can be exercised end to end.
"""

from __future__ import annotations

from typing import Dict, List

from wonders.cards.base import Card, Color, Cost
from wonders.cards.base import Ressource as R
from wonders.cards.base import ScienceSymbol as S
from wonders.cards.wonder import Wonder, WonderStage
from wonders.effects.library import Coins, Military, Production, Science, ScienceWild, Tally, Trade, VictoryPoints

# --- Wonders (A sides) ---------------------------------------------------


def wonder_rhodes_a() -> Wonder:
    return Wonder(
        name='The Colossus of Rhodes',
        starting_resource=R.ORE,
        side='A',
        stages=(
            WonderStage(Cost.of(R.WOOD, R.WOOD), (VictoryPoints(3),)),
            WonderStage(Cost.of(R.BRICKS, R.BRICKS, R.BRICKS), (Military(2),)),
            WonderStage(Cost.of(R.ORE, R.ORE, R.ORE, R.ORE), (VictoryPoints(7),)),
        ),
    )


def wonder_gizah_a() -> Wonder:
    return Wonder(
        name='The Pyramids of Giza',
        starting_resource=R.STONE,
        side='A',
        stages=(
            WonderStage(Cost.of(R.STONE, R.STONE), (VictoryPoints(3),)),
            WonderStage(Cost.of(R.WOOD, R.WOOD, R.WOOD), (VictoryPoints(5),)),
            WonderStage(Cost.of(R.STONE, R.STONE, R.STONE, R.STONE), (VictoryPoints(7),)),
        ),
    )


def wonder_ephesos_a() -> Wonder:
    return Wonder(
        name='The Temple of Artemis in Ephesus',
        starting_resource=R.PAPER,
        side='A',
        stages=(
            WonderStage(Cost.of(R.STONE, R.STONE), (VictoryPoints(3),)),
            WonderStage(Cost.of(R.WOOD, R.WOOD), (Coins(9),)),
            WonderStage(Cost.of(R.PAPER, R.PAPER), (VictoryPoints(7),)),
        ),
    )


# --- Cards ---------------------------------------------------------------
# Grouped by Age; costs and chains follow the base game. Only a subset per Age.


def _age_i() -> List[Card]:
    return [
        # Brown — raw materials
        Card('Lumber Yard', Color.BROWN, effects=(Production(R.WOOD),), age=1),
        Card('Stone Pit', Color.BROWN, effects=(Production(R.STONE),), age=1),
        Card('Clay Pool', Color.BROWN, effects=(Production(R.BRICKS),), age=1),
        Card('Ore Vein', Color.BROWN, effects=(Production(R.ORE),), age=1),
        Card('Tree Farm', Color.BROWN, Cost(coins=1), effects=(Production(choice=(R.WOOD, R.BRICKS)),), age=1),
        # Grey — manufactured goods
        Card('Loom', Color.GREY, effects=(Production(R.LINEN),), age=1),
        Card('Glassworks', Color.GREY, effects=(Production(R.GLASS),), age=1),
        Card('Press', Color.GREY, effects=(Production(R.PAPER),), age=1),
        # Blue — civilian
        Card('Altar', Color.BLUE, effects=(VictoryPoints(2),), chain_enables=('Temple',), age=1),
        Card('Baths', Color.BLUE, Cost.of(R.STONE), (VictoryPoints(3),), chain_enables=('Aqueduct',), age=1),
        # Yellow — commercial
        Card(
            'East Trading Post',
            Color.YELLOW,
            effects=(Trade(side='right', category='raw'),),
            chain_enables=('Forum',),
            age=1,
        ),
        Card(
            'West Trading Post',
            Color.YELLOW,
            effects=(Trade(side='left', category='raw'),),
            chain_enables=('Forum',),
            age=1,
        ),
        Card(
            'Marketplace',
            Color.YELLOW,
            effects=(Trade(side='both', category='manufactured'),),
            chain_enables=('Caravansery',),
            age=1,
        ),
        Card('Tavern', Color.YELLOW, effects=(Coins(5),), age=1),
        # Red — military
        Card('Stockade', Color.RED, Cost.of(R.WOOD), (Military(1),), age=1),
        Card('Barracks', Color.RED, Cost.of(R.ORE), (Military(1),), age=1),
        Card('Guard Tower', Color.RED, Cost.of(R.BRICKS), (Military(1),), age=1),
        # Green — science
        Card('Apothecary', Color.GREEN, Cost.of(R.LINEN), (Science(S.COMPASS),), chain_enables=('Dispensary',), age=1),
        Card('Workshop', Color.GREEN, Cost.of(R.GLASS), (Science(S.GEAR),), chain_enables=('Laboratory',), age=1),
        Card(
            'Scriptorium',
            Color.GREEN,
            Cost.of(R.PAPER),
            (Science(S.TABLET),),
            chain_enables=('Library', 'Courthouse'),
            age=1,
        ),
    ]


def _age_ii() -> List[Card]:
    return [
        Card('Sawmill', Color.BROWN, Cost(coins=1), effects=(Production(R.WOOD), Production(R.WOOD)), age=2),
        Card(
            'Aqueduct',
            Color.BLUE,
            Cost.of(R.STONE, R.STONE, R.STONE),
            (VictoryPoints(5),),
            chain_requires='Baths',
            age=2,
        ),
        Card(
            'Temple',
            Color.BLUE,
            Cost.of(R.WOOD, R.BRICKS, R.GLASS),
            (VictoryPoints(3),),
            chain_requires='Altar',
            chain_enables=('Pantheon',),
            age=2,
        ),
        Card(
            'Forum',
            Color.YELLOW,
            Cost.of(R.BRICKS, R.BRICKS),
            effects=(Production(choice=(R.GLASS, R.LINEN, R.PAPER)),),
            chain_requires='East Trading Post',
            age=2,
        ),
        Card(
            'Walls',
            Color.RED,
            Cost.of(R.STONE, R.STONE, R.STONE),
            (Military(2),),
            chain_enables=('Fortifications',),
            age=2,
        ),
        Card(
            'Library',
            Color.GREEN,
            Cost.of(R.STONE, R.STONE, R.LINEN),
            (Science(S.TABLET),),
            chain_requires='Scriptorium',
            age=2,
        ),
        Card(
            'Dispensary',
            Color.GREEN,
            Cost.of(R.ORE, R.ORE, R.GLASS),
            (Science(S.COMPASS),),
            chain_requires='Apothecary',
            age=2,
        ),
    ]


def _age_iii() -> List[Card]:
    return [
        # Blue
        Card(
            'Pantheon',
            Color.BLUE,
            Cost.of(R.BRICKS, R.BRICKS, R.ORE, R.GLASS, R.PAPER, R.LINEN),
            (VictoryPoints(7),),
            chain_requires='Temple',
            age=3,
        ),
        Card(
            'Palace',
            Color.BLUE,
            Cost.of(R.STONE, R.ORE, R.WOOD, R.BRICKS, R.GLASS, R.PAPER, R.LINEN),
            (VictoryPoints(8),),
            age=3,
        ),
        # Green
        Card(
            'University',
            Color.GREEN,
            Cost.of(R.WOOD, R.WOOD, R.PAPER, R.GLASS),
            (Science(S.TABLET),),
            chain_requires='Library',
            age=3,
        ),
        # Yellow — end-game coin/VP per card
        Card(
            'Haven',
            Color.YELLOW,
            Cost.of(R.WOOD, R.ORE, R.LINEN),
            effects=(Tally('colors', 'self', {Color.BROWN}, coins_each=1, vp_each=1),),
            age=3,
        ),
        Card(
            'Lighthouse',
            Color.YELLOW,
            Cost.of(R.STONE, R.GLASS),
            effects=(Tally('colors', 'self', {Color.YELLOW}, coins_each=1, vp_each=1),),
            chain_requires='Caravansery',
            age=3,
        ),
        Card(
            'Arena',
            Color.YELLOW,
            Cost.of(R.STONE, R.STONE, R.ORE),
            effects=(Tally('wonder_stages', 'self', coins_each=3, vp_each=1),),
            age=3,
        ),
        # Red
        Card(
            'Fortifications',
            Color.RED,
            Cost.of(R.STONE, R.ORE, R.ORE, R.ORE),
            (Military(3),),
            chain_requires='Walls',
            age=3,
        ),
        # Purple — guilds
        Card(
            'Workers Guild',
            Color.PURPLE,
            Cost.of(R.ORE, R.ORE, R.STONE, R.BRICKS, R.WOOD),
            effects=(Tally('colors', 'neighbours', {Color.BROWN}, vp_each=1),),
            age=3,
        ),
        Card(
            'Traders Guild',
            Color.PURPLE,
            Cost.of(R.LINEN, R.PAPER, R.GLASS),
            effects=(Tally('colors', 'neighbours', {Color.YELLOW}, vp_each=1),),
            age=3,
        ),
        Card(
            'Builders Guild',
            Color.PURPLE,
            Cost.of(R.STONE, R.STONE, R.BRICKS, R.BRICKS, R.GLASS),
            effects=(Tally('wonder_stages', 'both', vp_each=1),),
            age=3,
        ),
        Card(
            'Strategists Guild',
            Color.PURPLE,
            Cost.of(R.ORE, R.ORE, R.STONE, R.LINEN),
            effects=(Tally('defeat_tokens', 'neighbours', vp_each=1),),
            age=3,
        ),
        Card(
            'Scientists Guild',
            Color.PURPLE,
            Cost.of(R.WOOD, R.WOOD, R.ORE, R.ORE, R.PAPER),
            effects=(ScienceWild(),),
            age=3,
        ),
    ]


def base_cards() -> Dict[int, List[Card]]:
    """Representative card pool keyed by Age (1, 2, 3)."""
    return {1: _age_i(), 2: _age_ii(), 3: _age_iii()}
