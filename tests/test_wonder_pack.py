"""Tests for the Wonder Pack: the Great Wall, Abu Simbel, Stonehenge and the Manneken Pis."""

import json
import random
from typing import Any, Callable, List, Sequence, Tuple

import pytest

from wonders.agents.worlds import _public_game, content_index
from wonders.cards.base import Card, Color, Cost
from wonders.cards.content import load_content
from wonders.effects.library import TRADEABLE, MultiProduction
from wonders.engine.actions import Action, DecisionKind
from wonders.engine.game import CARD, Game
from wonders.engine.observation import build_observation
from wonders.engine.phase import Phase
from wonders.engine.player import Player
from wonders.engine.record import GameRecord, PlayerSetup, play_and_record
from wonders.play.session import GameConfig, HumanSession, SeatConfig

ALL = ('base', 'leaders', 'cities', 'wonderpack')
CONTENT = load_content(*ALL)


def game(boards: Sequence[Tuple[str, str]], expansions: Sequence[str] = ALL, coins: int = 30) -> Game:
    """A game whose seat 0 can pay any stage itself (a yellow card producing every resource, 7 times)."""
    players = [Player.starting(f'P{i}', CONTENT.wonder(name, side), coins) for i, (name, side) in enumerate(boards)]
    players[0].add_card(Card('Supply', Color.YELLOW, effects=(MultiProduction(*[TRADEABLE] * 7),)))
    return Game(players, expansions=tuple(expansions))


def powers(g: Game, choose: Callable[[Any], Card]) -> List[Tuple[str, int]]:
    """Run the owed power decisions, answering each with ``choose(point)``; returns (kind, seat) asked."""
    asked = []
    steps = g._power_steps()
    try:
        point = next(steps)
        while True:
            (seat,) = point.options
            asked.append((point.kind.value, seat))
            card = choose(point)
            action = Action.free(card) if point.kind is DecisionKind.DISCARD_BUILD else Action.build(card)
            point = steps.send({seat: action})
    except StopIteration:
        pass
    return asked


def junk(name: str = 'junk', color: Color = Color.GREY) -> Card:
    return Card(name, color, Cost())


GIZA, RHODES, EPHESOS = (
    ('The Pyramids of Giza', 'A'),
    ('The Colossus of Rhodes', 'A'),
    ('The Temple of Artemis in Ephesus', 'A'),
)


# -- content ------------------------------------------------------------------------------------
def test_the_pack_has_four_boards_and_abu_simbel_needs_leaders():
    pack = {(w.name, w.side) for w in load_content('base', 'leaders', 'wonderpack').wonders}
    for name in ('The Great Wall', 'Abu Simbel', 'Stonehenge', 'Manneken Pis'):
        assert {(name, 'A'), (name, 'B')} <= pack
    assert not any(w.name == 'Abu Simbel' for w in load_content('base', 'wonderpack').wonders)
    great_wall_a = CONTENT.wonder('The Great Wall', 'A')
    assert great_wall_a.any_order and len(great_wall_a.stages) == 4
    assert len(CONTENT.wonder('Manneken Pis', 'B').stages[0].cost.resources) == 7


def test_the_manneken_pis_starts_with_four_extra_coins_and_no_resource():
    g = game([('Manneken Pis', 'B'), GIZA, RHODES], coins=3)
    assert g.players[0].coins == 7 and g.players[1].coins == 3
    assert g.players[0].wonder.starting_resource is None


# -- the Great Wall -----------------------------------------------------------------------------
def test_the_great_wall_builds_its_stages_in_any_order():
    g = game([('The Great Wall', 'A'), GIZA, RHODES])
    wall = g.players[0]
    assert [slot for slot, _ in g.stage_options(0)] == [0, 1, 2, 3]
    g.apply(0, Action.wonder(junk()).staged(2))  # stage 3 first: 2 shields
    assert wall.built_slots == [2] and wall.shields() == 2
    assert [slot for slot, _ in g.stage_options(0)] == [0, 1, 3]
    g.apply(0, Action.wonder(junk('junk2')))  # no choice given: the first affordable one
    assert wall.built_slots == [2, 0] and wall.coins == 30 + 8


def test_a_chosen_stage_is_recorded_and_replayed():
    rng = random.Random(4)
    setups = [PlayerSetup('a', 'The Great Wall', 'A'), PlayerSetup('b', GIZA[0], 'A'), PlayerSetup('c', RHODES[0], 'A')]

    def last_stage(g: Game, seat: int, hand: List[Card]) -> Action:
        actions = g.legal_actions(seat, hand)
        wonder = [a for a in actions if a.kind.value == 'wonder']
        if seat == 0 and wonder:
            options = g.stage_options(0)
            affordable = [k for k in range(len(options)) if g._stage_payment(0, k) is not None]
            return wonder[0].staged(affordable[-1])
        return actions[0]

    content = load_content('base', 'wonderpack')
    game_, record = play_and_record(content, setups, last_stage, rng, ('base', 'wonderpack'))
    assert game_.players[0].built_slots and game_.players[0].built_slots != sorted(game_.players[0].built_slots)
    moves = [m for age in record.moves.values() for turn in age for m in turn]
    assert any(len(m) == 5 for m in moves)  # [code, card, null, null, stage]
    replayed = GameRecord.from_json(record.to_json()).replay(content)
    assert replayed.players[0].built_slots == game_.players[0].built_slots
    assert [replayed.score(i).total for i in range(3)] == [game_.score(i).total for i in range(3)]


def test_great_wall_b_plunder_makes_debt_only_with_cities():
    for expansions, debt in ((('base', 'wonderpack'), 0), (('base', 'cities', 'wonderpack'), 1)):
        g = game([('The Great Wall', 'B'), GIZA, RHODES], expansions)
        g.players[1].coins = 1
        g.apply(0, Action.wonder(junk()).staged(2))  # diplomacy + every other player pays 2
        assert g.players[0].diplomacy == 1
        assert g.players[1].coins == 0 and g.players[1].debt == debt


def test_great_wall_b_gives_its_neighbours_two_coins():
    g = game([('The Great Wall', 'B'), GIZA, RHODES, EPHESOS])
    g.apply(0, Action.wonder(junk()).staged(0))
    assert [p.coins for p in g.players] == [38, 32, 30, 32]


# -- the Manneken Pis ---------------------------------------------------------------------------
def test_the_manneken_pis_copies_its_neighbours_stages():
    g = game([('Manneken Pis', 'A'), GIZA, RHODES])  # left: Giza, right: Rhodes
    left, right = g.players[1].wonder, g.players[2].wonder
    slots = g.players[0].slots
    assert slots[0] == (left.stages[0],) and slots[1] == (right.stages[1],) and slots[2] == (left.stages[-1],)
    g.apply(0, Action.wonder(junk()))
    assert g.players[0].stages_built[0] is left.stages[0]
    assert g.score(0).wonders == g.score(1).wonders + 3  # Giza A stage 1: 3 VP


def test_the_manneken_pis_chooses_which_great_wall_stage_to_copy():
    g = game([('Manneken Pis', 'A'), ('The Great Wall', 'A'), RHODES])
    assert len(g.players[0].slots[0]) == 4 and len(g.players[0].slots[2]) == 4
    assert len(g.stage_options(0)) == 4  # the first board stage, as any Great Wall stage
    g.apply(0, Action.wonder(junk()).staged(2))  # copy the 2-shield stage
    assert g.players[0].shields() == 2 and g.players[0].built_variants == [2]


def test_copied_discard_builds_resolve_in_the_faq_order():
    # Halicarnassus (own stage) before the Great Wall, before a Manneken copy, before Solomon (a card).
    hali = CONTENT.wonder('The Mausoleum of Halicarnassus', 'A')
    stage = next(s for s in hali.stages if any(e.discard_builds() for e in s.effects))
    g = game([('Manneken Pis', 'A'), ('The Great Wall', 'A'), ('The Mausoleum of Halicarnassus', 'A'), EPHESOS])
    g.discard_pile = [junk(f'pile{k}') for k in range(4)]
    solomon = CONTENT.card('Solomon')
    g._activate(3, solomon.effects, CARD)
    g._activate(0, g.players[1].wonder.stages[3].effects, 2)  # as the Manneken Pis copying the wall's 4th
    g._activate(1, g.players[1].wonder.stages[3].effects)
    g._activate(2, stage.effects)
    asked = powers(g, lambda point: next(iter(point.options.values()))[0])
    assert asked == [('discard_build', 2), ('discard_build', 1), ('discard_build', 0), ('discard_build', 3)]


# -- Abu Simbel ---------------------------------------------------------------------------------
def test_abu_simbel_entombs_a_leader_for_twice_its_cost():
    g = game([('Abu Simbel', 'B'), GIZA, RHODES])
    archimedes = CONTENT.card('Archimedes')
    g.players[0].add_card(archimedes)
    assert g.players[0].resource_discount('green') == 1
    g.apply(0, Action.wonder(junk()))
    assert powers(g, lambda point: point.options[0][0]) == [('entomb_leader', 0)]
    p = g.players[0]
    assert p.entombed == [archimedes] and archimedes not in p.cards
    assert p.resource_discount('green') == 0  # its power is lost
    assert g.score(0).wonders == 8  # 4 coins x 2
    view = build_observation(g, 1, [], [])
    assert view.right.entombed == ['Archimedes']  # public


def test_abu_simbel_without_leaders_fizzles():
    g = game([('Abu Simbel', 'A'), GIZA, RHODES])
    g.apply(0, Action.wonder(junk()))
    g.apply(0, Action.wonder(junk('j2')))
    g.apply(0, Action.wonder(junk('j3')))
    assert powers(g, lambda point: point.options[0][0]) == [] and g.players[0].entombed == []


# -- Stonehenge ---------------------------------------------------------------------------------
def test_stonehenge_counts_stone_symbols_on_brown_cards():
    g = game([('Stonehenge', 'A'), GIZA, RHODES])
    for name in ('Quarry', 'Excavation', 'Lumber Yard', 'Caravansery'):  # 2 + 1 + 0, Caravansery is yellow
        g.players[0].add_card(CONTENT.card(name))
    for k in range(3):
        g.apply(0, Action.wonder(junk(f'j{k}')))
    assert g.score(0).wonders == 3 + 5 + 2 * 3


def test_stonehenge_b_pays_coins_now_and_points_later():
    g = game([('Stonehenge', 'B'), GIZA, RHODES])
    g.players[0].add_card(CONTENT.card('Quarry'))
    before = g.players[0].coins
    g.apply(0, Action.wonder(junk()))
    assert g.players[0].coins == before + 2  # 1 coin per stone symbol
    assert g.score(0).wonders == 2


def test_stonehenge_b_scores_the_buried_cards_colour_only_at_the_end():
    g = game([('Stonehenge', 'B'), GIZA, RHODES])
    for k, seat in enumerate((1, 1, 2)):
        g.players[seat].add_card(Card(f'blue{k}', Color.BLUE))
    g.apply(0, Action.wonder(junk('first')))
    g.apply(0, Action.wonder(junk('altar', Color.BLUE)))
    during = g.score(0).wonders
    view = build_observation(g, 1, [], [])
    assert view.right.score_breakdown['wonders'] == during  # nothing reveals the buried colour
    g.phase = Phase.FINISHED
    assert g.score(0).wonders == during + 3
    assert g.players[0].stage_cards[-1].name == 'altar'


# -- worlds, GUI --------------------------------------------------------------------------------
def test_sampled_worlds_rebuild_out_of_order_and_copied_stages():
    g = game([('The Great Wall', 'A'), ('Manneken Pis', 'A'), ('Abu Simbel', 'A'), RHODES])
    g.players[1].add_card(Card('Supply', Color.YELLOW, effects=(MultiProduction(*[TRADEABLE] * 7),)))
    g.age, g.phase = 1, Phase.AGE_I
    g.apply(0, Action.wonder(junk()).staged(3))
    g.apply(1, Action.wonder(junk('j1')))  # copies Abu Simbel's 1st stage (its left neighbour)
    g.apply(1, Action.wonder(junk('j2')).staged(1))  # then the wall (its right neighbour): its 2nd stage
    g.players[2].add_card(CONTENT.card('Nefertiti'))
    g.players[2].entomb(CONTENT.card('Nefertiti'))
    for p in g.players:  # the test-only supply card is not in the content a world is rebuilt from
        p.cards = [c for c in p.cards if c.name != 'Supply']
        p._cache.clear()
    obs = build_observation(g, 3, [], [])
    world = _public_game(obs, content_index(ALL))
    assert world.players[0].built_slots == [3] and world.players[1].built_variants == [0, 1]
    assert [c.name for c in world.players[2].entombed] == ['Nefertiti']
    assert [world.score(i).total for i in range(4)] == [g.score(i).total for i in range(4)]


def test_the_human_chooses_a_great_wall_stage_in_the_gui():
    seats = [SeatConfig(), SeatConfig('greedy'), SeatConfig('heuristic')]
    session = HumanSession(
        GameConfig(expansions=['base', 'wonderpack'], seats=seats, wonder='The Great Wall', side='A', seed=5)
    )
    chosen = None
    while not session.done:
        actions = session.state()['observation']['actions']
        staged = [a for a in actions if a.get('stages')]
        if staged and chosen is None:
            chosen = staged[0]['stages'][-1]
            session.act(staged[0]['index'], stage=chosen['index'])
            assert chosen['stage'] - 1 in session.env.game.players[session.human].built_slots
        else:
            session.act(0)
    assert chosen is not None
    saved = json.loads(json.dumps(session.saved()))
    restored = HumanSession.restore(saved)
    assert restored.env.game.players[0].built_slots == session.env.game.players[0].built_slots
    with pytest.raises(Exception):
        session.act(0, stage=0)  # the game is over
