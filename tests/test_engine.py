"""Tests for actions, trading through the engine, and a full game."""

import pytest

from wonders.cards.base import Card, Color, Cost
from wonders.cards.base import Ressource as R
from wonders.cards.library import base_cards, wonder_ephesos_a, wonder_gizah_a, wonder_rhodes_a
from wonders.effects.library import Production, VictoryPoints
from wonders.engine.actions import Action, ActionKind
from wonders.engine.game import Game, IllegalAction
from wonders.engine.phase import Phase
from wonders.engine.player import Player


def two_players():
    return Game([Player('P0', wonder_gizah_a()), Player('P1', wonder_rhodes_a())])


def test_build_free_card():
    g = two_players()
    g.apply(0, Action.build(Card('Altar', Color.BLUE, effects=(VictoryPoints(2),))))
    assert g.players[0].has('Altar')
    assert g.score(0).civilian == 2


def test_build_pays_bank_coins():
    g = two_players()
    g.players[0].coins = 5
    g.apply(0, Action.build(Card('Bribe', Color.YELLOW, Cost(coins=3))))
    assert g.players[0].coins == 2


def test_trade_moves_coins_to_neighbour():
    g = two_players()
    # P1 produces wood; P0 (start Stone) buys 1 wood at default price 2.
    g.players[1].add_card(Card('Lumber Yard', Color.BROWN, effects=(Production(R.WOOD),)))
    g.players[0].coins = 3
    before = g.players[1].coins
    g.apply(0, Action.build(Card('Shed', Color.BROWN, Cost.of(R.WOOD))))
    assert g.players[0].coins == 1  # paid 2
    assert g.players[1].coins == before + 2  # neighbour received it


def test_unaffordable_build_is_illegal():
    g = two_players()
    g.players[0].coins = 1
    with pytest.raises(IllegalAction):
        g.apply(0, Action.build(Card('Costly', Color.BLUE, Cost.of(R.GLASS, R.GLASS))))


def test_chain_build_is_free_even_without_resources():
    g = two_players()
    g.players[0].coins = 0
    g.apply(0, Action.build(Card('Baths', Color.BLUE, Cost.of(R.STONE), chain_enables=('Aqueduct',))))
    # Aqueduct normally costs 3 stone; with the Baths chain it is free.
    aqueduct = Card(
        'Aqueduct', Color.BLUE, Cost.of(R.STONE, R.STONE, R.STONE), (VictoryPoints(5),), chain_requires='Baths'
    )
    assert g.can_build_card(0, aqueduct)
    g.apply(0, Action.build(aqueduct))
    assert g.players[0].coins == 0 and g.score(0).civilian == 5


def test_chain_free_from_either_prerequisite():
    # Forum is unlocked by East OR West Trading Post; owning either builds it free.
    from wonders.cards.content import load_content

    content = load_content('base')
    for post in ('East Trading Post', 'West Trading Post'):
        g = two_players()
        g.players[0].coins = 0  # cannot pay Forum's 2-clay cost
        g.apply(0, Action.build(content.card(post)))
        assert g.can_build_card(0, content.card('Forum'))


def test_duplicate_structure_is_illegal():
    g = two_players()
    altar = Card('Altar', Color.BLUE, effects=(VictoryPoints(2),))
    g.apply(0, Action.build(altar))
    with pytest.raises(IllegalAction):
        g.apply(0, Action.build(Card('Altar', Color.BLUE, effects=(VictoryPoints(2),))))


def test_wonder_stages_build_in_order_and_score():
    g = Game([Player('P0', wonder_ephesos_a()), Player('P1', wonder_rhodes_a())])
    p = g.players[0]
    p.coins = 10
    # Ephesos starts with Paper; stage 1 needs 2 Stone -> buy from neighbour.
    g.players[1].add_card(Card('Quarry', Color.BROWN, effects=(Production(R.STONE),)))
    g.players[1].add_card(Card('Quarry2', Color.BROWN, effects=(Production(R.STONE),)))
    g.apply(0, Action.wonder(Card('junk', Color.GREY)))
    assert p.wonder_stages_built == 1
    assert g.score(0).wonders == 3  # stage 1 gives 3 VP


def test_discard_gives_three_coins():
    g = two_players()
    g.players[0].coins = 0
    g.apply(0, Action.discard(Card('junk', Color.GREY)))
    assert g.players[0].coins == 3
    assert len(g.discard_pile) == 1


def test_legal_actions_always_offers_discard():
    g = two_players()
    g.players[0].coins = 0
    hand = [Card('Costly', Color.BLUE, Cost.of(R.GLASS, R.GLASS))]
    actions = g.legal_actions(0, hand)
    kinds = {a.kind for a in actions}
    assert ActionKind.DISCARD in kinds
    assert ActionKind.BUILD not in kinds  # unaffordable, no producers


def test_observe_has_neighbour_views():
    g = two_players()
    obs = g.observe(0)
    assert set(obs) >= {'age', 'phase', 'self', 'left', 'right', 'score'}
    assert 'coins' in obs['self']


def _greedy(game, i, hand):
    actions = game.legal_actions(i, hand)
    for kind in (ActionKind.BUILD, ActionKind.WONDER, ActionKind.DISCARD):
        for a in actions:
            if a.kind is kind:
                return a
    return actions[0]


def test_full_three_age_game_runs_and_scores():
    wonders = [wonder_gizah_a(), wonder_rhodes_a(), wonder_ephesos_a()]
    g = Game([Player(f'P{i}', w) for i, w in enumerate(wonders)])
    pool = base_cards()

    def deal(age):
        cards = list(pool[age])
        while len(cards) < g.n * 7:  # pad to full 7-card hands
            cards += list(pool[age])
        return [cards[k * 7 : (k + 1) * 7] for k in range(g.n)]

    g.play(deal, _greedy)

    assert g.phase is Phase.FINISHED
    for p in g.players:
        assert p.coins >= 0  # never overspend
        # up to 2 conflict tokens per age (fewer on ties), across 3 ages
        assert 0 <= len(p.tokens) <= 6
    totals = [g.score(i).total for i in range(g.n)]
    assert all(isinstance(t, int) for t in totals)
    assert 0 <= g.winner() < g.n


def test_neighbours_sell_only_board_and_brown_grey_resources():
    from collections import Counter

    from wonders.cards.base import Cost, Ressource
    from wonders.cards.content import load_content
    from wonders.engine.game import Game
    from wonders.engine.player import Player

    content = load_content('base')
    players = [
        Player(f'P{i}', content.wonder(w, 'A'))
        for i, w in enumerate(
            ['The Colossus of Rhodes', 'The Statue of Zeus in Olympia', 'The Temple of Artemis in Ephesus']
        )
    ]
    game = Game(players)
    glass = Cost(0, Counter({Ressource.GLASS: 1}))
    players[1].add_card(content.card('Forum'))  # yellow: glass / loom / papyrus, for its owner only
    assert players[1].production_slots() != players[1].tradeable_slots()
    assert game.payment_for(0, glass) is None
    players[2].add_card(content.card('Glassworks'))  # grey: may be bought
    pay = game.payment_for(0, glass)
    assert pay is not None and pay.right == 2 and pay.left == 0
