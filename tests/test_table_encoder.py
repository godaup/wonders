"""Tests for encoder v2 (TableEncoder) and the observation fields it relies on."""

import json
import random

import pytest

from wonders.agents import HeuristicAgent
from wonders.agents.match import play_match
from wonders.agents.table_encoder import MAX_SEATS, TableEncoder
from wonders.cards.content import load_content
from wonders.engine.game import Game
from wonders.engine.observation import ObservationBuilder, build_observation
from wonders.engine.player import Player
from wonders.engine.record import PlayerSetup
from wonders.env import WondersEnv
from wonders.env.vocab import Vocab, universal_vocab

CONTENT = load_content('base', 'leaders', 'cities')
VOCAB = universal_vocab('table-v2')
WONDERS = list(dict.fromkeys(w.name for w in CONTENT.wonders))


def table(n=4, expansions=('base', 'leaders', 'cities')):
    players = [Player(f'P{i}', CONTENT.wonder(WONDERS[i], 'A')) for i in range(n)]
    g = Game(players, expansions=expansions)
    g.age = 1
    return g


def test_layout_is_contiguous_and_player_count_independent():
    enc = VOCAB.encoder()
    assert isinstance(enc, TableEncoder)
    sections = enc.sections()
    assert sections[0][1] == 0 and sum(n for _, _, n in sections) == enc.size
    for (_, off, n), (_, nxt, _) in zip(sections, sections[1:]):
        assert off + n == nxt
    assert [name for name, _, _ in sections[:MAX_SEATS]] == [f'city{k}' for k in range(MAX_SEATS)]
    for n in (3, 5, 7):
        g = table(n)
        assert len(enc.encode(build_observation(g, 0, [CONTENT.card('Altar')]))) == enc.size


def test_seat_relative_ordering_and_padding():
    enc = VOCAB.encoder()
    g = table(4)
    for i, p in enumerate(g.players):
        p.coins = 10 * (i + 1)  # coins / 10 = 1, 2, 3, 4
    vec = enc.encode(build_observation(g, 2, []))
    coins = [vec[off + 3] for name, off, _ in enc.sections()[:MAX_SEATS]]
    # viewer seat 2, then clockwise: seat 3 (left neighbour), seat 0, seat 1 (right neighbour)
    assert coins[:4] == [3.0, 4.0, 1.0, 2.0]
    assert coins[4:] == [0.0, 0.0, 0.0]  # empty seats
    city1 = enc.sections()[1][1]
    city3 = enc.sections()[3][1]
    assert vec[city1 + 1] == 1.0 and vec[city3 + 2] == 1.0  # left / right neighbour flags


def test_hidden_information_never_enters():
    enc = VOCAB.encoder()
    g = table(3)
    base = enc.encode(build_observation(g, 0, [CONTENT.card('Altar')]))
    # Opponents' leader hands and hands are private: changing them must not change the encoding.
    g.players[1].leaders_hand = [CONTENT.card('Plato'), CONTENT.card('Nero')]
    g.players[2].leaders_hand = [CONTENT.card('Cleopatra')]
    assert enc.encode(build_observation(g, 0, [CONTENT.card('Altar')])) == base
    # The viewer's own leader hand does enter.
    g.players[0].leaders_hand = [CONTENT.card('Midas')]
    assert enc.encode(build_observation(g, 0, [CONTENT.card('Altar')])) != base


def test_observation_carries_the_new_public_and_private_fields():
    g = table(3)
    g.players[1].tokens = [3, -1]
    g.players[1].debt = 2
    g.players[0].leaders_hand = [CONTENT.card('Midas')]
    obs = build_observation(g, 0, [CONTENT.card('Altar')])
    assert obs.table[1].tokens == [3, -1] and obs.table[1].debt == 2
    assert obs.table[1].next_stage_cost is not None and obs.table[1].score_breakdown['total'] == obs.table[1].score
    assert obs.leaders_hand == ['Midas']
    assert set(obs.buy_prices) == {'left', 'right'} and obs.buy_prices['left']['wood'] == 2
    assert obs.expansions == ['base', 'leaders', 'cities']
    assert obs.hand[0].can_build and obs.hand[0].trade_coins == 0


def test_seen_cards_remember_earlier_hands():
    content = load_content('base')
    builder = ObservationBuilder()
    g = Game([Player(f'P{i}', content.wonder(WONDERS[i], 'A')) for i in range(3)])
    g.age, g.turn = 1, 0
    first = [content.card(n) for n in ('Altar', 'Baths', 'Lumber Yard')]
    builder.build(g, 0, first)
    builder.record(1, 0, [(_kind('build'), 'Altar'), None, None])
    g.turn = 1
    second = [content.card(n) for n in ('Tavern', 'Stockade')]
    obs = builder.build(g, 0, second)
    assert sorted(obs.seen_cards) == ['Baths', 'Lumber Yard']  # passed on, not played, not in hand


def _kind(value):
    from wonders.engine.actions import ActionKind

    return ActionKind(value)


@pytest.mark.parametrize('expansions', [('base',), ('base', 'leaders', 'cities')])
def test_env_trains_with_the_v2_encoder(expansions):
    env = WondersEnv(
        n_players=5,
        expansions=expansions,
        card_names=VOCAB.card_names,
        decisions=VOCAB.decisions,
        encoder=VOCAB.encoder(),
    )
    rng = random.Random(0)
    observations, infos = env.reset(1)
    while observations:
        assert all(len(o['observation']) == VOCAB.obs_size for o in observations.values())
        observations, *_rest, infos = env.step(
            {s: rng.choice([i for i, ok in enumerate(o['action_mask']) if ok]) for s, o in observations.items()}
        )


def test_vocab_round_trips_through_checkpoint_meta():
    meta = VOCAB.meta()
    assert Vocab.from_checkpoint(VOCAB.card_names, json.loads(json.dumps(meta))) == VOCAB
    assert Vocab.from_checkpoint(VOCAB.card_names, {}).encoder_kind == 'card-v1'  # older checkpoints
    with pytest.raises(ValueError):
        VOCAB.with_encoder('nope').encoder()


def test_full_game_with_scripted_agents_encodes_every_decision():
    # Every observation the match produces must encode without errors in v2.
    enc = VOCAB.encoder()
    encoded = []

    class Probe(HeuristicAgent):
        def act(self, obs, options):
            encoded.append(len(enc.encode(obs)))
            return super().act(obs, options)

    setups = [PlayerSetup(f'P{i}', w, 'B') for i, w in enumerate(WONDERS[:5])]
    play_match(
        CONTENT, setups, [Probe() for _ in range(5)], rng=random.Random(2), expansions=('base', 'leaders', 'cities')
    )
    assert encoded and set(encoded) == {enc.size}
