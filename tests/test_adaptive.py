"""Tests for the adaptive scripted agents."""

from dataclasses import replace

import pytest

from wonders.agents import AdaptiveAgent, RivalAgent
from wonders.agents.bench import run_bench
from wonders.cards.content import load_content
from wonders.engine.observation import _static_summary
from wonders.env import WondersEnv


@pytest.fixture
def obs():
    env = WondersEnv(n_players=4)
    env.reset(3)
    return env.observation(0)


def _with_shields(obs, me, left, right, age=1):
    return replace(
        obs,
        age=age,
        me=replace(obs.me, shields=me),
        left=replace(obs.left, shields=left),
        right=replace(obs.right, shields=right),
    )


def test_military_value_follows_contestable_fronts(obs):
    agent = AdaptiveAgent()
    contested = agent._front_gain(_with_shields(obs, me=2, left=2, right=2), 1)
    hopeless = agent._front_gain(_with_shields(obs, me=0, left=9, right=9), 1)
    safe = agent._front_gain(_with_shields(obs, me=9, left=0, right=0), 1)
    assert contested > 1.0
    assert 0 <= hopeless < 0.2 * contested and 0 <= safe < 0.2 * contested


def test_phase_decays_over_the_game(obs):
    agent = AdaptiveAgent()
    start = agent._phase(obs)
    last_age = agent._phase(replace(obs, age=3, hand=obs.hand[:2]))
    assert start == pytest.approx(1.0)
    assert agent.w['production_floor'] <= last_age < 0.3


def test_wheel_discount_only_while_the_hand_can_come_back(obs):
    agent = AdaptiveAgent()
    card = obs.hand[0]
    assert len(obs.hand) > obs.n_players  # 7 cards, 4 players: the hand wheels
    assert 0.0 <= agent._wheel_discount(obs, card) <= agent.w['wheel']
    assert agent._wheel_discount(replace(obs, hand=obs.hand[:4]), card) == 0.0
    assert agent._wheel_discount(replace(obs, decision='last_card'), card) == 0.0


def test_rival_targets_the_score_leader(obs):
    table = [replace(c, score=s) for c, s in zip(obs.table, (10, 25, 12, 30))]
    leader = RivalAgent._rival(replace(obs, table=table))
    assert leader.seat == 3


@pytest.mark.parametrize('expansions', [('base',), ('base', 'leaders'), ('base', 'leaders', 'cities')])
def test_adaptive_agents_play_full_games(expansions):
    result = run_bench(['adaptive', 'rival', 'board', 'heuristic', 'greedy'], n_games=3, seed=2, expansions=expansions)
    assert result.stats['adaptive'].games == 3 and result.stats['rival'].games == 3


# -- combos ----------------------------------------------------------------------
def _summaries(name):
    card = next(c for c in load_content('base').cards if c.name == name)
    return [_static_summary(e) for e in card.effects]


def _view(obs, name, color, effects=()):
    return replace(obs.hand[0], name=name, color=color, effects=list(effects), chain_enables=(), chain_free=False)


def test_counts_hook_describes_counting_effects():
    counts = [e.get('counts') for e in _summaries('Haven')]
    assert {'what': 'colors', 'scope': 'self', 'colors': ['brown'], 'vp_each': 1, 'coins_each': 1} in counts


def test_owned_payoffs_make_counted_cards_worth_more(obs):
    agent = AdaptiveAgent()
    brown, blue = _view(obs, 'X', 'brown'), _view(obs, 'Y', 'blue')
    plain = replace(obs, me=replace(obs.me, cards=[]))
    with_haven = replace(obs, me=replace(obs.me, cards=['Haven']))
    assert agent._enabler_value(plain, brown) == 0.0
    assert agent._enabler_value(with_haven, brown) > 0.5
    assert agent._enabler_value(with_haven, blue) == 0.0


def test_feeding_a_neighbours_guild_costs_a_little(obs):
    agent = AdaptiveAgent()
    fed = replace(obs, me=replace(obs.me, cards=[]), left=replace(obs.left, cards=['Workers Guild']))
    assert agent._enabler_value(fed, _view(obs, 'X', 'brown')) < 0


def test_counting_cards_get_future_credit(obs):
    agent = AdaptiveAgent()
    haven = _view(obs, 'Haven', 'yellow', _summaries('Haven'))
    few = replace(obs, me=replace(obs.me, cards_by_color={}))
    many = replace(obs, me=replace(obs.me, cards_by_color={'brown': 4}))
    assert agent._payoff_future(many, haven) > agent._payoff_future(few, haven) > 0


def test_stage_counters_boost_the_wonder_action(obs):
    agent = AdaptiveAgent()
    with_arena = replace(obs, me=replace(obs.me, cards=['Arena']))
    assert agent._stage_payoffs(with_arena) > agent._stage_payoffs(replace(obs, me=replace(obs.me, cards=[])))
