"""Tests for the agent benchmark."""

import pytest

from wonders.agents.bench import AGENTS, CATEGORIES, AgentSpec, run_bench, wilson_interval
from wonders.agents.heuristic import HeuristicAgent


def test_bench_aggregates_every_seat():
    result = run_bench(['heuristic', 'greedy', 'random'], n_games=12, seed=3)
    assert set(result.stats) == {'heuristic', 'greedy', 'random'}
    assert all(s.games == 12 for s in result.stats.values())
    assert sum(s.wins for s in result.stats.values()) == 12  # exactly one winner per game
    assert sum(s.rank_total for s in result.stats.values()) == 12 * (1 + 2 + 3)
    for s in result.stats.values():
        assert sum(s.mean_category(c) for c in CATEGORIES) == pytest.approx(s.mean_score)
    assert 'heuristic' in result.table()


def test_bench_pools_duplicate_agents_and_is_reproducible():
    a = run_bench(['heuristic', 'greedy', 'greedy'], n_games=6, seed=1)
    b = run_bench(['heuristic', 'greedy', 'greedy'], n_games=6, seed=1)
    assert a.stats['greedy'].games == 12
    assert a.table() == b.table()


def test_bench_parallel_matches_serial():
    serial = run_bench(['heuristic', 'random', 'opportunist'], n_games=8, seed=5)
    parallel = run_bench(['heuristic', 'random', 'opportunist'], n_games=8, seed=5, processes=2)
    assert parallel.table() == serial.table()


def test_custom_agent_spec():
    spec = AgentSpec.of('timid', HeuristicAgent, weights={'military_gain': 0.0})
    result = run_bench([spec, AGENTS['greedy'], 'random'], n_games=4)
    assert result.stats['timid'].games == 4


def test_wilson_interval():
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi and hi - lo < 0.21
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(0, 0) == (0.0, 1.0)


def test_bench_rejects_unsupported_player_counts():
    with pytest.raises(ValueError):
        run_bench(['greedy', 'random'], n_games=1)  # the base deck needs 3+ players
