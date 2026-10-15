"""Checks for sparse, license-free credit-offer research benchmark."""

import numpy as np
import pandas as pd
import pytest

from banking_optimization.credit_offer_benchmark import (
    PortfolioLimits, evaluate_policy, greedy_feasible, make_synthetic_offers,
    reference_limits, run_benchmark, simulate_policy, solve_sparse,
)


def test_generator_is_deterministic_and_has_exactly_six_offers_per_customer():
    a = make_synthetic_offers(24, seed=7)
    pd.testing.assert_frame_equal(a, make_synthetic_offers(24, seed=7))
    assert len(a) == 144
    assert (a.groupby("customer_id").size() == 6).all()
    assert a["accept_prob"].between(0, 1).all()
    assert a["pd"].between(0, 1).all()


def test_sparse_optimizer_respects_limits_and_one_offer():
    offers = make_synthetic_offers(80, 10)
    limits = reference_limits(offers, 0.45)
    best = solve_sparse(offers, limits, time_limit=15, mip_rel_gap=0)
    assert best.offers_selected > 0
    assert best.expected_profit > 0
    assert best.expected_exposure <= limits.exposure + 1e-6
    assert best.expected_rwa <= limits.rwa + 1e-6
    assert best.expected_loss <= limits.expected_loss + 1e-6
    assert best.stressed_loss <= limits.stressed_loss + 1e-6
    assert offers.iloc[list(best.selected_indices)]["customer_id"].is_unique


def test_milp_dominates_greedy_on_small_exact_instance():
    offers = make_synthetic_offers(15, 5)
    limits = reference_limits(offers, .5)
    best = solve_sparse(offers, limits, time_limit=20, mip_rel_gap=0)
    baseline = greedy_feasible(offers, limits)
    assert best.status == "optimal_or_gap_met"
    assert best.expected_profit >= baseline.expected_profit - 1e-6


def test_true_evaluation_and_monte_carlo_are_consistent():
    offers = make_synthetic_offers(35, 2)
    solution = solve_sparse(offers, reference_limits(offers), time_limit=20, mip_rel_gap=0)
    exact = evaluate_policy(offers, solution)["true_expected_profit"]
    simulated = simulate_policy(offers, solution, trials=6000, seed=1)
    # Standard error of the Monte Carlo mean, plus a small numerical allowance.
    se = simulated["std_profit"] / np.sqrt(6000)
    assert abs(simulated["mean_profit"] - exact) < 5 * se + 0.1
    stressed = evaluate_policy(offers, solution, pd_shift=1.5)
    assert stressed["true_expected_profit"] <= exact + 1e-9


def test_no_offer_is_feasible_under_zero_budget():
    offers = make_synthetic_offers(12, 8)
    limits = PortfolioLimits(0, 0, 0, 0)
    result = solve_sparse(offers, limits, time_limit=15)
    assert result.offers_selected == 0
    assert result.expected_profit == 0


def test_validation_and_small_end_to_end_benchmark():
    offers = make_synthetic_offers(8, 2)
    with pytest.raises(ValueError):
        make_synthetic_offers(0)
    with pytest.raises(ValueError):
        reference_limits(offers, 1.2)
    with pytest.raises(ValueError):
        solve_sparse(offers, PortfolioLimits(-1, 2, 3, 4))
    with pytest.raises(ValueError):
        solve_sparse(offers, reference_limits(offers), time_limit=0)
    result = run_benchmark(12, 4, time_limit=20, trials=25)
    assert result["solver"].startswith("SciPy")
    assert result["optimized"]["offers_selected"] > 0
    assert result["offers"] == 72
