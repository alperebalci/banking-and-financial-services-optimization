"""Reconciliation and stress-feasibility tests for integrated capital planning."""
import numpy as np
import pytest

from banking_optimization.stress_capital import (
    DEFAULT_RATE, FIXED_OUTFLOW, FUNDING_COST, LGD, MARGIN, MIN_CAPITAL_RATIO,
    PROB, RISK_WEIGHT, RUNOFF_SHARE, START_CAPITAL, TOTAL_FUNDING, solve,
)


def test_all_scenarios_are_feasible_and_accounting_reconciles():
    result = solve()
    assert np.isclose(result.loans.sum() + result.cash_reserve, TOTAL_FUNDING)
    assert result.loans[1] <= 0.6 * result.loans.sum() + 1e-7
    assert (result.scenario_liquidity_surplus >= -1e-7).all()
    assert (result.scenario_capital_surplus >= -1e-7).all()
    assert (result.emergency_funding >= -1e-9).all()
    assert (result.equity_injections >= -1e-9).all()
    np.testing.assert_allclose(
        result.cash_reserve + result.emergency_funding
        - FIXED_OUTFLOW - RUNOFF_SHARE * result.loans.sum(),
        result.scenario_liquidity_surplus,
    )
    actual_profit = ((MARGIN - DEFAULT_RATE * LGD) @ result.loans
                     - FUNDING_COST * result.emergency_funding
                     - 0.35 * result.equity_injections
                     - 0.015 * result.cash_reserve)
    np.testing.assert_allclose(result.scenario_profit, actual_profit)
    required_rwa = MIN_CAPITAL_RATIO * np.dot(RISK_WEIGHT, result.loans)
    capital = (START_CAPITAL + (MARGIN - DEFAULT_RATE * LGD) @ result.loans
               - FUNDING_COST * result.emergency_funding
               + result.equity_injections - required_rwa)
    np.testing.assert_allclose(capital, result.scenario_capital_surplus)
    assert np.isclose(result.expected_profit, np.dot(PROB, actual_profit))


def test_penalizing_tail_losses_reduces_optimal_loss_cvar():
    neutral = solve(risk_aversion=0)
    cautious = solve(risk_aversion=1.0)
    assert cautious.cvar_loss <= neutral.cvar_loss + 1e-6
    assert np.isclose(cautious.risk_adjusted_objective,
                      cautious.expected_profit - cautious.cvar_loss)


@pytest.mark.parametrize("kwargs", [
    {"risk_aversion": -1}, {"cash_carry": -0.01},
    {"capital_issue_cost": -0.5}, {"confidence": 1.0},
])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        solve(**kwargs)
