"""Two-stage bank credit, liquidity and capital planning under stress.

Linear recourse model: loan mix and cash reserve are chosen before macro
scenarios; emergency funding and capital issuance respond after stress.
All numbers are synthetic and regulatory ratios are illustrative proxies.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog

SCENARIOS = ("base", "slowdown", "severe")
PROB = np.array([0.60, 0.30, 0.10])
MARGIN = np.array([0.055, 0.095])
RISK_WEIGHT = np.array([0.60, 1.00])
LGD = np.array([0.45, 0.65])
DEFAULT_RATE = np.array([[0.015, 0.040], [0.060, 0.120],
                         [0.150, 0.270]])
FIXED_OUTFLOW = np.array([8.0, 15.0, 23.0])
RUNOFF_SHARE = np.array([0.10, 0.18, 0.25])
FUNDING_COST = np.array([0.06, 0.09, 0.16])
TOTAL_FUNDING = 100.0
START_CAPITAL = 9.0
MIN_CAPITAL_RATIO = 0.12


@dataclass(frozen=True)
class CapitalPlan:
    loans: np.ndarray
    cash_reserve: float
    emergency_funding: np.ndarray
    equity_injections: np.ndarray
    scenario_profit: np.ndarray
    scenario_capital_surplus: np.ndarray
    scenario_liquidity_surplus: np.ndarray
    expected_profit: float
    cvar_loss: float
    risk_adjusted_objective: float


def tail_cvar(losses: np.ndarray, confidence: float = 0.85) -> float:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0,1)")
    return float(min(t + np.dot(PROB, np.maximum(losses - t, 0)) /
                     (1 - confidence) for t in losses))


def solve(
    risk_aversion: float = 0.40,
    cash_carry: float = 0.015,
    capital_issue_cost: float = 0.35,
    confidence: float = 0.85,
) -> CapitalPlan:
    if min(risk_aversion, cash_carry, capital_issue_cost) < 0 or not 0 < confidence < 1:
        raise ValueError("Invalid risk, cost, or tail confidence")

    names = ["loan_low", "loan_high", "reserve"]
    names += [f"fund_{s}" for s in range(3)]
    names += [f"equity_{s}" for s in range(3)]
    names += ["eta"] + [f"z_{s}" for s in range(3)]
    idx = {v: i for i, v in enumerate(names)}
    eq, rhs_eq, le, rhs_le = [], [], [], []
    def add(rows, right, values, bound):
        row = np.zeros(len(names))
        for key, value in values.items():
            row[idx[key]] = value
        rows.append(row)
        right.append(float(bound))

    add(eq, rhs_eq, {"loan_low": 1, "loan_high": 1, "reserve": 1},
        TOTAL_FUNDING)
    # Prevent the riskier loan segment from dominating the origination book.
    add(le, rhs_le, {"loan_high": 0.40, "loan_low": -0.60}, 0.0)

    c = np.zeros(len(names))
    scenario_profit_rows = []
    for s in range(3):
        income = MARGIN - DEFAULT_RATE[s] * LGD
        # profit = lending margin - credit loss - stress borrowing - capital issue cost
        #          - opportunity cost of idle liquidity.
        prof = {"loan_low": float(income[0]), "loan_high": float(income[1]),
                "reserve": -cash_carry,
                f"fund_{s}": -float(FUNDING_COST[s]),
                f"equity_{s}": -capital_issue_cost}
        scenario_profit_rows.append(prof)
        # Stress outflows must be funded with reserve and contingency borrowing.
        add(le, rhs_le, {"loan_low": float(RUNOFF_SHARE[s]),
                         "loan_high": float(RUNOFF_SHARE[s]),
                         "reserve": -1.0, f"fund_{s}": -1.0},
            -float(FIXED_OUTFLOW[s]))
        # Capital after credit loss, interest income, and funding cost
        # + scenario equity injection >= minimum ratio * risk-weighted loans.
        capital = {"loan_low": float(MIN_CAPITAL_RATIO * RISK_WEIGHT[0] - income[0]),
                   "loan_high": float(MIN_CAPITAL_RATIO * RISK_WEIGHT[1] - income[1]),
                   f"fund_{s}": float(FUNDING_COST[s]),
                   f"equity_{s}": -1.0}
        add(le, rhs_le, capital, START_CAPITAL)
        # z_s >= (-profit_s) - eta: shortfall of P&L, rather than VaR of PD.
        add(le, rhs_le,
            {**{k: -v for k, v in prof.items()}, "eta": -1.0,
             f"z_{s}": -1.0}, 0.0)
        for name, value in prof.items():
            c[idx[name]] -= float(PROB[s]) * value
        c[idx[f"z_{s}"]] = risk_aversion * PROB[s] / (1.0 - confidence)
    c[idx["eta"]] = risk_aversion
    bounds = [(0, None)] * len(names)
    bounds[idx["eta"]] = (None, None)
    answer = linprog(c, A_ub=np.array(le), b_ub=np.array(rhs_le),
                     A_eq=np.array(eq), b_eq=np.array(rhs_eq),
                     bounds=bounds, method="highs")
    if not answer.success:
        raise RuntimeError(f"Capital planning LP failed: {answer.message}")
    v = dict(zip(names, answer.x))
    loans = np.array([v["loan_low"], v["loan_high"]])
    reserve = float(v["reserve"])
    funding = np.array([v[f"fund_{s}"] for s in range(3)])
    equity = np.array([v[f"equity_{s}"] for s in range(3)])
    profit = np.array([sum(coef * v[key] for key, coef in row.items())
                       for row in scenario_profit_rows])
    capital_surplus = (START_CAPITAL + (MARGIN - DEFAULT_RATE * LGD) @ loans
                       - FUNDING_COST * funding + equity
                       - MIN_CAPITAL_RATIO * np.dot(RISK_WEIGHT, loans))
    liquidity_surplus = (reserve + funding - FIXED_OUTFLOW
                         - RUNOFF_SHARE * loans.sum())
    expected = float(np.dot(PROB, profit))
    cvar = tail_cvar(-profit, confidence)
    return CapitalPlan(loans, reserve, funding, equity, profit, capital_surplus,
                       liquidity_surplus, expected, cvar,
                       expected - risk_aversion * cvar)


def main() -> None:
    result = solve()
    print("Origination by segment:", result.loans.round(3))
    print(f"Liquidity reserve: {result.cash_reserve:.3f}")
    for s, name in enumerate(SCENARIOS):
        print(f"{name}: profit={result.scenario_profit[s]:.3f}, "
              f"funding={result.emergency_funding[s]:.3f}, "
              f"equity={result.equity_injections[s]:.3f}, "
              f"capital surplus={result.scenario_capital_surplus[s]:.3f}")
    print(f"Expected profit: {result.expected_profit:.3f}")
    print(f"CVaR loss: {result.cvar_loss:.3f}")
    print("Synthetic regulatory-style proxies; not regulatory capital advice.")


if __name__ == "__main__":
    main()
