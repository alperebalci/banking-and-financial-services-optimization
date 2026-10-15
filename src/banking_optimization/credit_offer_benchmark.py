"""License-free, sparse predict-then-optimize credit-offer research benchmark.

All parameters and customer records are synthetic. No underwriting decisions.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix, vstack


@dataclass(frozen=True)
class PortfolioLimits:
    exposure: float
    rwa: float
    expected_loss: float
    stressed_loss: float | None = None


@dataclass(frozen=True)
class PolicyResult:
    selected_indices: tuple[int, ...]
    expected_profit: float
    expected_exposure: float
    expected_rwa: float
    expected_loss: float
    stressed_loss: float
    solve_seconds: float
    status: str
    mip_gap: float | None = None

    @property
    def offers_selected(self) -> int:
        return len(self.selected_indices)


def make_synthetic_offers(customers: int = 500, seed: int = 42) -> pd.DataFrame:
    """Six loan offers per customer with noisy predictions and hidden true rates.

    Monetary values are synthetic units. All rates are illustrative, not calibrated.
    """
    if customers < 1:
        raise ValueError("customers must be positive")
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(customers, dtype=np.int32), 6)
    option = np.tile(np.arange(6), customers)
    segment_per_customer = rng.integers(0, 3, customers)
    base_limit = rng.uniform(20, 110, customers)
    base_pd = rng.uniform(0.008, 0.075, customers)
    base_accept = rng.uniform(0.55, 0.92, customers)
    lgd_customer = rng.uniform(0.28, 0.60, customers)
    utilization_customer = rng.uniform(0.40, 0.85, customers)
    rate = np.array([0.09, 0.09, 0.115, 0.115, 0.14, 0.14])[option]
    limit = base_limit[ids] * np.array([0.75, 1.15] * 3)[option]
    drawn = limit * utilization_customer[ids]

    # Offer-specific, controlled prediction errors: no trained credit model is claimed.
    true_accept = np.clip(
        base_accept[ids] - 2.6 * (rate - 0.09)
        + 0.065 * ((option % 2) == 1), 0.05, 0.98
    )
    true_pd = np.clip(
        base_pd[ids] + 0.010 * ((option % 2) == 1)
        + 0.16 * (rate - 0.09), 0.002, 0.35
    )
    accept = np.clip(true_accept + rng.normal(0, 0.065, len(ids)), 0.01, 0.99)
    pd_est = np.clip(true_pd * np.exp(rng.normal(0, 0.30, len(ids))), 0.001, 0.60)
    lgd = lgd_customer[ids]
    risk_weight = 0.55 + 5.0 * pd_est
    funding_rate, capital_charge_rate, origination_cost = 0.038, 0.018, 0.35
    expected_exposure = accept * drawn
    expected_rwa = expected_exposure * risk_weight
    expected_loss = expected_exposure * pd_est * lgd
    stressed_loss = expected_exposure * np.clip(1.7 * pd_est + 0.01, 0, 1) * lgd
    expected_profit = (expected_exposure * (rate - funding_rate)
                       - expected_loss - capital_charge_rate * expected_rwa
                       - accept * origination_cost)
    true_exposure = true_accept * drawn
    true_rwa = true_exposure * risk_weight
    true_profit = (true_exposure * (rate - funding_rate)
                   - true_exposure * true_pd * lgd
                   - capital_charge_rate * true_rwa
                   - true_accept * origination_cost)

    return pd.DataFrame({
        "customer_id": ids, "offer_option": option, "segment": segment_per_customer[ids],
        "rate": rate, "limit": limit, "drawn": drawn, "lgd": lgd,
        "risk_weight": risk_weight, "accept_prob": accept, "pd": pd_est,
        "true_accept_prob": true_accept, "true_pd": true_pd,
        "expected_exposure": expected_exposure, "expected_rwa": expected_rwa,
        "expected_loss": expected_loss, "stressed_loss": stressed_loss,
        "expected_profit": expected_profit, "true_profit": true_profit,
    })


def reference_limits(offers: pd.DataFrame, fraction: float = 0.55) -> PortfolioLimits:
    """Reproducible binding-ish budgets scaled by portfolio size, not regulation."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1]")
    group = offers.groupby("customer_id", sort=False)
    return PortfolioLimits(*(
        float(group[col].max().sum() * fraction)
        for col in ("expected_exposure", "expected_rwa", "expected_loss", "stressed_loss")
    ))


def _validate_offers(offers: pd.DataFrame, limits: PortfolioLimits, objective: str) -> None:
    required = {"customer_id", "expected_exposure", "expected_rwa",
                "expected_loss", "stressed_loss", objective}
    if not required.issubset(offers.columns):
        raise ValueError(f"missing columns: {sorted(required - set(offers.columns))}")
    if offers.empty or offers["customer_id"].isna().any():
        raise ValueError("nonempty offers and customer identifiers required")
    arr = offers[list(required - {"customer_id"})].to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("offers must be finite")
    if (offers[["expected_exposure", "expected_rwa", "expected_loss", "stressed_loss"]] < 0).any().any():
        raise ValueError("risk/exposure columns cannot be negative")
    if any(x < 0 or not np.isfinite(x) for x in asdict(limits).values() if x is not None):
        raise ValueError("limits must be finite and nonnegative")


def _totals(offers: pd.DataFrame, indices: tuple[int, ...], limits: PortfolioLimits,
            seconds: float, status: str, gap: float | None) -> PolicyResult:
    selected = offers.iloc[list(indices)]
    result = PolicyResult(
        indices, *(float(selected[col].sum()) for col in (
            "expected_profit", "expected_exposure", "expected_rwa", "expected_loss",
            "stressed_loss")), seconds, status, gap
    )
    customer_ids = selected["customer_id"]
    if customer_ids.duplicated().any():
        raise RuntimeError("multiple offers selected for a customer")
    pairs = ((result.expected_exposure, limits.exposure),
             (result.expected_rwa, limits.rwa),
             (result.expected_loss, limits.expected_loss))
    if limits.stressed_loss is not None:
        pairs += ((result.stressed_loss, limits.stressed_loss),)
    if any(used > budget + max(1e-6, 1e-8 * budget) for used, budget in pairs):
        raise RuntimeError("selected policy violates portfolio limits")
    return result


def solve_sparse(offers: pd.DataFrame, limits: PortfolioLimits, *,
                 objective: str = "expected_profit", time_limit: float = 30.0,
                 mip_rel_gap: float = 0.01) -> PolicyResult:
    """O(number of offers) sparse matrix, one choice per customer (including none).

    Time-limited feasible incumbents are returned as 'feasible_not_proven', not
    called optimal. With no feasible incumbent, raise instead of guessing.
    """
    _validate_offers(offers, limits, objective)
    if time_limit <= 0 or not 0 <= mip_rel_gap < 1:
        raise ValueError("invalid solver settings")
    t0 = perf_counter()
    n = len(offers)
    codes, unique_ids = pd.factorize(offers["customer_id"], sort=False)
    customer_constraints = coo_matrix(
        (np.ones(n), (codes, np.arange(n))), shape=(len(unique_ids), n)
    ).tocsr()
    columns = ["expected_exposure", "expected_rwa", "expected_loss"]
    caps = [limits.exposure, limits.rwa, limits.expected_loss]
    if limits.stressed_loss is not None:
        columns.append("stressed_loss")
        caps.append(limits.stressed_loss)
    budget_rows = csr_matrix(offers[columns].to_numpy(float).T)
    matrix = vstack([budget_rows, customer_constraints], format="csr")
    lower = np.full(matrix.shape[0], -np.inf)
    upper = np.concatenate([np.array(caps), np.ones(len(unique_ids))])
    answer = milp(
        c=-offers[objective].to_numpy(dtype=float),
        integrality=np.ones(n, dtype=np.uint8),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(matrix, lb=lower, ub=upper),
        options={"time_limit": float(time_limit), "mip_rel_gap": float(mip_rel_gap)},
    )
    if answer.x is None or answer.status not in (0, 1):
        raise RuntimeError(f"MILP has no feasible solution: {answer.message}")
    selected = tuple(int(i) for i in np.flatnonzero(answer.x > 0.5))
    gap = getattr(answer, "mip_gap", None)
    return _totals(offers, selected, limits, perf_counter() - t0,
                   "optimal_or_gap_met" if answer.status == 0 else "feasible_not_proven",
                   float(gap) if gap is not None and np.isfinite(gap) else None)


def greedy_feasible(offers: pd.DataFrame, limits: PortfolioLimits) -> PolicyResult:
    """Simple feasible benchmark, *not* a replacement for MILP optimality."""
    _validate_offers(offers, limits, "expected_profit")
    t0 = perf_counter()
    chosen = []
    seen: set[object] = set()
    cols = ["expected_exposure", "expected_rwa", "expected_loss"]
    caps = np.array([limits.exposure, limits.rwa, limits.expected_loss])
    if limits.stressed_loss is not None:
        cols.append("stressed_loss")
        caps = np.append(caps, limits.stressed_loss)
    running = np.zeros(len(cols))
    # Stable ordering makes results reproducible even with identical profitability.
    order = np.argsort(-offers["expected_profit"].to_numpy(), kind="stable")
    customers = offers["customer_id"].to_numpy()
    costs = offers[cols].to_numpy(dtype=float)
    profits = offers["expected_profit"].to_numpy(dtype=float)
    for idx in order:
        if profits[idx] <= 0:
            break
        if customers[idx] in seen or np.any(running + costs[idx] > caps + 1e-9):
            continue
        running += costs[idx]
        chosen.append(int(idx))
        seen.add(customers[idx])
    return _totals(offers, tuple(chosen), limits, perf_counter() - t0,
                   "heuristic_feasible", None)


def evaluate_policy(offers: pd.DataFrame, policy: PolicyResult, *,
                    pd_shift: float = 1.0) -> dict[str, float]:
    """Synthetic hidden-propensity expected value, distinct from estimated profit."""
    if pd_shift < 0 or not np.isfinite(pd_shift):
        raise ValueError("pd_shift must be finite and nonnegative")
    rows = offers.iloc[list(policy.selected_indices)]
    accepted = rows["true_accept_prob"].to_numpy()
    defaults = np.clip(rows["true_pd"].to_numpy() * pd_shift, 0, 1)
    exp_drawn = accepted * rows["drawn"].to_numpy()
    loss = exp_drawn * defaults * rows["lgd"].to_numpy()
    true_profit = (exp_drawn * (rows["rate"].to_numpy() - 0.038)
                   - loss - 0.018 * exp_drawn * rows["risk_weight"].to_numpy()
                   - 0.35 * accepted)
    return {"true_expected_profit": float(true_profit.sum()),
            "true_expected_loss": float(loss.sum()),
            "predicted_profit": float(policy.expected_profit),
            "prediction_value_error": float(policy.expected_profit - true_profit.sum())}


def simulate_policy(offers: pd.DataFrame, policy: PolicyResult, *,
                    trials: int = 500, seed: int = 2026, pd_shift: float = 1.0) -> dict[str, float]:
    """Out-of-sample Monte Carlo profit distribution using bounded memory."""
    if trials < 2:
        raise ValueError("trials must be >= 2")
    if pd_shift < 0 or not np.isfinite(pd_shift):
        raise ValueError("invalid pd_shift")
    chosen = offers.iloc[list(policy.selected_indices)]
    a = chosen["true_accept_prob"].to_numpy()
    d = np.clip(chosen["true_pd"].to_numpy() * pd_shift, 0, 1)
    drawn = chosen["drawn"].to_numpy()
    contribution = drawn * (chosen["rate"].to_numpy() - 0.038)
    contribution -= 0.018 * drawn * chosen["risk_weight"].to_numpy()
    contribution -= 0.35
    loss = drawn * chosen["lgd"].to_numpy()
    rng = np.random.default_rng(seed)
    results = np.zeros(trials)
    # Each loop holds at most 256 offers × trials binary outcomes in memory.
    for begin in range(0, len(chosen), 256):
        end = min(begin + 256, len(chosen))
        accepted = rng.random((trials, end - begin)) < a[begin:end]
        defaulted = rng.random((trials, end - begin)) < d[begin:end]
        results += (accepted * (contribution[begin:end] - defaulted * loss[begin:end])).sum(axis=1)
    return {"mean_profit": float(np.mean(results)),
            "std_profit": float(np.std(results, ddof=1)),
            "p05_profit": float(np.quantile(results, 0.05)),
            "p95_profit": float(np.quantile(results, 0.95)),
            "loss_probability": float(np.mean(results < 0))}


def run_benchmark(customers: int = 500, seed: int = 42, *,
                  time_limit: float = 30.0, trials: int = 250) -> dict:
    offers = make_synthetic_offers(customers, seed)
    limits = reference_limits(offers)
    greedy = greedy_feasible(offers, limits)
    greedy_truth = evaluate_policy(offers, greedy)
    result = {"customers": customers, "offers": len(offers),
              "solver": "SciPy milp / HiGHS (open source)",
              "limits": asdict(limits),
              "greedy": {**{k: v for k, v in asdict(greedy).items()
                             if k != "selected_indices"},
                         "offers_selected": greedy.offers_selected},
              "greedy_out_of_sample": greedy_truth,
              "disclaimer": "Synthetic research only; not banking or regulatory advice."}
    try:
        optimized = solve_sparse(offers, limits, time_limit=time_limit)
    except RuntimeError as error:
        # A large/time-limited MILP can terminate before obtaining an incumbent.
        # Do not present the greedy baseline as an optimized solution.
        result["optimized"] = {"status": "no_feasible_incumbent",
                               "reason": str(error)}
        return result
    truth = evaluate_policy(offers, optimized)
    result.update({
        "optimized": {**{k: v for k, v in asdict(optimized).items()
                          if k != "selected_indices"},
                      "offers_selected": optimized.offers_selected},
        "out_of_sample": truth,
        "true_profit_uplift_vs_greedy": (truth["true_expected_profit"]
                                        - greedy_truth["true_expected_profit"]),
        "stress_pd_x1_5": evaluate_policy(offers, optimized, pd_shift=1.5),
        "monte_carlo": simulate_policy(offers, optimized, trials=trials, seed=seed + 1),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=float, default=30)
    parser.add_argument("--trials", type=int, default=250)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.customers, args.seed,
                                   time_limit=args.time_limit, trials=args.trials), indent=2))


if __name__ == "__main__":
    main()
