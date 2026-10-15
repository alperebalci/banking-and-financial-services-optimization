# Integrated Credit, Liquidity, and Capital Stress Planning

This research exercise bridges the existing [credit-capital allocation](../credit-capital-allocation/) and [stochastic ALM](../asset-liability-management/) projects. Unlike two separate optimizers, it chooses the loan mix and cash reserve **jointly**, then handles stress-specific liquidity funding and new equity as **recourse**.

## Two-stage decisions

**Before the macroeconomic scenario:** allocate a synthetic 100-unit loan-or-cash budget to low- and high-risk loan segments and an upfront cash reserve. The risky segment is capped at 60% of total loans.

**After the scenario is known:** choose emergency borrowing and equity injection for each of three scenarios: base (60%), slowdown (30%), severe (10%). These recourse actions are *not* known to the initial optimizer, which uses one common first-stage loan and reserve decision.

## Economic model and feasibility

For segment `i` and scenario `s`, pretax model income is:

```text
loan_income_s = sum_i loans_i * (margin_i - PD_s,i * LGD_i)

scenario_profit_s = loan_income_s - funding_cost_s * borrowing_s
                    - equity_issue_premium * new_equity_s
                    - idle_cash_carry * reserve
```

Stress liquidity requires `reserve + borrowing_s >= fixed_outflow_s + runoff_s * total_loans`.

A deliberately simplified capital rule requires:

```text
starting_capital + loan_income_s - funding_cost_s * borrowing_s
+ new_equity_s >= illustrative_minimum_capital_ratio * risk_weighted_loans
```

The objective maximizes expected scenario profit less `lambda * CVaR` of **negative** scenario profit. The linear program has no binary decisions; the previous credit-capital project retains its separate loan-by-loan MILP.

## Run and test

```bash
pip install -e ".[dev]"
python -m banking_optimization.stress_capital
pytest -q tests/test_stress_capital.py
```

The regression tests reconcile the balance-sheet budget, lending concentration, scenario liquidity, scenario capital surplus, realized profit identities, and downside-risk behavior.

## Research next steps

- Calibrate PD, LGD, prepayment, deposit runoff, and funding curves with an **out-of-time** test set.
- Add **multi-period** capital conservation, dividend policy, RWAs that migrate with ratings, maturity ladders, capital conservation buffer and stress capital buffer rules for a *specified jurisdiction*.
- Tie loss and runoff scenarios to common macro factors, distinguish accounting impairments from cash outflows, and add investment-book repricing.
- Compare deterministic expected-value policy and stochastic policy, including EVPI/VSS where the benchmark conditions are valid.

**Scope:** a two-stage synthetic decision-modeling prototype. It does **not** calculate regulatory CET1, LCR, NSFR, CCAR, ICAAP, IFRS 9, CECL, or legally binding capital requirements. New equity is a priced recourse action, not a guaranteed funding source in a real stress event. Educational use only; not bank capital, investment, or regulatory advice.
