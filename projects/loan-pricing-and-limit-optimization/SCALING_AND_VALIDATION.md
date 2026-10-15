# License-free credit-offer scaling and validation benchmark

This is an additional research benchmark next to the existing small loan_pricing.py demonstration. It does not change the original model or require Gurobi, Xpress, Hexaly, or a commercial license.

## Run

From the repository root after installing development dependencies:

    pip install -e ".[dev]"
    python -m banking_optimization.credit_offer_benchmark --customers 500 --seed 42 --time-limit 30 --trials 250
    pytest tests/test_credit_offer_benchmark.py

The benchmark creates six candidate offers per synthetic customer. To explore size, try --customers 1000 first, then 10000 with a longer time limit and appropriate hardware. **A 100,000-customer run has not been validated.** Sparse constraint storage reduces model-construction memory versus a dense customer-offer matrix, but it does not guarantee fast MILP solutions. Large instances can finish without a feasible solver incumbent.

## Decision and assumptions

For each customer, select **at most one** rate/credit-limit offer; selecting none is allowed. Optimize expected net contribution under:

- expected exposure;
- expected risk-weighted assets (simplified proxy);
- expected credit loss;
- stressed expected loss (optional).

Both acceptance probabilities and probability of default (PD) are supplied, noisy, synthetic predictions. They are not trained or validated credit-risk models. Loan utilization, loss-given-default (LGD), funding cost, origination cost, and capital charges are simplified assumptions. Portfolio budgets are reproducible fractions of synthetic offer maxima—not regulatory ratios. All monetary values are arbitrary synthetic units.

Customer constraints use a sparse SciPy/HiGHS MILP, not the dense numpy.vstack pattern of the introductory model. Only open-source dependencies already declared in pyproject.toml are used.

## Baseline and evaluation

A deterministic feasible greedy benchmark selects profitable offers while respecting the same ex-ante constraints. It is a heuristic, not an exact benchmark for the optimum.

The output distinguishes:

1. **Estimated objective:** expected profit from predicted offer economics.
2. **Hidden-propensity expected value:** uses separately generated true acceptance and default probabilities. In this synthetic simulation only, those are known to the evaluator but withheld from the optimizer.
3. **Stress experiment:** increases hidden default probabilities by 50% (capped at 1).
4. **Monte Carlo distribution:** repeated accept/default outcomes, with mean, standard deviation, 5th/95th percentiles, and proportion of loss-making trials.
5. **Policy comparison:** estimated and hidden-propensity profit for the greedy and optimized policies, including the difference in hidden-propensity expected profit.

The hidden propensities are part of a synthetic data-generating process. Calling them an out-of-sample research evaluation does not make them an empirical out-of-time validation or demonstrate real lending uplift. The observed comparison may favor either policy.

### Solver result interpretation

- optimal_or_gap_met: HiGHS returned success under the requested mip_rel_gap threshold; this need not be mathematically exact optimality.
- feasible_not_proven: the time limit was reached but a feasible incumbent passed all portfolio and one-offer checks.
- no_feasible_incumbent: the solver had no valid incumbent within the limit. Only the greedy baseline is reported; it is **not** relabeled as an optimization result.

The output reports mip_gap when provided by HiGHS. Real execution time is machine-specific and no speedup claim should be inferred without matched benchmarking.

## Current scope and limitations

- Six discrete offers per customer; there is no continuous pricing model.
- No actual customer records, trained propensity/PD estimator, calibration, counterfactual identification, approval-policy engine, or fairness audit.
- No customer-level banking policy, legally applicable capital calculation, IFRS 9/CECL/ECL production accounting, transaction-ledger integration, or local regulatory compliance.
- The pricing, exposure, recovery, funding, and default model are simplified and synthetic.
- Python/SciPy and matrix construction can scale differently from the branch-and-bound solve; empirical benchmarks must report the size, runtime, incumbent status, and solution gap.
- End-to-end model training with chronological holdouts, causal acceptance estimation, external validation, model monitoring, and governance would be separate projects.

**Research/education only. Not a credit decision system or financial, investment, legal, or regulatory advice.**
