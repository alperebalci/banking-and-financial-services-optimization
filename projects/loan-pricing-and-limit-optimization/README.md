# Loan Pricing and Credit-Limit Optimization

A discrete offer-design model that chooses at most one rate/limit combination per synthetic customer.

The model separates estimation from decision-making. Acceptance probability and credit risk are treated as offer-level inputs; the MILP then chooses the portfolio of offers.

## Offer economics

Each candidate offer has:

- an interest rate;
- a credit limit;
- expected utilization;
- acceptance probability;
- PD and LGD;
- risk weight.

Expected profit deducts funding cost, expected credit loss, a capital charge, and an origination cost.

## Portfolio constraints

- expected booked exposure budget;
- expected RWA budget;
- expected-loss budget;
- at most one offer per customer.

## Run

```bash
python -m banking_optimization.loan_pricing
pytest tests/test_loan_pricing.py
```

## License-free scaling benchmark

A separate open-source sparse-MILP experiment with synthetic noisy offer forecasts,
portfolio risk/stress limits, a feasible greedy baseline, and synthetic
outcome evaluation is available in the module
`banking_optimization.credit_offer_benchmark`.

Run:

```bash
python -m banking_optimization.credit_offer_benchmark --customers 500 --seed 42
pytest tests/test_credit_offer_benchmark.py
```

See [Scaling and validation](SCALING_AND_VALIDATION.md) for assumptions,
status interpretation, and large-instance limitations.

## Extension path

A production research version could replace the synthetic acceptance function with a causal/choice model, add continuous pricing, fairness and policy controls, relationship value, prepayment, utilization uncertainty, and robust or stochastic optimization.

## Limitations

This project is educational and synthetic. It is not a lending policy, underwriting system, personalized financial recommendation, or production pricing engine.
