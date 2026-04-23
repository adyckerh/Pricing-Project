# Pricing Project — ORIE 5132 (Spring 2026)

Choice Modeling, Assortment Optimization, and Pricing on Expedia hotel search data.

## Dependencies

```
numpy pandas scipy scikit-learn gurobipy
```

Gurobi requires a free academic license (Problems 5 and 6). All other problems use scipy only.

## Files

| File | Purpose |
|---|---|
| `pricing project.ipynb` | Main notebook — run cells top to bottom |
| `Data/data.csv` | Full Expedia dataset (153,009 rows, 8,354 queries) |
| `Data/data1–4.csv` | Small hotel sets for assortment/pricing problems |
| `Data/data_empty_bool.csv` | data.csv with AI-generated booking_bool (Problem 7a) |
| `Data/test_20_predicted.csv` | AI-predicted bookings on 20% holdout (Problem 7b) |
| `Data/test_ground_truth.csv` | Real booking decisions for the 20% holdout |

## How to run

Execute cells in order (Cell 1 through Cell 23). Each cell depends on globals set by prior cells. Re-run from Cell 1 if kernel is restarted.

## Problem summary

| Problem | Topic |
|---|---|
| 1 | MNL estimation via MLE (L-BFGS-B), standardized and raw coefficients |
| 2 | Assortment optimization — revenue-ordered algorithm |
| 3 | Pricing optimization — gradient descent (equal-markup result) |
| 4 | Mixture MNL — early vs. late customers |
| 5 | MILP assortment optimization under mixture MNL (Gurobi) |
| 6 | Alternative mixture MNL — solo vs. group travelers |
| 7 | AI agent as customer — zero-shot simulation + predictive accuracy |
