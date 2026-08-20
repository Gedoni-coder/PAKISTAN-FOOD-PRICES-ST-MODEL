# Pakistan Food Price Spatio-Temporal Forecasting Model

A hybrid model that predicts food commodity prices in Pakistan using both
**spatial** (province, market, lat/lon) and **temporal** (date, seasonality,
inflation trend) features, trained on WFP (World Food Programme) retail price
data.

## Data

Source: `data/wfp_food_prices_pak.csv` — 9,286 records, 2004–2022, 17
commodities, across 4 provinces (Balochistan, Khyber Pakhtunkhwa, Punjab,
Sindh) and 5 markets. Zero missing values.

## Model architecture

The model is a **hybrid**, not a single black-box regressor:

1. **Trend component** — a per-commodity log-linear regression of price
   against time. This captures the long-run inflation trend, which is the
   one thing tree-based ML models structurally cannot extrapolate beyond
   their training range.
2. **Residual component** — a `HistGradientBoostingRegressor` trained on the
   *residual* (actual price minus trend), using spatial features (latitude,
   longitude, province, market), seasonal features (cyclical month encoding),
   and commodity/category. This captures regional price differences and
   seasonality on top of the smooth trend.
3. **Final prediction** = trend + residual, converted back from log scale.

## Validation methodology

**Temporal split**, not random: trained on data before 2020-01-01, tested on
2020-01-01 onward (2,132 rows). This simulates genuine forecasting — the test
set is the *future* relative to training, not just held-out interpolation.

## Results — read this before trusting the headline number

| Metric | Test (2020–2022) | Train (in-sample) |
|---|---|---|
| RMSE | 71.02 PKR | 18.41 PKR |
| MAE | 48.60 PKR | 11.13 PKR |
| MAPE | 23.2% | 12.2% |
| R² (pooled across all commodities) | **0.884** | 0.980 |

**The pooled R²=0.884 is misleading on its own.** It's inflated by the huge
price-scale differences between commodities (Ghee ~10x the price of Wheat) —
a model that just roughly separates "cheap" from "expensive" commodities
scores well on pooled R² without actually forecasting any single commodity
well.

**Per-commodity R² on the test set is the metric that matters, and it's
mostly negative:**

| Commodity | R² | MAPE |
|---|---|---|
| Wage (non-qualified labour) | 0.641 | 6.6% |
| Fuel (petrol-gasoline) | 0.425 | 10.8% |
| Wheat flour | -0.060 | 15.1% |
| Fuel (diesel) | -0.137 | 11.3% |
| Wheat | -0.141 | 10.7% |
| Milk | -0.287 | 12.2% |
| Rice (coarse) | -0.610 | 15.0% |
| Poultry | -0.582 | 20.6% |
| Eggs | -1.429 | 25.5% |
| Rice (basmati, broken) | -5.287 | 31.4% |
| Lentils (masur) | -5.489 | 34.2% |
| Lentils (moong) | -4.359 | 39.1% |
| Sugar | -8.080 | 28.5% |
| Beans (mash) | -10.399 | 31.5% |
| Oil (cooking) | -2.958 | 35.6% |
| Ghee (artificial) | -3.123 | 36.9% |
| Salt | n/a — no data after 2018 in source dataset | — |

Negative R² means the model performs **worse than a naive average** for that
commodity in the test window.

### Why: this isn't a bug, it's what happened in Pakistan 2020–2022

The test period covers COVID-19 supply shocks and the *start* of Pakistan's
2022 currency/inflation crisis — genuine structural breaks in price behavior
that no smooth trend extrapolation can anticipate. The commodities that held
up (wage labor, fuel) are the ones with more policy-linked, less globally-
volatile pricing. The commodities that failed badly (cooking oil, ghee,
sugar, lentils) are exactly the ones most exposed to global commodity price
shocks and import-dependent supply chains during that period.

**Practical takeaway:** this model is reasonably trustworthy for near-term,
stable-regime forecasting (predicting next month's price under normal
conditions) but should not be trusted to anticipate crisis-driven price
shocks — no model trained only on historical smooth trends can do that
without additional real-time economic indicators (currency reserves, import
data, policy announcements) as inputs.

## Repo structure

```
├── data/                     # Source WFP price data
├── src/
│   └── train_hybrid_model.py # Full training + evaluation pipeline
├── models/
│   └── hybrid_price_model.joblib  # Saved trained model (trend + residual)
├── results/
│   ├── test_predictions.csv       # Actual vs predicted, test set
│   ├── model_metrics.json         # Full metrics, overall + per-commodity
│   └── actual_vs_predicted.png    # Chart, 4 representative commodities
├── requirements.txt
└── README.md
```

## Reproducing

```bash
pip install -r requirements.txt
python src/train_hybrid_model.py
```

## Limitations

- No real-time economic indicators (currency reserves, import volumes,
  policy signals) — the model relies purely on historical price patterns.
- Only 5 markets and 4 provinces represented — geographic generalization
  beyond these is unvalidated.
- Salt has no data after April 2018 in the source dataset, so it's excluded
  from test-period evaluation entirely.
