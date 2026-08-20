"""
Hybrid spatio-temporal price prediction model for Pakistan WFP food price data.

Architecture:
  1. TREND component: per-commodity linear regression of log(price) on a
     continuous time index. Captures the (extrapolating) inflation trend —
     the part a tree model structurally cannot extrapolate.
  2. RESIDUAL component: HistGradientBoostingRegressor on the residuals
     (log(price) - trend), using spatial (lat, lon, admin1, market),
     seasonal (cyclical month), and commodity/category features.
     Captures regional differences, seasonality, and commodity-specific
     deviations from the smooth trend.
  Final prediction = trend + residual, back-transformed from log scale.

Validation: TEMPORAL split — train on data before 2020-01-01, test on
2020-01-01 onward. This simulates genuine forecasting (test = future,
unseen at train time), not just interpolation.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(REPO_ROOT / 'data' / 'wfp_food_prices_pak.csv', skiprows=[1])
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# ---- Feature engineering ----
df['time_idx'] = (df['date'] - df['date'].min()).dt.days / 30.44  # months since start
df['month'] = df['date'].dt.month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['log_price'] = np.log(df['price'])

CUTOFF = pd.Timestamp('2020-01-01')
train = df[df['date'] < CUTOFF].copy()
test = df[df['date'] >= CUTOFF].copy()

print(f"Train: {len(train)} rows ({train['date'].min().date()} to {train['date'].max().date()})")
print(f"Test:  {len(test)} rows ({test['date'].min().date()} to {test['date'].max().date()})")
print(f"Commodities missing from test set (no data after cutoff): "
      f"{set(df['commodity'].unique()) - set(test['commodity'].unique())}")

# ---- 1. TREND component: per-commodity log-linear trend ----
trend_models = {}
for c in train['commodity'].unique():
    sub = train[train['commodity'] == c]
    if len(sub) < 3:
        continue
    lr = LinearRegression()
    lr.fit(sub[['time_idx']], sub['log_price'])
    trend_models[c] = lr

def predict_trend_vectorized(df_):
    preds = np.full(len(df_), np.nan)
    for c, m in trend_models.items():
        mask = (df_['commodity'] == c).values
        if mask.any():
            preds[mask] = m.predict(df_.loc[mask, ['time_idx']].values)
    return preds

train['trend_pred'] = predict_trend_vectorized(train)
test['trend_pred'] = predict_trend_vectorized(test)
train['residual'] = train['log_price'] - train['trend_pred']

# ---- 2. RESIDUAL component: spatio-temporal ML model ----
cat_features = ['admin1', 'market', 'commodity', 'category']
num_features = ['latitude', 'longitude', 'month_sin', 'month_cos']
features = cat_features + num_features

X_train = train[features].copy()
X_test = test[features].copy()
for c in cat_features:
    X_train[c] = X_train[c].astype('category')
    X_test[c] = pd.Categorical(X_test[c], categories=X_train[c].cat.categories)

cat_idx = [X_train.columns.get_loc(c) for c in cat_features]

residual_model = HistGradientBoostingRegressor(
    max_iter=300, max_depth=6, learning_rate=0.05,
    categorical_features=cat_idx, random_state=42
)
residual_model.fit(X_train, train['residual'])

train['residual_pred'] = residual_model.predict(X_train)
test['residual_pred'] = residual_model.predict(X_test)

# ---- Final hybrid prediction ----
train['log_price_pred'] = train['trend_pred'] + train['residual_pred']
test['log_price_pred'] = test['trend_pred'] + test['residual_pred']
train['price_pred'] = np.exp(train['log_price_pred'])
test['price_pred'] = np.exp(test['log_price_pred'])

# ---- Evaluation (drop rows with no trend, e.g. Salt with no post-2020 data won't appear anyway) ----
test_eval = test.dropna(subset=['price_pred'])

def metrics(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"{label}: RMSE={rmse:.2f} PKR | MAE={mae:.2f} PKR | MAPE={mape:.1f}% | R2={r2:.3f}")
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}

overall = metrics(test_eval['price'], test_eval['price_pred'], "TEST (2020-2022, held-out future)")
train_metrics = metrics(train.dropna(subset=['price_pred'])['price'],
                         train.dropna(subset=['price_pred'])['price_pred'], "TRAIN (in-sample)")

per_commodity = {}
for c in test_eval['commodity'].unique():
    sub = test_eval[test_eval['commodity'] == c]
    if len(sub) < 3:
        continue
    per_commodity[c] = metrics(sub['price'], sub['price_pred'], f"  {c}")

# ---- Save everything ----
joblib.dump({"trend_models": trend_models, "residual_model": residual_model,
             "cat_features": cat_features, "num_features": num_features,
             "date_origin": df['date'].min()}, REPO_ROOT / 'models' / 'hybrid_price_model.joblib')

test_eval[['date','admin1','market','commodity','category','price','price_pred']].to_csv(
    REPO_ROOT / 'results' / 'test_predictions.csv', index=False)

with open(REPO_ROOT / 'results' / 'model_metrics.json', 'w') as f:
    json.dump({"overall_test": overall, "train_in_sample": train_metrics,
                "per_commodity_test": per_commodity,
                "cutoff_date": str(CUTOFF), "n_train": len(train), "n_test": len(test)}, f, indent=2, default=str)

print("\nSaved model, predictions, and metrics.")
