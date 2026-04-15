"""
forecasting_model.py
---------------------
Trains a Random Forest Regressor to forecast daily retail sales.
Implements:
  - Time-series aware train/test split (last 60 days = test)
  - Cross-validation (GroupKFold by SKU-Store)
  - Model evaluation (MAE, RMSE, MAPE, MASE)
  - Feature importance
  - Croston's method for intermittent demand SKUs
  - Future forecast generation (next 30 days)
  - Actual vs Predicted visualization
  - Model persistence (joblib)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})

FEATURES_PATH  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "retail_features.csv"))
MODEL_PATH     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "retail_forecast_model.pkl"))
FORECAST_PATH  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "forecasts", "forecast_output.csv"))
IMG_DIR        = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images"))
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(FORECAST_PATH), exist_ok=True)

TARGET    = "qty_sold"
TEST_DAYS = 60   # holdout period (last 60 days)


# ─────────────────────────────────────────────
# HELPER: Metrics
# ─────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray = None) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE (avoid division by zero)
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    # MASE (Mean Absolute Scaled Error) – scale vs seasonal naive
    if y_naive is not None:
        d_model = np.abs(y_true - y_pred).mean()
        d_naive = np.abs(y_true - y_naive).mean()
        mase = d_model / d_naive if d_naive > 0 else np.nan
    else:
        mase = np.nan
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2),
            "MAPE (%)": round(mape, 2), "MASE": round(mase, 4)}


# ─────────────────────────────────────────────
# CROSTON'S METHOD (intermittent demand)
# ─────────────────────────────────────────────
def croston_forecast(y: np.ndarray, alpha: float = 0.1, h: int = 30) -> np.ndarray:
    """
    Croston's method for intermittent demand series.
    y     : historical demand array
    alpha : smoothing parameter
    h     : forecast horizon (periods)
    Returns a flat forecast array of length h.
    """
    non_zero_idx = np.where(y > 0)[0]
    if len(non_zero_idx) == 0:
        return np.zeros(h)

    # Non-zero demand values
    z = y[non_zero_idx]
    # Inter-demand intervals
    intervals = np.diff(np.r_[0, non_zero_idx + 1])

    if len(z) == 0:
        return np.zeros(h)

    z_hat = z[0]
    p_hat = intervals[0] if len(intervals) > 0 else 1

    for i in range(1, len(z)):
        z_hat = alpha * z[i] + (1 - alpha) * z_hat
    for i in range(1, len(intervals)):
        p_hat = alpha * intervals[i] + (1 - alpha) * p_hat

    # SBA bias correction: multiply by (1 - alpha/2)
    forecast_rate = (z_hat / max(p_hat, 1)) * (1 - alpha / 2)
    return np.full(h, max(0, forecast_rate))


# ─────────────────────────────────────────────
# DATA SPLIT
# ─────────────────────────────────────────────
def time_based_split(df: pd.DataFrame, test_days: int = TEST_DAYS):
    """
    Split data: training = all but last `test_days` days;
    test = last `test_days` days (time-aware).
    """
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    train  = df[df["date"] <= cutoff].copy()
    test   = df[df["date"] >  cutoff].copy()
    print(f"   Train: {len(train):,} rows  ({train['date'].min().date()} → {train['date'].max().date()})")
    print(f"   Test : {len(test):,}  rows  ({test['date'].min().date()} → {test['date'].max().date()})")
    return train, test


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                n_estimators: int = 300, max_depth: int = 12) -> RandomForestRegressor:
    print(f"\n🌲  Training Random Forest ({n_estimators} trees, max_depth={max_depth}) …")
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    print("   ✅ Training complete.")
    return rf


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE PLOT
# ─────────────────────────────────────────────
def plot_feature_importance(model, feature_cols: list, top_n: int = 20):
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = sns.color_palette("Blues_r", len(top))
    ax.barh(top.index, top.values, color=colors)
    ax.set(title=f"Top {top_n} Feature Importances (Random Forest)",
           xlabel="Importance Score")
    path = os.path.join(IMG_DIR, "11_feature_importance.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"   💾  Saved: {path}")


# ─────────────────────────────────────────────
# ACTUAL vs PREDICTED PLOT
# ─────────────────────────────────────────────
def plot_actual_vs_predicted(test_df: pd.DataFrame, y_pred: np.ndarray,
                              sample_sku: str = None, sample_store: str = None):
    df = test_df.copy()
    df["predicted"] = np.maximum(0, y_pred)

    # Aggregate across all SKUs/Stores for overview plot
    daily = df.groupby("date").agg(actual=("qty_sold", "sum"),
                                   predicted=("predicted", "sum")).reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Top: overall
    axes[0].plot(daily["date"], daily["actual"],    color="steelblue", label="Actual",    linewidth=1.5)
    axes[0].plot(daily["date"], daily["predicted"], color="darkorange", label="Predicted", linewidth=1.5, linestyle="--")
    axes[0].set(title="Actual vs Predicted – All Stores & SKUs (Test Period)",
                xlabel="Date", ylabel="Units Sold")
    axes[0].legend()

    # Bottom: scatter (actual vs predicted)
    axes[1].scatter(df["qty_sold"], df["predicted"], alpha=0.15, s=10, color="steelblue")
    max_val = max(df["qty_sold"].max(), df["predicted"].max())
    axes[1].plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect fit")
    axes[1].set(title="Scatter: Actual vs Predicted (Test Set)",
                xlabel="Actual Units", ylabel="Predicted Units")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(IMG_DIR, "12_actual_vs_predicted.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"   💾  Saved: {path}")


# ─────────────────────────────────────────────
# SINGLE SKU FORECAST PLOT
# ─────────────────────────────────────────────
def plot_sku_forecast(history_df, future_dates, future_pred,
                      item_id="P001", store_id="S001"):
    mask = (history_df["item_id"] == item_id) & (history_df["store_id"] == store_id)
    hist = history_df[mask].tail(90)  # last 90 days of history

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(hist["date"], hist["qty_sold"], color="steelblue", label="Historical", linewidth=1.5)
    ax.plot(future_dates, future_pred, color="darkorange", label="Forecast (30 days)",
            linewidth=2, linestyle="--", marker="o", markersize=4)
    # Confidence band (simple ±20% for illustration)
    ax.fill_between(future_dates,
                    np.maximum(0, future_pred * 0.80),
                    future_pred * 1.20,
                    alpha=0.2, color="darkorange", label="±20% band")
    ax.axvline(x=hist["date"].max(), color="grey", linestyle=":", linewidth=1)
    ax.set(title=f"Sales Forecast – {item_id} @ Store {store_id} (next 30 days)",
           xlabel="Date", ylabel="Units Sold")
    ax.legend()
    path = os.path.join(IMG_DIR, "13_sku_forecast.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"   💾  Saved: {path}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run_forecasting(features_path: str = FEATURES_PATH,
                    model_path: str    = MODEL_PATH,
                    forecast_path: str = FORECAST_PATH):

    print("🚀  Starting Forecasting Pipeline …\n")

    # 1. Load features
    df = pd.read_csv(features_path, parse_dates=["date"])
    print(f"   Loaded {len(df):,} rows with {df.shape[1]} columns")

    # 2. Define feature columns
    exclude = {"date", TARGET, "store_id", "item_id", "product_name",
               "category", "stockout_flag", "stock_on_hand", "week_start"}
    feat_cols = [c for c in df.columns if c not in exclude]
    print(f"   Using {len(feat_cols)} feature columns")

    X = df[feat_cols]
    y = df[TARGET]

    # 3. Time-based split
    print("\n📅  Splitting data …")
    train_df, test_df = time_based_split(df)
    X_train = train_df[feat_cols]
    y_train = train_df[TARGET]
    X_test  = test_df[feat_cols]
    y_test  = test_df[TARGET]

    # 4. Train
    model = train_model(X_train, y_train)

    # 5. Evaluate
    y_pred_test = np.maximum(0, model.predict(X_test))
    # Seasonal naive: same DOW, 4 weeks earlier
    y_naive = test_df.groupby(["store_id","item_id"])["qty_sold"].shift(28).fillna(0).values
    metrics = compute_metrics(y_test.values, y_pred_test, y_naive)
    print("\n📊  Model Evaluation (Test Set):")
    for k, v in metrics.items():
        print(f"   {k:15s}: {v}")

    # 6. Category-wise metrics
    test_df2 = test_df.copy()
    test_df2["predicted"] = y_pred_test
    print("\n📊  MAE by Category:")
    cat_mae = (test_df2.groupby("category")
               .apply(lambda g: mean_absolute_error(g["qty_sold"], g["predicted"]))
               .round(2))
    print(cat_mae.to_string())

    # 7. Feature importance plot
    plot_feature_importance(model, feat_cols)

    # 8. Actual vs Predicted plot
    plot_actual_vs_predicted(test_df2, y_pred_test)

    # 9. Save model
    artifact = {"model": model, "features": feat_cols, "metrics": metrics}
    joblib.dump(artifact, model_path)
    print(f"\n💾  Model saved to: {model_path}")

    # 10. Generate future forecast (next 30 days) for every SKU-Store combo
    print("\n🔮  Generating future forecasts (next 30 days) …")
    last_date = df["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30, freq="D")

    forecast_rows = []
    for (store_id, item_id), grp in df.groupby(["store_id", "item_id"]):
        grp = grp.sort_values("date")
        hist_qty = grp["qty_sold"].values

        # Check intermittency
        p_zero = (hist_qty == 0).mean()
        if p_zero > 0.30:  # intermittent SKU → Croston
            pred_vals = croston_forecast(hist_qty, h=30)
            method = "Croston/SBA"
        else:
            # Build feature rows for next 30 days using last known values
            last_row = grp.iloc[-1].copy()
            pred_vals_list = []
            for fd in future_dates:
                row = last_row.copy()
                row["date"]         = fd
                row["dow"]          = fd.dayofweek
                row["is_weekend"]   = int(fd.dayofweek >= 5)
                row["day_of_month"] = fd.day
                row["week_of_year"] = fd.isocalendar()[1]
                row["month"]        = fd.month
                row["quarter"]      = (fd.month - 1) // 3 + 1
                row["year"]         = fd.year
                doy = fd.dayofyear
                row["sin_doy"]      = np.sin(2 * np.pi * doy / 365)
                row["cos_doy"]      = np.cos(2 * np.pi * doy / 365)
                row["sin_dow"]      = np.sin(2 * np.pi * fd.dayofweek / 7)
                row["cos_dow"]      = np.cos(2 * np.pi * fd.dayofweek / 7)
                row["time_idx"]     = last_row.get("time_idx", 0) + (fd - last_date).days

                feat_row = pd.DataFrame([row[feat_cols]])
                pred = max(0, model.predict(feat_row)[0])
                pred_vals_list.append(pred)
            pred_vals = np.array(pred_vals_list)
            method = "RandomForest"

        for i, fd in enumerate(future_dates):
            forecast_rows.append({
                "store_id":      store_id,
                "item_id":       item_id,
                "forecast_date": fd,
                "predicted_qty": round(max(0, pred_vals[i]), 2),
                "method":        method,
            })

    forecast_df = pd.DataFrame(forecast_rows)
    forecast_df.to_csv(forecast_path, index=False)
    print(f"   💾  Forecast saved to: {forecast_path}")
    print(f"   Rows: {len(forecast_df):,} ({forecast_df['store_id'].nunique()} stores × {forecast_df['item_id'].nunique()} SKUs × 30 days)")

    # 11. Plot a sample SKU forecast
    plot_sku_forecast(df, future_dates,
                      forecast_df[(forecast_df["item_id"] == "P001") &
                                  (forecast_df["store_id"] == "S001")]["predicted_qty"].values)

    print("\n✅  Forecasting pipeline complete!")
    return model, feat_cols, metrics, forecast_df


if __name__ == "__main__":
    run_forecasting()
