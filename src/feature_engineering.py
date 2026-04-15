"""
feature_engineering.py
------------------------
Creates ML-ready features from the cleaned retail dataset.
Features include:
  - Lag features (recency signals)
  - Rolling statistics (trend + volatility)
  - Calendar features (day, week, month, DOW)
  - Promotional features
  - Price/discount features
  - Festival / holiday features
"""

import pandas as pd
import numpy as np
import os

PROCESSED_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "processed", "retail_clean.csv")
)
FEATURES_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "processed", "retail_features.csv")
)


def make_lag_features(df: pd.DataFrame,
                      lags: tuple = (1, 2, 3, 7, 14),
                      target: str = "qty_sold") -> pd.DataFrame:
    """Create lag features for the target variable per SKU-Store group."""
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(["store_id", "item_id"])[target].shift(lag)
    return df


def make_rolling_features(df: pd.DataFrame,
                           windows: tuple = (7, 14, 28),
                           target: str = "qty_sold") -> pd.DataFrame:
    """Create rolling mean, std, min, max features per SKU-Store group (lag-1 to avoid leakage)."""
    for w in windows:
        # shift(1) ensures we don't include current row (no data leakage)
        rolled = df.groupby(["store_id", "item_id"])[target].shift(1).rolling(w)
        df[f"rollmean_{w}"] = rolled.mean().values
        df[f"rollstd_{w}"]  = rolled.std().values
        df[f"rollmin_{w}"]  = rolled.min().values
        df[f"rollmax_{w}"]  = rolled.max().values
    return df


def make_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar-based features from the date column."""
    df["dow"]         = df["date"].dt.dayofweek           # 0=Mon, 6=Sun
    df["is_weekend"]  = (df["dow"] >= 5).astype(int)
    df["day_of_month"]= df["date"].dt.day
    df["week_of_year"]= df["date"].dt.isocalendar().week.astype(int)
    df["month"]       = df["date"].dt.month
    df["quarter"]     = df["date"].dt.quarter
    df["year"]        = df["date"].dt.year
    # Fourier terms for annual seasonality (captures sine/cosine waves)
    doy = df["date"].dt.dayofyear
    df["sin_doy"]     = np.sin(2 * np.pi * doy / 365)
    df["cos_doy"]     = np.cos(2 * np.pi * doy / 365)
    df["sin_dow"]     = np.sin(2 * np.pi * df["dow"] / 7)
    df["cos_dow"]     = np.cos(2 * np.pi * df["dow"] / 7)
    return df


def make_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Price & discount features."""
    if "price" in df.columns and "unit_cost" in df.columns:
        df["gross_margin_pct"] = (df["price"] - df["unit_cost"]) / df["price"]
    if "discount_pct" in df.columns:
        df["is_discounted"] = (df["discount_pct"] > 0).astype(int)
        # Price relative to category median
        cat_med = df.groupby("category")["price"].transform("median")
        df["price_vs_cat_median"] = df["price"] / (cat_med + 1e-9)
    return df


def make_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling promo frequency (past 7 & 14 days) to capture promotion fatigue."""
    if "on_promo" in df.columns:
        df["promo_freq_7d"]  = (df.groupby(["store_id","item_id"])["on_promo"]
                                  .shift(1).rolling(7).mean().values)
        df["promo_freq_14d"] = (df.groupby(["store_id","item_id"])["on_promo"]
                                  .shift(1).rolling(14).mean().values)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical columns (store, item, category)."""
    for col in ["store_id", "item_id", "category"]:
        if col in df.columns:
            df[col + "_enc"] = df[col].astype("category").cat.codes
    return df


def add_trend_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple linear time index (days since first date) per SKU-Store.
    Captures long-run demand trend.
    """
    df["time_idx"] = (df.groupby(["store_id","item_id"])
                        .cumcount())  # 0, 1, 2, ... per group
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the final list of feature columns (exclude identifiers and target)."""
    exclude = {"date", "qty_sold", "store_id", "item_id",
               "product_name", "category", "stockout_flag",
               "stock_on_hand", "week_start"}
    return [c for c in df.columns if c not in exclude]


def engineer_features(data_path: str = PROCESSED_PATH,
                      save_path: str  = FEATURES_PATH) -> pd.DataFrame:
    print("⚙️  Starting feature engineering pipeline …")

    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    print(f"   Input shape: {df.shape}")

    # Apply all feature transformations
    df = make_calendar_features(df)
    df = make_lag_features(df)
    df = make_rolling_features(df)
    df = make_price_features(df)
    df = make_promo_features(df)
    df = encode_categoricals(df)
    df = add_trend_feature(df)

    # Drop rows with NaN from lags/rolling (cold-start rows)
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = before - len(df)
    print(f"   Dropped {dropped:,} cold-start rows (NaN from lags/rolling)")
    print(f"   Output shape: {df.shape}")

    feat_cols = get_feature_columns(df)
    print(f"   Feature columns ({len(feat_cols)}): {feat_cols[:8]} … + {len(feat_cols)-8} more")

    save_path = os.path.abspath(save_path)
    df.to_csv(save_path, index=False)
    print(f"   💾 Features saved to: {save_path}")
    return df


if __name__ == "__main__":
    df = engineer_features()
    feat_cols = get_feature_columns(df)
    print(f"\nFinal feature list ({len(feat_cols)}):")
    for f in feat_cols:
        print(f"  • {f}")
