"""
preprocess.py
--------------
Loads raw retail CSV, runs quality checks, cleans data,
and saves a processed version ready for EDA and modelling.
"""

import pandas as pd
import numpy as np
import os

RAW_PATH       = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "retail_timeseries.csv")
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "retail_clean.csv")


def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    """Load raw CSV and parse date column."""
    path = os.path.abspath(path)
    print(f"📂 Loading data from: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"   Shape: {df.shape}  |  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    return df


def run_quality_checks(df: pd.DataFrame) -> dict:
    """Run basic data integrity checks and return a report dict."""
    print("\n🔍 Running quality checks …")

    required_cols = {"store_id", "item_id", "date", "qty_sold"}
    missing_cols  = required_cols - set(df.columns)
    duplicates    = df.duplicated(["store_id", "item_id", "date"]).sum()
    missing_qty   = df["qty_sold"].isna().sum()
    neg_qty       = (df["qty_sold"] < 0).sum()
    pct_stockout  = df["stockout_flag"].mean() * 100 if "stockout_flag" in df.columns else None

    report = {
        "total_rows":       len(df),
        "missing_columns":  list(missing_cols),
        "duplicate_rows":   int(duplicates),
        "missing_qty":      int(missing_qty),
        "negative_qty":     int(neg_qty),
        "pct_stockout":     round(pct_stockout, 2) if pct_stockout is not None else "N/A",
    }

    for key, val in report.items():
        status = "✅" if (val == 0 or val == [] or key == "total_rows" or key == "pct_stockout") else "⚠️"
        print(f"   {status}  {key}: {val}")

    if missing_cols:
        raise ValueError(f"Required columns missing: {missing_cols}")

    return report


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps:
    1. Remove duplicate rows
    2. Drop rows with missing qty_sold
    3. Clip negative qty to 0
    4. Remove stockout-censored rows (demand was suppressed by 0 stock)
    5. Fill missing promo/price/discount columns
    6. Sort by store, item, date
    """
    print("\n🧹 Cleaning data …")
    original_len = len(df)

    # 1. Duplicates
    df = df.drop_duplicates(["store_id", "item_id", "date"])
    print(f"   After removing duplicates: {len(df):,} rows")

    # 2. Missing qty
    df = df.dropna(subset=["qty_sold"])

    # 3. Negative qty → 0
    df["qty_sold"] = df["qty_sold"].clip(lower=0)

    # 4. Stockout censoring – remove rows where stock was 0 (demand is unknown)
    if "stockout_flag" in df.columns:
        before = len(df)
        df = df[df["stockout_flag"] == 0].copy()
        print(f"   Removed {before - len(df):,} stockout-censored rows")

    # 5. Fill optional columns
    for col in ["on_promo", "discount_pct", "holiday_flag", "festival_flag"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    if "price" in df.columns:
        df["price"] = df.groupby("item_id")["price"].transform(lambda s: s.fillna(s.median()))

    # 6. Sort
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    removed = original_len - len(df)
    print(f"   ✅ Final clean dataset: {len(df):,} rows  (removed {removed:,} rows total)")
    return df


def compute_weekly_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Also create a weekly aggregated version (useful for less volatile forecasting).
    Week start = Monday.
    """
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="D")
    agg_cols = {
        "qty_sold":    "sum",
        "price":       "mean",
        "on_promo":    "max",
        "discount_pct":"mean",
        "holiday_flag":"max",
        "festival_flag":"max",
        "unit_cost":   "first",
        "supplier_lead_time_days": "first",
        "ordering_cost":"first",
        "holding_cost_rate": "first",
    }
    # keep only columns present
    agg_cols = {k: v for k, v in agg_cols.items() if k in df.columns}
    weekly = (
        df.groupby(["store_id", "item_id", "category", "product_name", "week_start"])
          .agg(agg_cols)
          .reset_index()
          .rename(columns={"week_start": "date"})
    )
    return weekly


def save_data(df: pd.DataFrame, path: str = PROCESSED_PATH):
    path = os.path.abspath(path)
    df.to_csv(path, index=False)
    print(f"\n💾 Processed data saved to: {path}")


def preprocess_pipeline(raw_path: str = RAW_PATH, processed_path: str = PROCESSED_PATH):
    df     = load_data(raw_path)
    report = run_quality_checks(df)
    df     = clean_data(df)

    weekly = compute_weekly_aggregation(df)
    weekly_path = processed_path.replace("retail_clean.csv", "retail_weekly.csv")
    weekly_path = os.path.abspath(weekly_path)
    weekly.to_csv(weekly_path, index=False)
    print(f"💾 Weekly aggregation saved to: {weekly_path}")

    save_data(df, processed_path)
    return df, report


if __name__ == "__main__":
    df, report = preprocess_pipeline()
    print("\n📋 Sample cleaned data:")
    print(df.head(5).to_string())
    print(f"\n📊 Quality report: {report}")
