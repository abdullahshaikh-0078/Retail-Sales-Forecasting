"""
main.py
--------
Master pipeline runner for the Retail Sales Forecasting &
Inventory Optimization project.

Run this single file to execute the full end-to-end pipeline:
  Step 1 → Generate synthetic dataset
  Step 2 → Preprocess & clean data
  Step 3 → Exploratory Data Analysis (EDA)
  Step 4 → Feature Engineering
  Step 5 → Train forecasting model + generate forecasts
  Step 6 → Inventory optimization (SS, ROP, EOQ)
  Step 7 → Business insights & reports

Usage:
  python main.py                     # full pipeline
  python main.py --skip-data-gen     # skip dataset generation (use existing)
  python main.py --steps 1,2,3       # run specific steps only
"""

import sys
import os
import time
import argparse

# Ensure src/ is on the Python path (works from project root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from generate_dataset    import build_dataset
from preprocess          import preprocess_pipeline
from eda                 import run_eda
from feature_engineering import engineer_features
from forecasting_model   import run_forecasting
from inventory_optimization import run_inventory_optimization
from business_insights   import run_business_insights

# ── Paths ─────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(__file__)
RAW_CSV   = os.path.join(BASE, "data", "raw",       "retail_timeseries.csv")
CLEAN_CSV = os.path.join(BASE, "data", "processed", "retail_clean.csv")


def banner(step: int, title: str):
    print("\n" + "=" * 65)
    print(f"  STEP {step}: {title}")
    print("=" * 65)


def run_pipeline(steps: list = None, skip_data_gen: bool = False):
    all_steps = list(range(1, 8))
    if steps:
        run_steps = [int(s) for s in steps]
    else:
        run_steps = all_steps

    t_start = time.time()
    results  = {}

    # ── Step 1: Dataset Generation ─────────────────────────────────────
    if 1 in run_steps and not skip_data_gen:
        banner(1, "Dataset Generation")
        df_raw = build_dataset()
        os.makedirs(os.path.join(BASE, "data", "raw"), exist_ok=True)
        df_raw.to_csv(RAW_CSV, index=False)
        print(f"\n✅  Raw dataset saved → {RAW_CSV}")
        results["step1"] = "Done"
    elif skip_data_gen or 1 not in run_steps:
        print(f"\n⏭️   Skipping Step 1 (dataset generation) — using existing: {RAW_CSV}")

    # ── Step 2: Preprocessing ──────────────────────────────────────────
    if 2 in run_steps:
        banner(2, "Data Preprocessing & Quality Checks")
        clean_df, report = preprocess_pipeline(RAW_CSV, CLEAN_CSV)
        results["step2"] = report
        print(f"\n✅  Clean dataset saved → {CLEAN_CSV}")

    # ── Step 3: EDA ────────────────────────────────────────────────────
    if 3 in run_steps:
        banner(3, "Exploratory Data Analysis (EDA)")
        run_eda(CLEAN_CSV)
        results["step3"] = "10 charts generated"

    # ── Step 4: Feature Engineering ────────────────────────────────────
    if 4 in run_steps:
        banner(4, "Feature Engineering")
        feat_df = engineer_features(CLEAN_CSV)
        results["step4"] = f"{feat_df.shape[1]} columns"

    # ── Step 5: Forecasting Model ──────────────────────────────────────
    if 5 in run_steps:
        banner(5, "Forecasting Model Training & Prediction")
        model, feat_cols, metrics, forecast_df = run_forecasting()
        results["step5"] = metrics

    # ── Step 6: Inventory Optimization ────────────────────────────────
    if 6 in run_steps:
        banner(6, "Inventory Optimization (SS · ROP · EOQ)")
        inv_df, alerts = run_inventory_optimization()
        results["step6"] = {
            "total_combos": len(inv_df),
            "reorder_alerts": int(inv_df["reorder_alert"].sum()),
        }

    # ── Step 7: Business Insights & Reporting ─────────────────────────
    if 7 in run_steps:
        banner(7, "Business Insights & Reports")
        kpis = run_business_insights()
        results["step7"] = kpis

    # ── Final summary ──────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 65)
    print(f"  🎉  PIPELINE COMPLETE  —  Total time: {elapsed:.1f}s")
    print("=" * 65)
    print("\n📂  Output files:")
    print(f"   data/processed/retail_clean.csv        — cleaned dataset")
    print(f"   data/processed/retail_features.csv     — ML feature matrix")
    print(f"   models/retail_forecast_model.pkl        — trained RF model")
    print(f"   outputs/forecasts/forecast_output.csv   — 30-day forecast")
    print(f"   outputs/inventory/inventory_policy_table.csv")
    print(f"   outputs/inventory/reorder_alerts.csv")
    print(f"   outputs/reports/business_report.html")
    print(f"   outputs/reports/kpi_summary.csv")
    print(f"   images/  ← 18 charts & visualizations")
    print("\n💡  To launch the Streamlit dashboard, run:")
    print("   streamlit run app/streamlit_app.py\n")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Retail Sales Forecasting Pipeline")
    parser.add_argument("--skip-data-gen", action="store_true",
                        help="Skip dataset generation (use existing CSV)")
    parser.add_argument("--steps", type=str, default=None,
                        help="Comma-separated steps to run, e.g. --steps 1,2,3")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    steps = args.steps.split(",") if args.steps else None
    run_pipeline(steps=steps, skip_data_gen=args.skip_data_gen)
