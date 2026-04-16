"""
test_pipeline.py
-----------------
Basic unit tests for the Retail Sales Forecasting project.
Run with: python -m pytest tests/test_pipeline.py -v
"""

import sys, os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── test: dataset generation ──────────────────────────────────────────────
class TestDatasetGeneration:
    def test_build_dataset_shape(self):
        from generate_dataset import build_dataset
        df = build_dataset()
        assert len(df) == 21900, "Expected 3 stores × 10 SKUs × 730 days"
        assert df.shape[1] == 19, "Expected 19 columns"

    def test_no_negative_qty(self):
        from generate_dataset import build_dataset
        df = build_dataset()
        assert (df["qty_sold"] < 0).sum() == 0, "qty_sold must not be negative"

    def test_required_columns_present(self):
        from generate_dataset import build_dataset
        df = build_dataset()
        required = {"date", "store_id", "item_id", "qty_sold", "price", "on_promo"}
        assert required.issubset(set(df.columns))

    def test_date_range(self):
        from generate_dataset import build_dataset
        df = build_dataset()
        assert df["date"].min() == pd.Timestamp("2022-01-01")
        assert df["date"].max() == pd.Timestamp("2023-12-31")


# ── test: preprocessing ───────────────────────────────────────────────────
class TestPreprocessing:
    def test_no_duplicates_after_clean(self):
        from generate_dataset import build_dataset
        from preprocess import clean_data
        df = build_dataset()
        clean = clean_data(df)
        dups = clean.duplicated(["store_id", "item_id", "date"]).sum()
        assert dups == 0, "No duplicates should remain after cleaning"

    def test_no_negative_qty_after_clean(self):
        from generate_dataset import build_dataset
        from preprocess import clean_data
        df = build_dataset()
        clean = clean_data(df)
        assert (clean["qty_sold"] < 0).sum() == 0

    def test_quality_checks_pass(self):
        from generate_dataset import build_dataset
        from preprocess import run_quality_checks
        df = build_dataset()
        report = run_quality_checks(df)
        assert report["duplicate_rows"] == 0
        assert report["missing_qty"] == 0
        assert report["negative_qty"] == 0


# ── test: feature engineering ─────────────────────────────────────────────
class TestFeatureEngineering:
    @pytest.fixture
    def sample_df(self):
        from generate_dataset import build_dataset
        from preprocess import clean_data
        df = build_dataset()
        return clean_data(df)

    def test_feature_count(self, sample_df):
        from feature_engineering import engineer_features
        import tempfile
        # Save temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        sample_df.to_csv(tmp.name, index=False)
        feat_df = engineer_features(tmp.name, tmp.name.replace(".csv", "_feat.csv"))
        assert feat_df.shape[1] > 30, "Should have > 30 columns after feature engineering"

    def test_no_nan_after_engineering(self, sample_df):
        from feature_engineering import engineer_features
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        sample_df.to_csv(tmp.name, index=False)
        feat_df = engineer_features(tmp.name, tmp.name.replace(".csv", "_feat.csv"))
        assert feat_df.isnull().sum().sum() == 0, "No NaN should remain after engineering"


# ── test: inventory optimization ─────────────────────────────────────────
class TestInventoryOptimization:
    def test_compute_inventory_policy_basic(self):
        from inventory_optimization import compute_inventory_policy
        forecast = np.array([20.0] * 30)
        result = compute_inventory_policy(
            forecast_h    = forecast,
            resid_std     = 5.0,
            on_hand       = 50.0,
            lead_time     = 7,
            service       = 0.95,
            annual_demand = 7300,
            ordering_cost = 500,
            unit_cost     = 100,
            holding_rate  = 0.20,
        )
        assert result["safety_stock"] > 0, "Safety stock must be positive"
        assert result["reorder_point"] > result["safety_stock"], "ROP = mu_L + SS > SS"
        assert result["EOQ"] > 0, "EOQ must be positive"

    def test_safety_stock_increases_with_service_level(self):
        from inventory_optimization import compute_inventory_policy
        forecast = np.array([20.0] * 30)
        r90 = compute_inventory_policy(forecast, 5.0, 50.0, 7, service=0.90,
                                       annual_demand=7300, ordering_cost=500,
                                       unit_cost=100, holding_rate=0.20)
        r99 = compute_inventory_policy(forecast, 5.0, 50.0, 7, service=0.99,
                                       annual_demand=7300, ordering_cost=500,
                                       unit_cost=100, holding_rate=0.20)
        assert r99["safety_stock"] > r90["safety_stock"], \
            "Higher service level should require more safety stock"

    def test_reorder_alert_when_stock_below_rop(self):
        from inventory_optimization import compute_inventory_policy
        forecast = np.array([20.0] * 30)
        # on_hand = 1 (very low) → must trigger alert
        result = compute_inventory_policy(forecast, 5.0, on_hand=1.0,
                                          lead_time=7, service=0.95,
                                          annual_demand=7300, ordering_cost=500,
                                          unit_cost=100, holding_rate=0.20)
        assert result["reorder_alert"] is True


# ── test: Croston's method ─────────────────────────────────────────────────
class TestCroston:
    def test_all_zeros(self):
        from forecasting_model import croston_forecast
        y = np.zeros(100)
        result = croston_forecast(y, h=10)
        assert np.all(result == 0), "All-zero series should give zero forecast"

    def test_output_length(self):
        from forecasting_model import croston_forecast
        y = np.array([0, 0, 5, 0, 0, 3, 0, 8, 0, 0])
        result = croston_forecast(y, h=14)
        assert len(result) == 14, "Forecast should have length h=14"

    def test_positive_forecast(self):
        from forecasting_model import croston_forecast
        y = np.array([0, 0, 5, 0, 0, 3, 0, 8, 0, 0] * 10)
        result = croston_forecast(y, h=7)
        assert np.all(result >= 0), "Forecast values must be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
