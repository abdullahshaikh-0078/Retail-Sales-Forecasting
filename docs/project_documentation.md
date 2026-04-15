# Technical Documentation
## Retail Sales Forecasting & Inventory Optimization System

---

## 1. System Overview

This document provides complete technical documentation for the Retail Sales Forecasting & Inventory Optimization system — a Python-based data science pipeline built for student portfolio purposes.

---

## 2. Module Descriptions

### `src/generate_dataset.py`
Generates 21,900 rows of synthetic retail data across 3 stores, 10 SKUs, and 730 days.

**Simulation components:**
- Poisson demand with store footfall multiplier
- Weekly seasonality (Saturday/Sunday spike)
- Annual seasonality via sine wave
- Festival lift (1.6×) for Diwali/Navratri windows
- Promotional lift proportional to discount percentage
- Gentle upward trend (0.03% per day)

### `src/preprocess.py`
Data quality pipeline:
- Duplicate detection and removal
- Missing value handling
- Negative quantity clipping
- Stockout-censored row filtering
- Weekly aggregation generation

### `src/eda.py`
Generates 10 saved visualization charts:

| Chart | Description |
|-------|-------------|
| `01_overall_sales_trend.png` | Daily total + 7-day moving average |
| `02_category_sales.png` | Horizontal bar — total units by category |
| `03_store_comparison.png` | Monthly lines per store |
| `04_dow_pattern.png` | Day-of-week average sales |
| `05_promo_lift.png` | Side-by-side promo vs non-promo + lift % |
| `06_top_skus_revenue.png` | Top 10 SKUs by revenue |
| `07_intermittency.png` | Zero-demand share per SKU |
| `08_correlation_heatmap.png` | Correlation matrix of numeric variables |
| `09_revenue_heatmap.png` | Category × Month revenue heatmap |
| `10_sales_distribution.png` | Boxplot of daily sales by category |

### `src/feature_engineering.py`
Creates 48 ML features from the cleaned dataset:

**Feature groups:**
1. **Lag features** — `lag_1`, `lag_2`, `lag_3`, `lag_7`, `lag_14` (recency signals)
2. **Rolling mean** — `rollmean_7`, `rollmean_14`, `rollmean_28` (trend smoothing)
3. **Rolling std** — `rollstd_7`, `rollstd_14`, `rollstd_28` (volatility)
4. **Rolling min/max** — bounds of recent demand
5. **Calendar** — `dow`, `is_weekend`, `day_of_month`, `week_of_year`, `month`, `quarter`, `year`
6. **Fourier terms** — `sin_doy`, `cos_doy`, `sin_dow`, `cos_dow` (smooth seasonality)
7. **Price** — `price`, `discount_pct`, `gross_margin_pct`, `price_vs_cat_median`
8. **Promo** — `on_promo`, `promo_freq_7d`, `promo_freq_14d`
9. **Supply** — `supplier_lead_time_days`, `ordering_cost`, `holding_cost_rate`
10. **Categorical encodings** — `store_id_enc`, `item_id_enc`, `category_enc`
11. **Trend** — `time_idx` (days since group start)

**Critical design note:** All rolling/lag features use `.shift(1)` before aggregation to prevent target leakage — the model never sees the current day's sales in its input features.

### `src/forecasting_model.py`
**Training strategy:**
- Time-based holdout split: last 60 days = test set
- Random Forest Regressor: 300 trees, max depth 12, min_samples_leaf 5
- Predictions clipped to ≥ 0 (negative sales don't exist)

**Evaluation metrics:**
| Metric | Formula | What it means |
|--------|---------|---------------|
| MAE | mean(|y - ŷ|) | Average error in units |
| RMSE | √mean((y-ŷ)²) | Penalises large errors more |
| MAPE | mean(|y-ŷ|/y) × 100 | % error (biased for near-zero) |
| MASE | MAE_model / MAE_naive | <1 = better than naive baseline |

**Intermittent demand (Croston/SBA):**
- Triggered when a SKU-Store has P_zero > 30%
- Separately smooths non-zero demand magnitudes (z) and inter-demand intervals (p)
- SBA correction: multiply by (1 − α/2) to reduce positive bias
- Returns flat forecast of estimated demand rate

### `src/inventory_optimization.py`
**Inventory policy computation:**

```
Lead Time Demand Mean:   μ_L = Σ forecast[t..t+L]
Lead Time Demand Std:    σ_L = resid_std × √L
Safety Stock:            SS  = z × σ_L
Reorder Point:           ROP = μ_L + SS
Annual Holding Cost:     H   = unit_cost × holding_rate
EOQ:                     EOQ = √(2 × D × K / H)
Order Quantity:          Q   = max(EOQ, ROP − OnHand)  if OnHand < ROP
Reorder Alert:           True if OnHand ≤ ROP
```

**Parameters used:**
- z = 1.645 (95% service level via `scipy.stats.norm.ppf(0.95)`)
- K = ₹500 per order (fixed ordering cost)
- holding_rate = 20% of unit cost per year
- lead_time = category-specific (1–10 days)

---

## 3. Data Flow Diagram

```
retail_timeseries.csv (raw)
        │
        ▼ preprocess.py
retail_clean.csv
        │
        ├──► eda.py ──────────────────────► images/01–10_*.png
        │
        ▼ feature_engineering.py
retail_features.csv (48 features)
        │
        ▼ forecasting_model.py
        ├──► retail_forecast_model.pkl
        ├──► forecast_output.csv (30-day predictions)
        └──► images/11–13_*.png
             │
             ▼ inventory_optimization.py
             ├──► inventory_policy_table.csv
             ├──► reorder_alerts.csv
             └──► images/14–17_*.png
                  │
                  ▼ business_insights.py
                  ├──► 18_executive_dashboard.png
                  ├──► business_report.html
                  └──► kpi_summary.csv
```

---

## 4. Inventory Theory Reference

### Safety Stock
Buffer stock held to protect against demand variability and supply uncertainty.

- Too little → stockouts, lost sales, unhappy customers
- Too much → excess holding cost, capital tied up, wastage risk

**Formula:** `SS = z × σ_L`
- z = service level factor (e.g., 1.645 for 95%)
- σ_L = demand standard deviation during lead time = σ_daily × √(lead_time)

### Reorder Point (ROP)
The stock level that triggers a new purchase order.

**Formula:** `ROP = μ_L + SS`
- μ_L = expected demand during lead time = Σ(forecast for next L days)
- When on-hand stock falls to or below ROP → place an order

### Economic Order Quantity (EOQ)
The optimal order size that minimises total annual ordering cost + holding cost.

**Formula:** `EOQ = √(2 × D × K / H)`
- D = annual demand (units/year)
- K = ordering cost per order placed (₹)
- H = holding cost per unit per year = unit_cost × holding_rate

**Assumptions of classic EOQ:**
- Constant demand rate
- Instantaneous replenishment
- No quantity discounts

In practice, EOQ serves as a starting point and is adjusted for real constraints.

---

## 5. Model Selection Rationale

| Model | Pros | Cons | Used For |
|-------|------|------|----------|
| Random Forest | Handles many features, robust, no stationarity needed | Slower than linear models, can't extrapolate | Regular SKUs |
| Croston/SBA | Designed for intermittent demand, unbiased | Assumes stationary demand rate | Sparse SKUs |
| Seasonal Naive | Zero-code baseline | Ignores features and trends | Benchmark comparison |

---

## 6. GitHub Commit Strategy

### Day-by-Day Plan

| Day | Action | Commit Message |
|-----|--------|---------------|
| Day 1 | Project structure, README skeleton | `feat: initialize project structure and README` |
| Day 2 | `generate_dataset.py` complete | `feat: add synthetic dataset generator with seasonality` |
| Day 3 | `preprocess.py` + quality checks | `feat: data preprocessing and quality validation module` |
| Day 4 | `eda.py` + 10 charts | `feat: EDA with 10 visualizations saved to images/` |
| Day 5 | `feature_engineering.py` | `feat: 48-feature engineering pipeline with lag/rolling/calendar` |
| Day 6 | `forecasting_model.py` | `feat: Random Forest forecaster + Croston + backtesting` |
| Day 7 | `inventory_optimization.py` | `feat: inventory policy engine (SS, ROP, EOQ, alerts)` |
| Day 8 | `business_insights.py` + dashboard | `feat: executive dashboard and HTML report generator` |
| Day 9 | `streamlit_app.py` | `feat: interactive Streamlit dashboard with 5 tabs` |
| Day 10 | Tests + docs + final polish | `docs: complete README, tests, and documentation` |

---

## 7. Troubleshooting Guide

| Problem | Likely Cause | Solution |
|---------|-------------|----------|
| `ModuleNotFoundError: pandas` | Package not installed | `pip install -r requirements.txt` |
| `FileNotFoundError: retail_timeseries.csv` | Data not generated yet | Run `python src/generate_dataset.py` first |
| `FileNotFoundError: retail_features.csv` | Feature eng not run | Run steps in order via `python main.py` |
| Model MAE seems high | Normal for retail demand | Check MASE; if < 1, model beats naive baseline |
| Streamlit won't start | Port conflict or missing data | Try `streamlit run app/streamlit_app.py --server.port 8502` |
| `ValueError: Input contains NaN` | Cold-start rows not dropped | Ensure `dropna()` is called after feature engineering |
| Chart not saved | Wrong image directory path | Check `IMG_DIR` path in each module |
| `PermissionError` on model save | Read-only folder | Ensure `models/` directory exists and is writable |
| Git push rejected | Large `.pkl` file | Add `models/*.pkl` to `.gitignore` or use Git LFS |
| Duplicate date entries | Multiple stores/items | Always group by `["store_id", "item_id", "date"]` |

---

## 8. Resume & LinkedIn Content

### Resume Bullet Points

```
• Built end-to-end Retail Sales Forecasting & Inventory Optimization system in Python;
  trained Random Forest Regressor (MASE=0.25, 4× better than seasonal naive baseline)
  on 21,900 rows of daily SKU-Store sales data with 48 engineered features.

• Implemented inventory science pipeline computing Safety Stock, Reorder Point (ROP),
  and Economic Order Quantity (EOQ) per SKU-Store at 95% service level using SciPy
  statistical distributions; generated automated reorder alerts.

• Deployed interactive Streamlit dashboard with live filtering, 30-day forecast
  visualization, inventory policy tables, and reorder alert panels; published
  complete project with 18 visualizations and HTML business report on GitHub.
```

### LinkedIn Project Description
```
🛒 Retail Sales Forecasting & Inventory Optimization System

Built a production-style data science pipeline that forecasts daily retail sales and
converts forecasts into optimal stocking decisions.

Tech: Python | Random Forest | Scikit-learn | SciPy | Streamlit | Pandas | Matplotlib

✅ 48-feature engineering (lags, rolling stats, Fourier seasonality)
✅ Hybrid model: Random Forest for regular SKUs, Croston/SBA for intermittent demand
✅ Inventory policy engine: Safety Stock, Reorder Point, EOQ at 95% service level
✅ 18 business visualizations | Interactive Streamlit dashboard
✅ Simulates ₹76.5M revenue, 857K units across 3 stores, 10 SKUs, 2 years

Relevant for: Data Analyst | Business Analyst | Supply Chain Analyst | Data Scientist roles

GitHub: [link]
```
