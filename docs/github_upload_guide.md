# 📤 GitHub Upload Guide — Step by Step

## Prerequisites
- Git installed on your computer
- GitHub account created
- Project files ready locally

---

## Step 1: Create the GitHub Repository

1. Go to **github.com** → click **"New repository"** (+ icon, top right)
2. Fill in:
   - **Repository name:** `Retail-Sales-Forecasting-Inventory-Optimization`
   - **Description:** `End-to-end retail demand forecasting + inventory optimization system (Random Forest · Safety Stock · EOQ · ROP · Streamlit Dashboard)`
   - **Visibility:** Public
   - **DO NOT** initialize with README (we already have one)
3. Click **"Create repository"**

---

## Step 2: Initialize Git in Your Project Folder

Open terminal / command prompt inside your project folder:

```bash
cd Retail-Sales-Forecasting

# Initialize git
git init

# Set your identity (first time only)
git config user.name "Your Name"
git config user.email "your.email@gmail.com"
```

---

## Step 3: Stage and Commit Files

```bash
# Add all files
git add .

# First commit
git commit -m "feat: initial project — retail sales forecasting + inventory optimization system"
```

---

## Step 4: Connect to GitHub and Push

```bash
# Add the GitHub remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/Retail-Sales-Forecasting-Inventory-Optimization.git

# Rename branch to 'main'
git branch -M main

# Push to GitHub
git push -u origin main
```

---

## Step 5: Add GitHub Topics (Tags)

In your GitHub repository page:
1. Click ⚙️ gear icon next to **"About"**
2. Add these topics:
   ```
   machine-learning  python  retail-analytics  demand-forecasting
   inventory-optimization  streamlit  data-science  scikit-learn
   time-series  eoq  safety-stock  portfolio-project
   ```

---

## Step 6: Day-by-Day Commit Strategy (Looks Real!)

To make your GitHub contribution graph look active, commit in stages:

```bash
# Day 1 — Setup
git add main.py requirements.txt .gitignore
git commit -m "feat: project scaffold, requirements, and main runner"
git push

# Day 2 — Dataset
git add src/generate_dataset.py data/raw/
git commit -m "feat: synthetic retail dataset generator (21,900 rows, 3 stores, 10 SKUs)"
git push

# Day 3 — Preprocessing
git add src/preprocess.py data/processed/retail_clean.csv
git commit -m "feat: data preprocessing pipeline with quality checks and weekly aggregation"
git push

# Day 4 — EDA
git add src/eda.py images/01_* images/02_* images/03_* images/04_* images/05_*
git add images/06_* images/07_* images/08_* images/09_* images/10_*
git commit -m "feat: exploratory data analysis — 10 charts (trends, seasonality, promo lift)"
git push

# Day 5 — Feature Engineering
git add src/feature_engineering.py data/processed/retail_features.csv
git commit -m "feat: 48-feature engineering — lags, rolling stats, Fourier terms, price features"
git push

# Day 6 — Forecasting Model
git add src/forecasting_model.py models/ images/11_* images/12_* images/13_*
git add outputs/forecasts/
git commit -m "feat: Random Forest forecaster (MAE=5.94) + Croston for intermittent SKUs"
git push

# Day 7 — Inventory Optimization
git add src/inventory_optimization.py outputs/inventory/ images/14_* images/15_*
git add images/16_* images/17_*
git commit -m "feat: inventory policy engine — Safety Stock, ROP, EOQ, reorder alerts"
git push

# Day 8 — Business Insights
git add src/business_insights.py outputs/reports/ images/18_*
git commit -m "feat: executive dashboard, HTML business report, KPI summary"
git push

# Day 9 — Dashboard
git add app/streamlit_app.py
git commit -m "feat: interactive Streamlit dashboard — 5 tabs, live filtering, CSV downloads"
git push

# Day 10 — Tests + Docs
git add tests/ docs/ README.md
git commit -m "docs: complete README, technical docs, unit tests, GitHub guide"
git push
```

---

## Step 7: Pin the Repository

On your GitHub profile page:
1. Click **"Customize your pins"**
2. Select this repository
3. It will now appear prominently on your profile

---

## Step 8: Screenshot Checklist (Proof Assets)

Save these screenshots to `images/screenshots/`:

| # | Screenshot | Filename |
|---|-----------|---------|
| 1 | Dataset preview (first 10 rows, column names) | `ss_01_dataset_preview.png` |
| 2 | Quality check output in terminal | `ss_02_quality_checks.png` |
| 3 | EDA grid — overall sales trend | `ss_03_eda_trend.png` |
| 4 | Promo lift chart | `ss_04_promo_lift.png` |
| 5 | Feature importance chart | `ss_05_feature_importance.png` |
| 6 | Model training terminal output (MAE, RMSE, MASE) | `ss_06_model_metrics.png` |
| 7 | Actual vs Predicted chart | `ss_07_actual_vs_predicted.png` |
| 8 | 30-day forecast chart | `ss_08_forecast.png` |
| 9 | Inventory policy table | `ss_09_inventory_table.png` |
| 10 | Reorder alert screenshot | `ss_10_reorder_alerts.png` |
| 11 | Executive dashboard | `ss_11_dashboard.png` |
| 12 | Streamlit app running | `ss_12_streamlit.png` |
| 13 | GitHub repository page | `ss_13_github_repo.png` |
| 14 | GitHub commit history | `ss_14_commit_history.png` |

---

## Useful Git Commands

```bash
# Check status
git status

# See commit history
git log --oneline

# Undo last commit (keep files)
git reset HEAD~1

# Pull latest from GitHub
git pull origin main

# Create a new branch for a feature
git checkout -b feature/add-xgboost-model

# Merge branch back to main
git checkout main
git merge feature/add-xgboost-model
```
