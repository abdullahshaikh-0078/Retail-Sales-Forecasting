"""
eda.py
-------
Exploratory Data Analysis for the Retail Sales Forecasting project.
Generates charts, summary stats, and key insights.
All figures are saved to the images/ folder.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

# ── style ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})

PROCESSED_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "processed", "retail_clean.csv")
)
IMG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images"))
os.makedirs(IMG_DIR, exist_ok=True)


def _save(fig, name: str):
    path = os.path.join(IMG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"   💾  Saved: {path}")


# ── 1. Overall daily sales trend ───────────────────────────────────────────
def plot_overall_trend(df: pd.DataFrame):
    print("📈  Plotting overall sales trend …")
    daily = df.groupby("date")["qty_sold"].sum().reset_index()
    daily["7d_MA"] = daily["qty_sold"].rolling(7).mean()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(daily["date"], daily["qty_sold"], alpha=0.25, color="steelblue", label="Daily Sales")
    ax.plot(daily["date"], daily["7d_MA"], color="steelblue", linewidth=2, label="7-day MA")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    ax.set(title="Overall Daily Sales Trend (All Stores, All SKUs)",
           xlabel="Date", ylabel="Units Sold")
    ax.legend()
    _save(fig, "01_overall_sales_trend.png")


# ── 2. Category-wise total sales ───────────────────────────────────────────
def plot_category_sales(df: pd.DataFrame):
    print("📊  Plotting category-wise sales …")
    cat_sales = df.groupby("category")["qty_sold"].sum().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = sns.color_palette("muted", len(cat_sales))
    bars = ax.barh(cat_sales.index, cat_sales.values, color=colors)
    ax.bar_label(bars, labels=[f"{v:,.0f}" for v in cat_sales.values], padding=4)
    ax.set(title="Total Units Sold by Category (2022–2023)",
           xlabel="Total Units Sold")
    _save(fig, "02_category_sales.png")


# ── 3. Store-wise comparison ────────────────────────────────────────────────
def plot_store_comparison(df: pd.DataFrame):
    print("🏪  Plotting store-wise comparison …")
    store_monthly = (
        df.assign(month=df["date"].dt.to_period("M").dt.to_timestamp())
          .groupby(["store_id", "month"])["qty_sold"].sum()
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(14, 4))
    for store, grp in store_monthly.groupby("store_id"):
        ax.plot(grp["month"], grp["qty_sold"], marker="o", markersize=3, label=store)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=30)
    ax.set(title="Monthly Sales by Store", xlabel="Month", ylabel="Units Sold")
    ax.legend(title="Store")
    _save(fig, "03_store_comparison.png")


# ── 4. Weekly seasonality (day-of-week pattern) ────────────────────────────
def plot_dow_pattern(df: pd.DataFrame):
    print("📅  Plotting day-of-week pattern …")
    df2 = df.copy()
    df2["dow"] = df2["date"].dt.day_name()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_avg   = df2.groupby("dow")["qty_sold"].mean().reindex(dow_order)

    fig, ax = plt.subplots(figsize=(9, 4))
    palette = ["#E74C3C" if d in ("Saturday","Sunday") else "#3498DB" for d in dow_order]
    ax.bar(dow_order, dow_avg.values, color=palette)
    ax.set(title="Average Daily Sales by Day of Week",
           xlabel="Day", ylabel="Avg Units Sold")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, "04_dow_pattern.png")


# ── 5. Promotion lift analysis ─────────────────────────────────────────────
def plot_promo_lift(df: pd.DataFrame):
    print("🎯  Plotting promotion lift …")
    promo_avg = df.groupby(["category", "on_promo"])["qty_sold"].mean().unstack()
    promo_avg.columns = ["No Promo", "On Promo"]
    promo_avg["Lift %"] = ((promo_avg["On Promo"] / promo_avg["No Promo"]) - 1) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    promo_avg[["No Promo", "On Promo"]].plot(kind="bar", ax=axes[0], color=["#95A5A6", "#E74C3C"])
    axes[0].set(title="Avg Daily Sales: Promo vs Non-Promo",
                xlabel="Category", ylabel="Avg Units Sold")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend()

    promo_avg["Lift %"].sort_values().plot(kind="barh", ax=axes[1], color="#27AE60")
    axes[1].axvline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set(title="Promo Lift % by Category", xlabel="Lift %")

    fig.suptitle("Promotion Impact Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "05_promo_lift.png")


# ── 6. Top SKUs by revenue ─────────────────────────────────────────────────
def plot_top_skus(df: pd.DataFrame):
    print("🏆  Plotting top SKUs by revenue …")
    df2 = df.copy()
    df2["revenue"] = df2["qty_sold"] * df2["price"]
    sku_rev = (
        df2.groupby(["item_id", "product_name"])["revenue"].sum()
           .reset_index().sort_values("revenue", ascending=False).head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(sku_rev["product_name"], sku_rev["revenue"] / 1e6, color=sns.color_palette("Blues_r", 10))
    ax.bar_label(bars, labels=[f"₹{v:.1f}M" for v in sku_rev["revenue"] / 1e6], padding=3, fontsize=9)
    ax.set(title="Top 10 SKUs by Total Revenue (2022–2023)",
           xlabel="Product", ylabel="Revenue (₹ Million)")
    ax.tick_params(axis="x", rotation=35)
    _save(fig, "06_top_skus_revenue.png")


# ── 7. Intermittency / zero-demand share per SKU ───────────────────────────
def plot_intermittency(df: pd.DataFrame):
    print("📉  Plotting intermittency (zero-demand share) …")
    p_zero = (
        df.groupby(["item_id", "product_name"])["qty_sold"]
          .apply(lambda s: (s == 0).mean() * 100)
          .reset_index(name="pct_zero")
          .sort_values("pct_zero", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#E74C3C" if v > 20 else "#3498DB" for v in p_zero["pct_zero"]]
    ax.bar(p_zero["product_name"], p_zero["pct_zero"], color=colors)
    ax.axhline(20, color="red", linestyle="--", linewidth=1, label="20% threshold (intermittent)")
    ax.set(title="Zero-Demand Share per SKU (%)",
           xlabel="Product", ylabel="% Days with Zero Sales")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    _save(fig, "07_intermittency.png")


# ── 8. Correlation heatmap ─────────────────────────────────────────────────
def plot_correlation(df: pd.DataFrame):
    print("🔗  Plotting correlation heatmap …")
    num_cols = ["qty_sold", "price", "discount_pct", "on_promo",
                "holiday_flag", "festival_flag", "stock_on_hand"]
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title("Correlation Heatmap of Key Variables", fontsize=13, fontweight="bold")
    _save(fig, "08_correlation_heatmap.png")


# ── 9. Monthly revenue heatmap (category × month) ─────────────────────────
def plot_revenue_heatmap(df: pd.DataFrame):
    print("🗓️  Plotting revenue heatmap …")
    df2 = df.copy()
    df2["revenue"] = df2["qty_sold"] * df2["price"]
    df2["month"]   = df2["date"].dt.to_period("M").astype(str)
    pivot = df2.pivot_table(values="revenue", index="category", columns="month",
                            aggfunc="sum", fill_value=0)
    pivot_k = pivot / 1000  # in thousands

    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(pivot_k, cmap="YlOrRd", ax=ax, linewidths=0.3,
                cbar_kws={"label": "Revenue (₹ Thousands)"})
    ax.set(title="Monthly Revenue Heatmap by Category (₹ Thousands)",
           xlabel="Month", ylabel="Category")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, "09_revenue_heatmap.png")


# ── 10. Sales distribution box-plot per category ──────────────────────────
def plot_sales_distribution(df: pd.DataFrame):
    print("📦  Plotting sales distribution …")
    fig, ax = plt.subplots(figsize=(12, 5))
    order = df.groupby("category")["qty_sold"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="category", y="qty_sold", order=order,
                palette="pastel", ax=ax, showfliers=False)
    ax.set(title="Daily Sales Distribution by Category (outliers hidden)",
           xlabel="Category", ylabel="Units Sold per Day")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, "10_sales_distribution.png")


# ── Summary stats printout ─────────────────────────────────────────────────
def print_summary(df: pd.DataFrame):
    print("\n" + "="*60)
    print("📋  EDA SUMMARY")
    print("="*60)
    print(f"  Date range   : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Stores       : {df['store_id'].nunique()}")
    print(f"  SKUs         : {df['item_id'].nunique()}")
    print(f"  Categories   : {df['category'].nunique()}")
    print(f"  Total rows   : {len(df):,}")
    print(f"  Total units  : {df['qty_sold'].sum():,.0f}")
    print(f"  Total revenue: ₹{(df['qty_sold']*df['price']).sum()/1e6:.2f} M")
    print(f"  Promo days % : {df['on_promo'].mean()*100:.1f}%")
    print(f"  Zero-demand %: {(df['qty_sold']==0).mean()*100:.1f}%")
    print("="*60 + "\n")


def run_eda(data_path: str = PROCESSED_PATH):
    print("🚀  Starting EDA pipeline …")
    df = pd.read_csv(data_path, parse_dates=["date"])
    print_summary(df)
    plot_overall_trend(df)
    plot_category_sales(df)
    plot_store_comparison(df)
    plot_dow_pattern(df)
    plot_promo_lift(df)
    plot_top_skus(df)
    plot_intermittency(df)
    plot_correlation(df)
    plot_revenue_heatmap(df)
    plot_sales_distribution(df)
    print("\n✅  EDA complete!  All charts saved to images/")
    return df


if __name__ == "__main__":
    run_eda()
