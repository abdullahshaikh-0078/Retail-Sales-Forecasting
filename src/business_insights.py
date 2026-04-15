"""
business_insights.py
---------------------
Generates executive-level business insights and a final summary report.
Creates:
  - KPI summary printout
  - Combined insights dashboard image (multi-panel figure)
  - HTML summary report
  - CSV report for stakeholders
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})

CLEAN_PATH    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "retail_clean.csv"))
FORECAST_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "forecasts", "forecast_output.csv"))
INV_PATH      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "inventory", "inventory_policy_table.csv"))
REPORT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "reports"))
IMG_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images"))
os.makedirs(REPORT_DIR, exist_ok=True)


def compute_kpis(clean_df: pd.DataFrame, inv_df: pd.DataFrame, forecast_df: pd.DataFrame) -> dict:
    """Calculate business KPIs from data."""
    clean_df["revenue"] = clean_df["qty_sold"] * clean_df["price"]

    total_revenue   = clean_df["revenue"].sum()
    total_units     = clean_df["qty_sold"].sum()
    avg_daily_sales = clean_df.groupby("date")["qty_sold"].sum().mean()
    top_category    = clean_df.groupby("category")["revenue"].sum().idxmax()
    reorder_alerts  = inv_df["reorder_alert"].sum()
    total_combos    = len(inv_df)
    avg_ss          = inv_df["safety_stock"].mean()
    avg_rop         = inv_df["reorder_point"].mean()
    avg_eoq         = inv_df["EOQ"].mean()
    promo_lift      = (clean_df.groupby("on_promo")["qty_sold"].mean().get(1, 0) /
                       max(clean_df.groupby("on_promo")["qty_sold"].mean().get(0, 1), 1) - 1) * 100

    # Forecast 30-day projection
    forecast_total = forecast_df.groupby("forecast_date")["predicted_qty"].sum().sum()

    return {
        "Total Revenue (₹ M)":    round(total_revenue / 1e6, 2),
        "Total Units Sold":        f"{int(total_units):,}",
        "Avg Daily Units Sold":    round(avg_daily_sales, 0),
        "Top Revenue Category":    top_category,
        "Promo Lift (%)":          round(promo_lift, 1),
        "SKU-Store Combos":        total_combos,
        "Reorder Alerts":          int(reorder_alerts),
        "Alert Rate (%)":          round(reorder_alerts / total_combos * 100, 1),
        "Avg Safety Stock (units)":round(avg_ss, 1),
        "Avg Reorder Point":       round(avg_rop, 1),
        "Avg EOQ (units)":         round(avg_eoq, 1),
        "30-Day Forecast Total":   f"{int(forecast_total):,}",
    }


def create_executive_dashboard(clean_df: pd.DataFrame, inv_df: pd.DataFrame, forecast_df: pd.DataFrame):
    """Create a multi-panel executive summary dashboard."""
    print("🎨  Creating executive dashboard …")

    clean_df["revenue"] = clean_df["qty_sold"] * clean_df["price"]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Retail Sales Forecasting & Inventory Optimization\nExecutive Dashboard",
                 fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: Daily revenue trend ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    daily_rev = clean_df.groupby("date")["revenue"].sum().reset_index()
    daily_rev["7d_MA"] = daily_rev["revenue"].rolling(7).mean()
    ax1.fill_between(daily_rev["date"], daily_rev["revenue"] / 1e3, alpha=0.2, color="#3498DB")
    ax1.plot(daily_rev["date"], daily_rev["7d_MA"] / 1e3, color="#2980B9", linewidth=2, label="7-day MA")
    ax1.set(title="Daily Revenue Trend (₹ Thousands)", ylabel="₹ '000")
    ax1.tick_params(axis="x", rotation=20)
    ax1.legend(fontsize=8)

    # ── Panel 2: KPI cards (text) ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    total_rev = clean_df["revenue"].sum() / 1e6
    total_u   = clean_df["qty_sold"].sum()
    n_alerts  = int(inv_df["reorder_alert"].sum())
    avg_ss    = inv_df["safety_stock"].mean()
    kpi_text  = (
        f"  📊 KEY PERFORMANCE INDICATORS\n"
        f"  ─────────────────────────────\n"
        f"  Revenue    :  ₹{total_rev:.2f} M\n"
        f"  Units Sold :  {total_u:,.0f}\n"
        f"  SKU-Stores :  {len(inv_df)}\n"
        f"  Alerts     :  {n_alerts} reorders\n"
        f"  Avg SS     :  {avg_ss:.1f} units\n"
        f"  Avg EOQ    :  {inv_df['EOQ'].mean():.1f} units\n"
        f"  Avg ROP    :  {inv_df['reorder_point'].mean():.1f} units"
    )
    ax2.text(0.05, 0.95, kpi_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#EBF5FB", alpha=0.9))
    ax2.set_title("KPI Summary", fontweight="bold")

    # ── Panel 3: Category revenue pie ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    cat_rev = clean_df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    colors  = sns.color_palette("muted", len(cat_rev))
    wedges, texts, autotexts = ax3.pie(
        cat_rev.values, labels=cat_rev.index, autopct="%1.1f%%",
        colors=colors, startangle=90, pctdistance=0.75,
        textprops={"fontsize": 8}
    )
    ax3.set_title("Revenue by Category", fontweight="bold")

    # ── Panel 4: Monthly units by store ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    monthly = (clean_df.assign(month=clean_df["date"].dt.to_period("M").dt.to_timestamp())
               .groupby(["store_id", "month"])["qty_sold"].sum().reset_index())
    for store, grp in monthly.groupby("store_id"):
        ax4.plot(grp["month"], grp["qty_sold"], marker="o", markersize=2, label=store)
    ax4.set(title="Monthly Units by Store", ylabel="Units")
    ax4.tick_params(axis="x", rotation=30, labelsize=7)
    ax4.legend(fontsize=8)

    # ── Panel 5: Promo vs No-Promo boxplot ────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    clean_df["Promotion"] = clean_df["on_promo"].map({1: "On Promo", 0: "No Promo"})
    sns.boxplot(data=clean_df, x="Promotion", y="qty_sold", ax=ax5,
                palette={"On Promo": "#E74C3C", "No Promo": "#95A5A6"}, showfliers=False)
    ax5.set(title="Promo vs No-Promo Sales", ylabel="Units/Day")

    # ── Panel 6: Forecast – next 30 days ─────────────────────────────────
    ax6 = fig.add_subplot(gs[2, :2])
    fc_daily = forecast_df.groupby("forecast_date")["predicted_qty"].sum().reset_index()
    # Last 30 days of actuals for context
    last30 = clean_df.groupby("date")["qty_sold"].sum().tail(30).reset_index()
    ax6.plot(last30["date"], last30["qty_sold"], color="steelblue", label="Historical (last 30d)", linewidth=1.5)
    ax6.plot(fc_daily["forecast_date"], fc_daily["predicted_qty"], color="darkorange",
             linestyle="--", linewidth=2, label="Forecast (next 30d)", marker="o", markersize=3)
    ax6.axvline(x=last30["date"].max(), color="grey", linestyle=":", linewidth=1)
    ax6.set(title="30-Day Sales Forecast vs Historical", ylabel="Units Sold")
    ax6.tick_params(axis="x", rotation=20)
    ax6.legend(fontsize=8)

    # ── Panel 7: Reorder alerts summary ──────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis("off")
    alerts = inv_df[inv_df["reorder_alert"]].sort_values("on_hand").head(8)
    if len(alerts) > 0:
        table_data = alerts[["item_id", "product_name", "on_hand", "order_qty"]].values.tolist()
        table = ax7.table(
            cellText=table_data,
            colLabels=["SKU", "Product", "On Hand", "Order Qty"],
            loc="center", cellLoc="left"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1, 1.4)
        # Header styling
        for j in range(4):
            table[(0, j)].set_facecolor("#2C3E50")
            table[(0, j)].set_text_props(color="white", fontweight="bold")
    ax7.set_title("🚨 Top Reorder Alerts", fontweight="bold", fontsize=10)

    path = os.path.join(IMG_DIR, "18_executive_dashboard.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"   💾  Dashboard saved: {path}")


def generate_html_report(kpis: dict, inv_df: pd.DataFrame, forecast_df: pd.DataFrame):
    """Generate a simple HTML report for sharing."""
    alerts_html = inv_df[inv_df["reorder_alert"]][[
        "store_id", "item_id", "product_name", "category",
        "on_hand", "reorder_point", "safety_stock", "order_qty"
    ]].head(20).to_html(index=False, classes="table", border=0, float_format=lambda x: f"{x:.1f}")

    inv_html = inv_df[[
        "store_id", "item_id", "product_name", "category",
        "safety_stock", "reorder_point", "EOQ", "order_qty", "reorder_alert"
    ]].head(30).to_html(index=False, classes="table", border=0, float_format=lambda x: f"{x:.1f}")

    kpi_rows = "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in kpis.items())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Retail Sales Forecasting & Inventory Optimization – Report</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto; color: #2c3e50; }}
  h1   {{ background: #2C3E50; color: white; padding: 18px 24px; border-radius: 6px; }}
  h2   {{ color: #2980B9; border-bottom: 2px solid #2980B9; padding-bottom: 6px; }}
  .table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }}
  .table th {{ background: #2C3E50; color: white; padding: 8px 12px; text-align: left; }}
  .table td {{ padding: 7px 12px; border-bottom: 1px solid #ecf0f1; }}
  .table tr:nth-child(even) {{ background: #f8f9fa; }}
  .kpi-table td {{ padding: 6px 14px; }}
  .badge-alert {{ background: #E74C3C; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; }}
  .badge-ok    {{ background: #27AE60; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; }}
  img {{ max-width: 100%; border-radius: 6px; margin: 10px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
</style>
</head>
<body>
<h1>🛒 Retail Sales Forecasting & Inventory Optimization – Project Report</h1>
<p><em>Generated by the Retail Analytics System | Python · Random Forest · Safety Stock · EOQ · ROP</em></p>

<h2>📊 Key Performance Indicators</h2>
<table class="table kpi-table">
<tr><th>Metric</th><th>Value</th></tr>
{kpi_rows}
</table>

<h2>🗺️ Executive Dashboard</h2>
<img src="../../images/18_executive_dashboard.png" alt="Executive Dashboard">

<h2>📈 Forecast vs Historical</h2>
<img src="../../images/12_actual_vs_predicted.png" alt="Actual vs Predicted">
<img src="../../images/13_sku_forecast.png" alt="SKU Forecast">

<h2>📦 Inventory Policy Table (Top 30 SKU-Stores)</h2>
{inv_html}

<h2>🚨 Reorder Alerts (Items Needing Immediate Reorder)</h2>
{alerts_html}

<h2>📉 Safety Stock vs Reorder Point</h2>
<img src="../../images/14_ss_vs_rop.png" alt="Safety Stock vs ROP">
<img src="../../images/16_reorder_alert_heatmap.png" alt="Reorder Alert Heatmap">

<h2>📊 Sales Analysis</h2>
<img src="../../images/01_overall_sales_trend.png" alt="Overall Sales Trend">
<img src="../../images/02_category_sales.png" alt="Category Sales">
<img src="../../images/05_promo_lift.png" alt="Promo Lift">

<hr>
<p style="color:#7f8c8d; font-size:12px;">
  Project: Retail Sales Forecasting &amp; Inventory Optimization System |
  Tech Stack: Python, Pandas, Scikit-learn, SciPy, Matplotlib, Seaborn, Streamlit
</p>
</body>
</html>"""

    path = os.path.join(REPORT_DIR, "business_report.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"   💾  HTML report saved: {path}")


def run_business_insights(
    clean_path:    str = CLEAN_PATH,
    forecast_path: str = FORECAST_PATH,
    inv_path:      str = INV_PATH,
):
    print("🚀  Generating Business Insights …\n")
    clean_df    = pd.read_csv(clean_path,    parse_dates=["date"])
    forecast_df = pd.read_csv(forecast_path, parse_dates=["forecast_date"])
    inv_df      = pd.read_csv(inv_path)

    # KPIs
    kpis = compute_kpis(clean_df, inv_df, forecast_df)
    print("📋  KPI Summary:")
    for k, v in kpis.items():
        print(f"   {k:35s}: {v}")

    # Dashboard
    create_executive_dashboard(clean_df, inv_df, forecast_df)

    # HTML report
    generate_html_report(kpis, inv_df, forecast_df)

    # Save KPI CSV
    kpi_df = pd.DataFrame(list(kpis.items()), columns=["Metric", "Value"])
    kpi_df.to_csv(os.path.join(REPORT_DIR, "kpi_summary.csv"), index=False)
    print(f"   💾  KPI CSV saved.")

    print("\n✅  Business insights pipeline complete!")
    return kpis


if __name__ == "__main__":
    run_business_insights()
