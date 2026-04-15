"""
inventory_optimization.py
---------------------------
Translates sales forecasts into inventory decisions for every SKU-Store.

Computes for each SKU-Store:
  - Mean demand during lead time (μ_L)
  - Demand std during lead time (σ_L)
  - Safety Stock (SS)         = z * σ_L
  - Reorder Point (ROP)       = μ_L + SS
  - Economic Order Quantity (EOQ) = sqrt(2*D*K / H)
  - Order Quantity (Q)        = max(EOQ, ROP - OnHand)
  - Reorder Alert flag

Also generates:
  - Inventory recommendation table (CSV)
  - Reorder alert table (CSV)
  - Visualizations (inventory metrics chart, ROP heatmap, EOQ chart)
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})

CLEAN_PATH    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "retail_clean.csv"))
FORECAST_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "forecasts", "forecast_output.csv"))
INV_OUT       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "inventory"))
IMG_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "images"))
os.makedirs(INV_OUT, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

SERVICE_LEVEL = 0.95   # 95% service level (z ≈ 1.645)
ORDERING_COST = 500    # Rs per order (K)
DEFAULT_HOLD  = 0.20   # 20% holding cost rate per year


# ─────────────────────────────────────────────
# CORE INVENTORY POLICY FUNCTION
# ─────────────────────────────────────────────
def compute_inventory_policy(
    forecast_h: np.ndarray,
    resid_std: float,
    on_hand: float,
    lead_time: int,
    service: float = SERVICE_LEVEL,
    annual_demand: float = None,
    ordering_cost: float = ORDERING_COST,
    unit_cost: float = 100.0,
    holding_rate: float = DEFAULT_HOLD,
) -> dict:
    """
    Compute safety stock, reorder point, EOQ, and order quantity.

    Parameters
    ----------
    forecast_h   : array of predicted demand for next H periods
    resid_std    : historical forecast residual std (proxy for demand uncertainty)
    on_hand      : current inventory on hand
    lead_time    : supplier lead time in periods (days)
    service      : desired service level (e.g., 0.95)
    annual_demand: total annual demand (units); estimated if None
    ordering_cost: fixed cost per order placed (Rs)
    unit_cost    : unit purchase cost (Rs)
    holding_rate : annual holding cost as fraction of unit cost

    Returns
    -------
    dict with SS, ROP, EOQ, order_qty, reorder_alert
    """
    z       = norm.ppf(service)                          # e.g., 1.645 for 95%
    mu_L    = float(np.sum(forecast_h[:lead_time]))      # mean demand during lead time
    sigma_L = float(resid_std * (lead_time ** 0.5))      # std during lead time (σ_L = σ*√L)

    SS  = z * sigma_L                                    # Safety Stock
    ROP = mu_L + SS                                      # Reorder Point

    # Annual demand estimate
    if annual_demand is None:
        daily_avg     = mu_L / max(lead_time, 1)
        annual_demand = daily_avg * 365

    H   = unit_cost * holding_rate                       # annual holding cost per unit
    EOQ = (np.sqrt((2 * annual_demand * ordering_cost) / H)
           if H > 0 else mu_L)                           # Economic Order Quantity

    # Order quantity = if on-hand < ROP, order enough to cover EOQ or shortage
    order_qty     = max(0, max(EOQ, ROP - on_hand)) if on_hand < ROP else 0
    reorder_alert = bool(on_hand <= ROP)

    return {
        "z_score":       round(z, 3),
        "mu_L":          round(mu_L, 2),
        "sigma_L":       round(sigma_L, 2),
        "safety_stock":  round(SS, 2),
        "reorder_point": round(ROP, 2),
        "EOQ":           round(EOQ, 2),
        "order_qty":     round(order_qty, 2),
        "on_hand":       round(on_hand, 2),
        "reorder_alert": reorder_alert,
    }


# ─────────────────────────────────────────────
# BUILD FULL INVENTORY TABLE
# ─────────────────────────────────────────────
def build_inventory_table(
    clean_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    service: float = SERVICE_LEVEL,
) -> pd.DataFrame:
    """
    For every SKU-Store combination, compute inventory policy parameters.
    """
    print("📦  Computing inventory policy for all SKU-Store combos …")
    rows = []

    for (store_id, item_id), hist_grp in clean_df.groupby(["store_id", "item_id"]):
        hist_grp = hist_grp.sort_values("date")

        # Meta from history
        lead_time    = int(hist_grp["supplier_lead_time_days"].iloc[-1])
        unit_cost    = float(hist_grp["unit_cost"].iloc[-1])
        ordering_cost= float(hist_grp["ordering_cost"].iloc[-1])
        holding_rate = float(hist_grp["holding_cost_rate"].iloc[-1])
        product_name = hist_grp["product_name"].iloc[-1]
        category     = hist_grp["category"].iloc[-1]

        # Current stock on hand (last known value)
        on_hand = float(hist_grp["stock_on_hand"].iloc[-1])

        # Forecast array for next 30 days
        fc_mask = ((forecast_df["store_id"] == store_id) &
                   (forecast_df["item_id"]  == item_id))
        fc_vals = forecast_df.loc[fc_mask, "predicted_qty"].values
        if len(fc_vals) == 0:
            fc_vals = np.array([hist_grp["qty_sold"].mean()] * 30)

        # Residual std (uncertainty proxy)
        resid_std = float(hist_grp["qty_sold"].std())
        if resid_std < 0.1:
            resid_std = 0.1

        # Annual demand estimate from history
        days_in_hist  = (hist_grp["date"].max() - hist_grp["date"].min()).days + 1
        annual_demand = (hist_grp["qty_sold"].sum() / days_in_hist) * 365

        # Compute policy
        policy = compute_inventory_policy(
            forecast_h    = fc_vals,
            resid_std     = resid_std,
            on_hand       = on_hand,
            lead_time     = lead_time,
            service       = service,
            annual_demand = annual_demand,
            ordering_cost = ordering_cost,
            unit_cost     = unit_cost,
            holding_rate  = holding_rate,
        )

        rows.append({
            "store_id":         store_id,
            "item_id":          item_id,
            "product_name":     product_name,
            "category":         category,
            "unit_cost":        unit_cost,
            "lead_time_days":   lead_time,
            "annual_demand_est":round(annual_demand, 1),
            **policy,
        })

    inv_df = pd.DataFrame(rows)
    return inv_df


# ─────────────────────────────────────────────
# REORDER ALERTS
# ─────────────────────────────────────────────
def generate_reorder_alerts(inv_df: pd.DataFrame) -> pd.DataFrame:
    alerts = inv_df[inv_df["reorder_alert"]].copy()
    alerts["urgency"] = alerts.apply(
        lambda r: "🔴 CRITICAL" if r["on_hand"] <= r["safety_stock"]
                  else "🟡 WARNING",
        axis=1
    )
    alerts = alerts.sort_values("on_hand")
    return alerts[["store_id", "item_id", "product_name", "category",
                   "on_hand", "reorder_point", "safety_stock", "order_qty", "urgency"]]


# ─────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────
def plot_safety_stock_vs_rop(inv_df: pd.DataFrame):
    """Bar chart comparing Safety Stock and ROP by SKU."""
    pivot = inv_df.groupby("product_name")[["safety_stock", "reorder_point"]].mean().sort_values("reorder_point")

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(pivot))
    width = 0.35
    ax.bar(x - width/2, pivot["safety_stock"],  width, label="Safety Stock",  color="#3498DB")
    ax.bar(x + width/2, pivot["reorder_point"], width, label="Reorder Point", color="#E74C3C")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=35, ha="right")
    ax.set(title="Safety Stock vs Reorder Point by Product (avg across stores)",
           xlabel="Product", ylabel="Units")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "14_ss_vs_rop.png")
    fig.savefig(path); plt.close(fig)
    print(f"   💾  Saved: {path}")


def plot_eoq_by_category(inv_df: pd.DataFrame):
    """EOQ distribution by category (box plot)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    order = inv_df.groupby("category")["EOQ"].median().sort_values(ascending=False).index
    sns.boxplot(data=inv_df, x="category", y="EOQ", order=order, palette="Set2", ax=ax)
    ax.set(title="EOQ Distribution by Category",
           xlabel="Category", ylabel="Economic Order Quantity (units)")
    ax.tick_params(axis="x", rotation=20)
    path = os.path.join(IMG_DIR, "15_eoq_by_category.png")
    fig.savefig(path); plt.close(fig)
    print(f"   💾  Saved: {path}")


def plot_reorder_alert_heatmap(inv_df: pd.DataFrame):
    """Heatmap: reorder alert status for each SKU × Store."""
    pivot = inv_df.pivot_table(values="reorder_alert", index="product_name",
                               columns="store_id", aggfunc="first").astype(float)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn_r",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Reorder Alert (1=YES)"})
    ax.set(title="Reorder Alert Status (1 = Needs Reorder) by Product × Store",
           xlabel="Store", ylabel="Product")
    path = os.path.join(IMG_DIR, "16_reorder_alert_heatmap.png")
    fig.savefig(path); plt.close(fig)
    print(f"   💾  Saved: {path}")


def plot_on_hand_vs_rop(inv_df: pd.DataFrame):
    """Scatter: On-Hand Stock vs Reorder Point, colored by alert status."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = inv_df["reorder_alert"].map({True: "#E74C3C", False: "#27AE60"})
    ax.scatter(inv_df["reorder_point"], inv_df["on_hand"], c=colors, alpha=0.7, s=80)
    max_val = max(inv_df["reorder_point"].max(), inv_df["on_hand"].max()) * 1.05
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, label="On-Hand = ROP (threshold)")
    ax.set(title="On-Hand Stock vs Reorder Point",
           xlabel="Reorder Point (units)", ylabel="Current On-Hand Stock (units)")
    ax.legend()
    # Add legend patches
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor="#E74C3C", label="Reorder Alert"),
                  Patch(facecolor="#27AE60", label="Stock OK")]
    ax.legend(handles=legend_els, loc="upper left")
    path = os.path.join(IMG_DIR, "17_on_hand_vs_rop.png")
    fig.savefig(path); plt.close(fig)
    print(f"   💾  Saved: {path}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run_inventory_optimization(
    clean_path:    str = CLEAN_PATH,
    forecast_path: str = FORECAST_PATH,
    service_level: float = SERVICE_LEVEL,
):
    print("🚀  Starting Inventory Optimization Pipeline …\n")

    clean_df    = pd.read_csv(clean_path,    parse_dates=["date"])
    forecast_df = pd.read_csv(forecast_path, parse_dates=["forecast_date"])

    # Build full inventory policy table
    inv_df = build_inventory_table(clean_df, forecast_df, service=service_level)

    # Save full table
    inv_path = os.path.join(INV_OUT, "inventory_policy_table.csv")
    inv_df.to_csv(inv_path, index=False)
    print(f"\n💾  Inventory policy table saved: {inv_path}")

    # Generate reorder alerts
    alerts = generate_reorder_alerts(inv_df)
    alert_path = os.path.join(INV_OUT, "reorder_alerts.csv")
    alerts.to_csv(alert_path, index=False)
    print(f"💾  Reorder alerts saved: {alert_path}")

    # Print summary
    n_alerts   = inv_df["reorder_alert"].sum()
    total_skus = len(inv_df)
    print(f"\n🔔  Reorder Alerts: {n_alerts} / {total_skus} SKU-Store combos need reorder")
    print(f"   Service Level Target: {service_level*100:.0f}%")
    print(f"   Avg Safety Stock : {inv_df['safety_stock'].mean():.1f} units")
    print(f"   Avg Reorder Point: {inv_df['reorder_point'].mean():.1f} units")
    print(f"   Avg EOQ          : {inv_df['EOQ'].mean():.1f} units")

    # Preview alert table
    if len(alerts) > 0:
        print("\n🚨  Top Reorder Alerts:")
        print(alerts.head(10).to_string(index=False))

    # Visualizations
    print("\n📊  Generating inventory charts …")
    plot_safety_stock_vs_rop(inv_df)
    plot_eoq_by_category(inv_df)
    plot_reorder_alert_heatmap(inv_df)
    plot_on_hand_vs_rop(inv_df)

    print("\n✅  Inventory optimization pipeline complete!")
    return inv_df, alerts


if __name__ == "__main__":
    inv_df, alerts = run_inventory_optimization()
