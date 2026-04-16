"""
streamlit_app.py
-----------------
Interactive Streamlit dashboard for the Retail Sales Forecasting
& Inventory Optimization project.

Run with:
  streamlit run app/streamlit_app.py

Features:
  - Overview KPIs
  - SKU-level sales history + forecast visualization
  - Inventory policy table (filterable)
  - Reorder alerts panel
  - Promotion analysis
  - Download buttons for CSV outputs
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys

# ── path setup ────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

# ── page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Forecast & Inventory Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #1a1a2e; color: white; border-radius: 10px;
    padding: 16px 20px; text-align: center; margin: 4px;
  }
  .metric-val { font-size: 28px; font-weight: bold; color: #00d2ff; }
  .metric-lbl { font-size: 13px; color: #b0c4de; margin-top: 4px; }
  .alert-red  { background: #E74C3C22; border-left: 4px solid #E74C3C;
                padding: 8px 14px; border-radius: 4px; color: #c0392b; }
  .alert-grn  { background: #27AE6022; border-left: 4px solid #27AE60;
                padding: 8px 14px; border-radius: 4px; color: #1e8449; }
  h1 { color: #2C3E50 !important; }
</style>
""", unsafe_allow_html=True)


# ── data loaders (cached) ─────────────────────────────────────────────────
@st.cache_data
def load_clean():
    path = os.path.join(BASE_DIR, "data", "processed", "retail_clean.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    df["revenue"] = df["qty_sold"] * df["price"]
    return df

@st.cache_data
def load_forecast():
    path = os.path.join(BASE_DIR, "outputs", "forecasts", "forecast_output.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["forecast_date"])

@st.cache_data
def load_inventory():
    path = os.path.join(BASE_DIR, "outputs", "inventory", "inventory_policy_table.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ── SIDEBAR ───────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/emoji/96/shopping-cart-emoji.png", width=70)
st.sidebar.title("🛒 Retail Analytics")
st.sidebar.markdown("**Forecasting & Inventory Optimization**")
st.sidebar.markdown("---")

clean_df    = load_clean()
forecast_df = load_forecast()
inv_df      = load_inventory()

data_ready = (clean_df is not None) and (forecast_df is not None) and (inv_df is not None)

if not data_ready:
    st.warning("⚠️  Data not found. Please run `python main.py` first to generate all outputs.")
    st.code("cd Retail-Sales-Forecasting\npython main.py", language="bash")
    st.stop()

# Sidebar filters
st.sidebar.subheader("🔍 Filters")
stores = sorted(clean_df["store_id"].unique().tolist())
items  = sorted(clean_df["item_id"].unique().tolist())
cats   = sorted(clean_df["category"].unique().tolist())

sel_store    = st.sidebar.selectbox("Store",    ["All"] + stores)
sel_category = st.sidebar.selectbox("Category", ["All"] + cats)
sel_item     = st.sidebar.selectbox("SKU",      items)
service_lvl  = st.sidebar.slider("Service Level Target (%)", 80, 99, 95)

st.sidebar.markdown("---")
st.sidebar.info("📁 Project by: Your Name\n\nGitHub: github.com/yourusername")

# Filter logic
flt = clean_df.copy()
if sel_store    != "All": flt = flt[flt["store_id"] == sel_store]
if sel_category != "All": flt = flt[flt["category"] == sel_category]


# ── MAIN CONTENT ──────────────────────────────────────────────────────────
st.title("🛒 Retail Sales Forecasting & Inventory Optimization")
st.markdown("*End-to-end analytics system — forecasting · safety stock · EOQ · reorder alerts*")
st.markdown("---")

# ── Tab navigation ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "📈 Forecast", "📦 Inventory", "🚨 Alerts", "📉 EDA"
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📊 Business Overview KPIs")

    col1, col2, col3, col4, col5 = st.columns(5)
    total_rev   = flt["revenue"].sum() / 1e6
    total_units = flt["qty_sold"].sum()
    n_skus      = flt["item_id"].nunique()
    n_alerts    = inv_df["reorder_alert"].sum()
    promo_lift  = (flt.groupby("on_promo")["qty_sold"].mean().get(1, 0) /
                   max(flt.groupby("on_promo")["qty_sold"].mean().get(0, 1), 1) - 1) * 100

    with col1: st.metric("💰 Total Revenue", f"₹{total_rev:.2f}M")
    with col2: st.metric("📦 Units Sold",    f"{total_units:,.0f}")
    with col3: st.metric("🛍️ Active SKUs",    n_skus)
    with col4: st.metric("🚨 Reorder Alerts", int(n_alerts))
    with col5: st.metric("🎯 Promo Lift",    f"+{promo_lift:.1f}%")

    st.markdown("---")

    # Daily revenue trend
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.subheader("Daily Revenue Trend")
        daily = flt.groupby("date")["revenue"].sum().reset_index()
        daily["7d_MA"] = daily["revenue"].rolling(7).mean()
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.fill_between(daily["date"], daily["revenue"] / 1e3, alpha=0.2, color="#3498DB")
        ax.plot(daily["date"], daily["7d_MA"] / 1e3, color="#2980B9", linewidth=2)
        ax.set(ylabel="Revenue (₹ Thousands)")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig); plt.close(fig)

    with col_r:
        st.subheader("Revenue by Category")
        cat_rev = flt.groupby("category")["revenue"].sum().sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))
        colors = plt.cm.Set2(np.linspace(0, 1, len(cat_rev)))
        ax2.barh(cat_rev.index[::-1], cat_rev.values[::-1] / 1e6, color=colors)
        ax2.set(xlabel="Revenue (₹M)")
        st.pyplot(fig2); plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2: FORECAST
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"📈 Sales Forecast — SKU: {sel_item}")

    # History + forecast for selected SKU
    hist = clean_df[(clean_df["item_id"] == sel_item) & (clean_df["store_id"] == stores[0])].tail(90)
    fc   = forecast_df[(forecast_df["item_id"] == sel_item) & (forecast_df["store_id"] == stores[0])]

    if len(hist) > 0:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(hist["date"], hist["qty_sold"], color="steelblue", label="Historical", linewidth=1.5)
        if len(fc) > 0:
            ax.plot(fc["forecast_date"], fc["predicted_qty"], color="darkorange",
                    linestyle="--", linewidth=2, label="Forecast (30 days)", marker="o", markersize=4)
            ax.fill_between(fc["forecast_date"],
                            fc["predicted_qty"] * 0.80, fc["predicted_qty"] * 1.20,
                            alpha=0.15, color="darkorange", label="±20% uncertainty band")
        ax.axvline(x=hist["date"].max(), color="grey", linestyle=":", linewidth=1)
        ax.set(xlabel="Date", ylabel="Units Sold",
               title=f"Historical + 30-Day Forecast — {sel_item} | Store {stores[0]}")
        ax.legend()
        st.pyplot(fig); plt.close(fig)

        # Forecast table
        if len(fc) > 0:
            st.subheader("Forecast Table")
            fc_display = fc[["forecast_date", "predicted_qty", "method"]].copy()
            fc_display["predicted_qty"] = fc_display["predicted_qty"].round(1)
            st.dataframe(fc_display.rename(columns={
                "forecast_date": "Date",
                "predicted_qty": "Predicted Units",
                "method":        "Model"
            }), use_container_width=True)
            st.download_button("⬇️ Download Forecast CSV",
                               fc.to_csv(index=False),
                               file_name=f"forecast_{sel_item}.csv")
    else:
        st.warning("No historical data found for selected filters.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 3: INVENTORY
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📦 Inventory Policy Table")
    st.markdown(f"*Service Level: **{service_lvl}%** | Z-score: {1.645:.3f}*")

    flt_inv = inv_df.copy()
    if sel_store    != "All": flt_inv = flt_inv[flt_inv["store_id"] == sel_store]
    if sel_category != "All": flt_inv = flt_inv[flt_inv["category"] == sel_category]

    display_cols = ["store_id", "item_id", "product_name", "category",
                    "on_hand", "safety_stock", "reorder_point", "EOQ", "order_qty", "reorder_alert"]
    inv_show = flt_inv[display_cols].copy()
    inv_show["reorder_alert"] = inv_show["reorder_alert"].map({True: "🔴 YES", False: "✅ OK"})

    st.dataframe(inv_show.style.applymap(
        lambda v: "background-color: #FADBD8" if v == "🔴 YES" else "",
        subset=["reorder_alert"]
    ), use_container_width=True)

    st.download_button("⬇️ Download Inventory Policy CSV",
                       inv_df.to_csv(index=False),
                       file_name="inventory_policy_table.csv")

    # EOQ vs Safety Stock chart
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Safety Stock by Product")
        ss_prod = flt_inv.groupby("product_name")["safety_stock"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(ss_prod.index, ss_prod.values, color="#3498DB")
        ax.set(xlabel="Safety Stock (units)")
        st.pyplot(fig); plt.close(fig)

    with col_b:
        st.subheader("EOQ by Product")
        eoq_prod = flt_inv.groupby("product_name")["EOQ"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(eoq_prod.index, eoq_prod.values, color="#E74C3C")
        ax.set(xlabel="EOQ (units)")
        st.pyplot(fig); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# TAB 4: ALERTS
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🚨 Reorder Alerts")
    alerts = inv_df[inv_df["reorder_alert"]].copy()
    alerts["urgency"] = alerts.apply(
        lambda r: "🔴 CRITICAL" if r["on_hand"] <= r["safety_stock"] else "🟡 WARNING", axis=1
    )

    n_critical = (alerts["urgency"] == "🔴 CRITICAL").sum()
    n_warning  = (alerts["urgency"] == "🟡 WARNING").sum()

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Total Alerts",   len(alerts))
    with col2: st.metric("🔴 Critical",    int(n_critical))
    with col3: st.metric("🟡 Warning",     int(n_warning))

    st.markdown("---")
    alert_show = alerts[["urgency", "store_id", "item_id", "product_name", "category",
                          "on_hand", "reorder_point", "safety_stock", "order_qty"]].copy()
    alert_show = alert_show.round(1)
    st.dataframe(alert_show.sort_values("on_hand"), use_container_width=True)

    st.download_button("⬇️ Download Reorder Alerts CSV",
                       alerts.to_csv(index=False),
                       file_name="reorder_alerts.csv")


# ══════════════════════════════════════════════════════════════════════════
# TAB 5: EDA
# ══════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("📉 Exploratory Data Analysis")

    # Day-of-week pattern
    flt2 = flt.copy()
    flt2["dow"] = flt2["date"].dt.day_name()
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow_avg   = flt2.groupby("dow")["qty_sold"].mean().reindex(dow_order)

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        st.subheader("Day-of-Week Pattern")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#E74C3C" if d in ("Saturday","Sunday") else "#3498DB" for d in dow_order]
        ax.bar(dow_order, dow_avg.values, color=colors)
        ax.set(ylabel="Avg Units Sold")
        ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig); plt.close(fig)

    with col_r2:
        st.subheader("Promo vs Non-Promo Sales")
        promo_avg = flt2.groupby(["category","on_promo"])["qty_sold"].mean().unstack()
        promo_avg.columns = ["No Promo","On Promo"]
        fig, ax = plt.subplots(figsize=(6, 4))
        promo_avg.plot(kind="bar", ax=ax, color=["#95A5A6","#E74C3C"])
        ax.set(ylabel="Avg Units/Day", xlabel="")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8)
        st.pyplot(fig); plt.close(fig)

    # Raw data preview
    st.subheader("Raw Data Preview")
    st.dataframe(flt.head(100), use_container_width=True)

st.markdown("---")
st.caption("🛒 Retail Sales Forecasting & Inventory Optimization | Built with Python · Streamlit · Scikit-learn")
