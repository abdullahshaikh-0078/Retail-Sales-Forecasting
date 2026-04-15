"""
generate_dataset.py
--------------------
Generates a realistic synthetic retail sales dataset for simulation.
Covers 2 years of daily sales across multiple stores, products, and categories.
Simulates: seasonality, trends, promotions, stockouts, lead times, pricing.
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

START_DATE  = "2022-01-01"
END_DATE    = "2023-12-31"

STORES = ["S001", "S002", "S003"]
STORE_FOOTFALL = {"S001": 1.0, "S002": 0.75, "S003": 1.25}  # multiplier

PRODUCTS = {
    "P001": {"name": "Rice 5kg",       "category": "Grocery",    "base_price": 250,  "unit_cost": 180, "pack_size": 5,  "shelf_life": 365},
    "P002": {"name": "Milk 1L",        "category": "Dairy",      "base_price": 60,   "unit_cost": 45,  "pack_size": 1,  "shelf_life": 7},
    "P003": {"name": "Shampoo 200ml",  "category": "Personal",   "base_price": 180,  "unit_cost": 110, "pack_size": 1,  "shelf_life": 730},
    "P004": {"name": "Biscuits 500g",  "category": "Snacks",     "base_price": 80,   "unit_cost": 55,  "pack_size": 1,  "shelf_life": 180},
    "P005": {"name": "Cooking Oil 1L", "category": "Grocery",    "base_price": 140,  "unit_cost": 100, "pack_size": 1,  "shelf_life": 365},
    "P006": {"name": "Soft Drink 2L",  "category": "Beverages",  "base_price": 90,   "unit_cost": 60,  "pack_size": 1,  "shelf_life": 180},
    "P007": {"name": "Detergent 1kg",  "category": "Household",  "base_price": 120,  "unit_cost": 80,  "pack_size": 1,  "shelf_life": 730},
    "P008": {"name": "Bread Loaf",     "category": "Bakery",     "base_price": 45,   "unit_cost": 30,  "pack_size": 1,  "shelf_life": 5},
    "P009": {"name": "Chips 100g",     "category": "Snacks",     "base_price": 30,   "unit_cost": 18,  "pack_size": 1,  "shelf_life": 90},
    "P010": {"name": "Toothpaste",     "category": "Personal",   "base_price": 110,  "unit_cost": 70,  "pack_size": 1,  "shelf_life": 730},
}

SUPPLIER_LEAD_TIMES = {
    "Grocery":   7,
    "Dairy":     2,
    "Personal":  10,
    "Snacks":    5,
    "Beverages": 5,
    "Household": 8,
    "Bakery":    1,
}

ORDERING_COST   = 500   # Rs per order
HOLDING_RATE    = 0.20  # 20% of unit cost per year

# ─────────────────────────────────────────────
# HELPER: INDIA HOLIDAYS / FESTIVALS
# ─────────────────────────────────────────────
HOLIDAYS = [
    "2022-01-26", "2022-03-18", "2022-04-14", "2022-08-15",
    "2022-10-02", "2022-10-05", "2022-10-24", "2022-11-08",
    "2022-12-25", "2023-01-26", "2023-03-08", "2023-03-30",
    "2023-04-14", "2023-08-15", "2023-10-02", "2023-10-24",
    "2023-11-12", "2023-12-25",
]

FESTIVAL_WINDOWS = [
    ("2022-10-01", "2022-10-10"),  # Navratri/Dussehra
    ("2022-10-20", "2022-10-28"),  # Diwali window
    ("2023-10-15", "2023-10-25"),  # Navratri/Dussehra
    ("2023-11-08", "2023-11-14"),  # Diwali window
]


def is_festival(date):
    for start, end in FESTIVAL_WINDOWS:
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return 1
    return 0


def generate_promo(date, item_id, store_id):
    """Simulate ~15% days on promo, randomized but reproducible."""
    hash_val = hash((str(date.date()), item_id, store_id)) % 100
    return 1 if hash_val < 15 else 0


def simulate_demand(dates, item_id, store_id, base_price, footfall_mult, category):
    """
    Generate realistic daily demand with:
    - Weekly seasonality (weekends spike)
    - Annual seasonality (summer/winter)
    - Festival lift
    - Promo lift
    - Trend
    - Noise
    - Stockout simulation
    """
    n = len(dates)
    demand = np.zeros(n)
    on_promo = np.zeros(n, dtype=int)
    discount_pct = np.zeros(n)
    price = np.full(n, float(base_price))
    stockout_flag = np.zeros(n, dtype=int)
    stock_on_hand = np.zeros(n)

    # Base demand (units/day) varies by category
    base_map = {
        "Grocery":   25, "Dairy": 50, "Personal": 15,
        "Snacks":    40, "Beverages": 35, "Household": 12,
        "Bakery":    60,
    }
    base = base_map.get(category, 20) * footfall_mult

    stock = base * 14  # start with ~2 weeks stock
    lead_time = SUPPLIER_LEAD_TIMES.get(category, 7)
    reorder_trigger = base * lead_time * 1.3

    for i, date in enumerate(dates):
        # --- seasonal components ---
        dow = date.dayofweek          # 0=Mon … 6=Sun
        doy = date.dayofyear
        week_effect   = 1.3 if dow >= 5 else (0.85 if dow == 0 else 1.0)
        annual_effect = 1 + 0.25 * np.sin(2 * np.pi * (doy - 90) / 365)  # summer peak ~Apr
        festival_lift = 1.6 if is_festival(date) else 1.0
        trend_factor  = 1 + 0.0003 * i   # gentle upward trend

        # --- promotions ---
        promo = generate_promo(date, item_id, store_id)
        on_promo[i] = promo
        disc = round(np.random.choice([0.05, 0.10, 0.15, 0.20], p=[0.3, 0.4, 0.2, 0.1]) if promo else 0, 2)
        discount_pct[i] = disc
        actual_price = base_price * (1 - disc)
        price[i] = actual_price

        promo_lift = 1 + 1.2 * disc if promo else 1.0

        # --- compute raw demand ---
        mu = base * week_effect * annual_effect * festival_lift * trend_factor * promo_lift
        raw_demand = int(max(0, np.random.poisson(mu)))

        # --- stockout simulation ---
        if stock <= 0:
            stockout_flag[i] = 1
            actual_sold = 0
        else:
            actual_sold = min(raw_demand, int(stock))
            stock -= actual_sold

        # --- replenishment trigger ---
        if stock <= reorder_trigger:
            reorder_qty = int(base * (lead_time + 7))
            # Arrives after lead_time (simplified: instant for simulation)
            stock += reorder_qty

        stock_on_hand[i] = max(0, stock)
        demand[i] = actual_sold

    return demand, on_promo, discount_pct, price, stockout_flag, stock_on_hand


def build_dataset():
    print("🔧 Generating synthetic retail dataset...")
    dates  = pd.date_range(START_DATE, END_DATE, freq="D")
    rows   = []

    for store_id in STORES:
        footfall = STORE_FOOTFALL[store_id]
        for item_id, info in PRODUCTS.items():
            cat      = info["category"]
            bp       = info["base_price"]
            uc       = info["unit_cost"]
            shelf    = info["shelf_life"]
            lead_t   = SUPPLIER_LEAD_TIMES.get(cat, 7)

            demand, on_promo, disc, price, so_flag, soh = simulate_demand(
                dates, item_id, store_id, bp, footfall, cat
            )

            for i, date in enumerate(dates):
                rows.append({
                    "date":                  date,
                    "store_id":              store_id,
                    "item_id":               item_id,
                    "product_name":          info["name"],
                    "category":              cat,
                    "qty_sold":              int(demand[i]),
                    "price":                 round(float(price[i]), 2),
                    "on_promo":              int(on_promo[i]),
                    "discount_pct":          float(disc[i]),
                    "stock_on_hand":         int(soh[i]),
                    "stockout_flag":         int(so_flag[i]),
                    "unit_cost":             uc,
                    "pack_size":             info["pack_size"],
                    "shelf_life_days":       shelf,
                    "supplier_lead_time_days": lead_t,
                    "ordering_cost":         ORDERING_COST,
                    "holding_cost_rate":     HOLDING_RATE,
                    "holiday_flag":          1 if str(date.date()) in HOLIDAYS else 0,
                    "festival_flag":         is_festival(date),
                })

    df = pd.DataFrame(rows)
    print(f"✅ Dataset created: {len(df):,} rows | {df['store_id'].nunique()} stores | {df['item_id'].nunique()} SKUs")
    print(df.dtypes)
    return df


if __name__ == "__main__":
    df = build_dataset()

    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "retail_timeseries.csv")
    raw_path = os.path.abspath(raw_path)
    df.to_csv(raw_path, index=False)
    print(f"\n💾 Saved to: {raw_path}")
    print(df.head(3).to_string())
