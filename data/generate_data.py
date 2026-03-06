import pandas as pd
import numpy as np

np.random.seed(42)

days = 365
dates = pd.date_range(start="2024-01-01", periods=days)

products = {
    "P001": {"base": 18, "trend": 0.015, "noise": 2.0, "promo_boost": 8},
    "P002": {"base": 10, "trend": 0.010, "noise": 3.5, "promo_boost": 5},
    "P003": {"base": 35, "trend": 0.020, "noise": 3.0, "promo_boost": 12},
    "P004": {"base": 22, "trend": 0.012, "noise": 2.5, "promo_boost": 15},
    "P005": {"base": 8,  "trend": 0.018, "noise": 1.8, "promo_boost": 4},
}

# Weekly demand pattern (Mon to Sun)
weekly_pattern = np.array([0.95, 1.00, 1.02, 1.00, 1.08, 1.20, 1.15])

rows = []

for product_id, cfg in products.items():
    base = cfg["base"]
    trend_strength = cfg["trend"]
    noise_scale = cfg["noise"]
    promo_boost = cfg["promo_boost"]

    # random promotion days
    promo_days = set(np.random.choice(range(days), size=18, replace=False))

    for i, date in enumerate(dates):
        trend = base * (1 + trend_strength * (i / days) * 10)

        seasonal = weekly_pattern[date.dayofweek]

        noise = np.random.normal(0, noise_scale)

        promo_effect = promo_boost if i in promo_days else 0

        # occasional negative shock (stockout / disruption / low footfall)
        shock = 0
        if np.random.rand() < 0.03:
            shock = -np.random.uniform(3, 8)

        sales = trend * seasonal + noise + promo_effect + shock
        sales = max(0, round(sales))

        rows.append([date, product_id, int(sales)])

df = pd.DataFrame(rows, columns=["date", "product_id", "sales"])
df.to_csv("sales_data.csv", index=False)

print("Realistic multi-product sales_data.csv generated.")
print(df.head())
print(f"\nTotal rows: {len(df)}")