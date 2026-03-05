import pandas as pd
import numpy as np

np.random.seed(42)

days = 200
products = ["P001", "P002", "P003"]  # start with 3 products
dates = pd.date_range(start="2024-01-01", periods=days)

rows = []

for pid in products:
    base = np.random.randint(10, 40)
    trend_strength = np.random.uniform(0.02, 0.08)  # slow upward trend
    noise_scale = np.random.uniform(2, 5)

    weekly = np.array([1.0, 1.05, 1.1, 1.08, 1.02, 1.2, 1.25])
    weekly = weekly / weekly.mean()

    for i, dt in enumerate(dates):
        trend = base * (1 + trend_strength * (i / days))
        seasonal = weekly[dt.dayofweek]
        noise = np.random.normal(0, noise_scale)

        sales = trend * seasonal + noise
        sales = max(0, sales)

        rows.append([dt, pid, int(round(sales))])

df = pd.DataFrame(rows, columns=["date", "product_id", "sales"])
df.to_csv("sales_data.csv", index=False)

print("Multi-product sales_data.csv generated.")