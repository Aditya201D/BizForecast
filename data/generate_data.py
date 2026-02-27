import pandas as pd
import numpy as np

np.random.seed(42)

days = 200
dates = pd.date_range(start = "2024-01-01", periods = days)

trend = np.linspace(20, 50, days)
seasonality = 10 * np.sin(np.linspace(0, 3*np.pi, days))
noise = np.random.normal(0,3, days)

sales = trend + seasonality + noise
sales = np.maximum(0,sales)

df = pd.DataFrame({
    "date": dates,
    "sales": sales.astype(int)
})

df.to_csv("sales_data.csv", index=False)
print("Sample data has been generated.")