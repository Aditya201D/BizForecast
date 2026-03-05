import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_loader import load_data
from preprocessing import preprocess_data
from arima_model import train_sarima, forecast_sarima
from inventory import safety_stock, reorder_point, recommended_order_quantity, inventory_decision

# Loading the raw, generated data
df_raw = load_data("../data/sales_data.csv")

#Regression has its own dataframe to include feature
#engineering, ARIMA dataframe won't need it
df_reg = preprocess_data(df_raw.copy())

features = [
    'day_of_week',
    'month',
    'lag_1',
    'lag_7',
    'rolling_mean_7'
]

X = df_reg[features]
Y = df_reg['sales']

split_index = int(len(df_reg) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]
Y_train = Y[:split_index]
Y_test = Y[split_index:]

test_dates = df_reg['date'][split_index:]

# Regression model Implementation
reg_model = LinearRegression()
reg_model.fit(X_train, Y_train)
reg_predictions = reg_model.predict(X_test)

reg_mae = mean_absolute_error(Y_test, reg_predictions)
reg_rmse = np.sqrt(mean_squared_error(Y_test, reg_predictions))

# INVENTORY SYSTEM USING REGRESSION MODEL:

lead_time_days = 7
service_level = 0.95
current_inventory = 120
target_days = 14

avg_daily_demand = float(np.mean(reg_predictions))
residuals = Y_test.values - reg_predictions
demand_std = float(np.std(residuals))

ss = safety_stock(demand_std, lead_time_days, service_level)
rop = reorder_point(avg_daily_demand, lead_time_days, ss)
order_qty = recommended_order_quantity(target_days, avg_daily_demand, current_inventory, ss)
status = inventory_decision(current_inventory, rop)

print("\n----- Inventory Recommendation (Regression-Based) -----")
print(f"Avg daily demand (forecast): {avg_daily_demand:.2f}")
print(f"Demand uncertainty (sigma):  {demand_std:.2f}")
print(f"Safety stock:               {ss:.2f}")
print(f"Reorder point (ROP):        {rop:.2f}")
print(f"Current inventory:          {current_inventory:.2f}")
print(f"Status:                     {status}")
print(f"Recommended order qty:      {order_qty:.2f}")

## Naive Baseline model for comparison
# Simplest baseline for building a naive model :
#     y_​t ​= y_(t−1)​
# MEANING : Tomorrow’s sales will be same as today’s.

naive_predictions = df_reg['lag_1'][split_index:]

naive_mae = mean_absolute_error(Y_test, naive_predictions)
naive_rmse = np.sqrt(mean_squared_error(Y_test, naive_predictions))


# SARIMA Model Implementation
#we use the same date range as the regression model

series = df_raw.set_index('date')['sales'].sort_index()
series = series.asfreq("D")

split_date = test_dates.iloc[0]

train_series = series[series.index < split_date]
test_series = series[series.index >= split_date]

sarima_order = (2,1,2)
seasonal_order  = (1,0,0,7)

sarima_model = train_sarima(
    train_series,
    order = sarima_order,
    seasonal_order= seasonal_order
)

sarima_prediction = forecast_sarima(sarima_model, steps = len(test_series))

min_len = min(len(test_series), len(sarima_prediction))
sarima_true = test_series.values[:min_len]
sarima_prediction = sarima_prediction[:min_len]

sarima_mae = mean_absolute_error(sarima_true, sarima_prediction)
sarima_rmse = np.sqrt(mean_squared_error(sarima_true, sarima_prediction))

print("\n +++++ Model Comparison +++++")
print("------Regression-----")
print("MAE: ", round(reg_mae, 2))
print("RMSE: ", round(reg_rmse, 2))

print("------Naive Baseline Model------")
print("MAE: ", round(naive_mae, 2))
print("RMSE: ", round(naive_rmse, 2))

print("------SARIMA Model--------")
print("MAE: ", round(sarima_mae, 2))
print("RMSE: ", round(sarima_rmse, 2))

# Actual values from csv file used to plot
actual = Y_test.values

plt.figure(figsize=(12 , 6))

plt.plot(test_dates, actual, label = "Actual")
plt.plot(test_dates, reg_predictions, label = "Regression Model Predictions")
plt.plot(test_dates, naive_predictions, label = "Baseline Model Predictions")
plt.plot(test_dates.values[:min_len], sarima_prediction, label = f"ARIMA {sarima_order} x {seasonal_order} prediction")

plt.title("Model Comparison")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.xticks(rotation = 45)
plt.legend()
plt.tight_layout()
plt.show()