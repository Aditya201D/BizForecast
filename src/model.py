import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_loader import load_data
from preprocessing import preprocess_data
from arima_model import train_arima, forecast_arima

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

## Naive Baseline model for comparison
# Simplest baseline for building a naive model :
#     y_​t ​= y_(t−1)​
# MEANING : Tomorrow’s sales will be same as today’s.

naive_predictions = df_reg['lag_1'][split_index:]

naive_mae = mean_absolute_error(Y_test, naive_predictions)
naive_rmse = np.sqrt(mean_squared_error(Y_test, naive_predictions))


# ARIMA Model Implementation
#we use the same date range as the regression model

series = df_raw.set_index('date')['sales'].sort_index()

split_date = test_dates.iloc[0]

train_series = series[series.index < split_date]
test_series = series[series.index >= split_date]

#Matching the length of dataframes of raw and regression model
arima_order = (1, 1, 1)
arima_model = train_arima(train_series, order = arima_order, trend = "t")
arima_prediction = forecast_arima(arima_model, steps = len(test_series))

min_len = min(len(test_series), len(arima_prediction), len(Y_test))
arima_true = test_series.values[:min_len]
arima_prediction = arima_prediction[:min_len]

arima_mae = mean_absolute_error(arima_true, arima_prediction)
arima_rmse = np.sqrt(mean_squared_error(arima_true, arima_prediction))

print("\n +++++ Model Comparison +++++")
print("------Regression-----")
print("MAE: ", round(reg_mae, 2))
print("RMSE: ", round(reg_rmse, 2))

print("------Naive Baseline Model------")
print("MAE: ", round(naive_mae, 2))
print("RMSE: ", round(naive_rmse, 2))

print("------ARIMA Model--------")
print("MAE: ", round(arima_mae, 2))
print("RMSE: ", round(arima_rmse, 2))

# Actual values from csv file used to plot
actual = Y_test.values

plt.figure(figsize=(12 , 6))

plt.plot(test_dates, actual, label = "Actual")
plt.plot(test_dates, reg_predictions, label = "Regression Model Predictions")
plt.plot(test_dates, naive_predictions, label = "Baseline Model Predictions")
plt.plot(test_dates.values[:min_len], arima_prediction, label = f"ARIMA {arima_order} prediction")

plt.title("Model Comparison")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.xticks(rotation = 45)
plt.legend()
plt.tight_layout()
plt.show()