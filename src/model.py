from data_loader import load_data
from preprocessing import preprocess_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

df = load_data("../data/sales_data.csv")
df = preprocess_data(df)

features = [
    'day_of_week',
    'month',
    'lag_1',
    'lag_7',
    'rolling_mean_7'
]

X = df[features]
Y = df['sales']

split_index = int(len(df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]
Y_train = Y[:split_index]
Y_test = Y[split_index:]

reg_model = LinearRegression()
reg_model.fit(X_train, Y_train)
reg_predictions = reg_model.predict(X_test)

reg_mae = mean_absolute_error(Y_test, reg_predictions)
reg_rmse = np.sqrt(mean_squared_error(Y_test, reg_predictions))

## Naive Baseline model for comparison

# Simplest baseline for building a naive model :
#     y_​t ​= y_(t−1)​
# MEANING : Tomorrow’s sales will be same as today’s.

naive_predictions = df['lag_1'][split_index:]

naive_mae = mean_absolute_error(Y_test, naive_predictions)
naive_rmse = np.sqrt(mean_squared_error(Y_test, naive_predictions))


print("\n +++++ Model Comparison +++++")
print("------Regression-----")
print("MAE: ", round(reg_mae, 2))
print("RMSE: ", round(reg_rmse, 2))

print("------Naive Baseline Model------")
print("MAE: ", round(naive_mae, 2))
print("RMSE: ", round(naive_rmse, 2))
