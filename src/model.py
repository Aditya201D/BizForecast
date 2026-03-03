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

model = LinearRegression()
model.fit(X_train, Y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(Y_test, predictions)
rmse = np.sqrt(mean_squared_error(Y_test, predictions))

print("MAE: ", round(mae, 2))
print("RMSE: ", round(rmse, 2))
