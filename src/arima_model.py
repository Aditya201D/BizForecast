import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_sarima (train_series, order = (1,1,1), seasonal_order = (1,0,1,7)):
    model = SARIMAX(
        train_series,
        order = order,
        seasonal_order = seasonal_order,
        enforce_stationarity = False,
        enforce_invertibility = False
    )
    fitted = model.fit(disp = False)
    return fitted

def forecast_sarima(fitted_model, steps):
    forecast = fitted_model.forecast(steps = steps)
    return np.array(forecast)

def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse