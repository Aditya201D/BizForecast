import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Fitting ARIMA model on the training series
def train_arima (train_series, order = (1,1,1), trend = "t"):
    model = ARIMA(train_series, order = order, trend=trend)
    fitted = model.fit()
    return fitted

# Used to forecast steps ahead
def forecast_arima(fitted_model, steps):
    forecast = fitted_model.forecast(steps = steps)
    return np.array(forecast)

def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse