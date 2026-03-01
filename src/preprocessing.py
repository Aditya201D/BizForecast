import pandas as pd

def add_time_features(df):
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    return df

def add_lag_features(df , lags=[1,7]):
    for lag in lags:
        df[f'lag_{lag}'] = df['sales'].shift(lag)
    return df

def add_rolling_features(df, window=7):
    df[f'rolling_mean_{window}'] = df['sales'].rolling(window=window).mean()
    return df

def preprocess_data(df):
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = df.dropna()
    return df
