import pandas as pd
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="DataFrameGroupBy.apply operated on the grouping columns"
)

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

def preprocess_one_product(df_product):
    df_product = df_product.sort_values("date").copy()
    df_product = add_time_features(df_product)
    df_product = add_lag_features(df_product)
    df_product = add_rolling_features(df_product)
    df_product = df_product.dropna()
    return df_product

def preprocess_all_products(df):
    df = df[["date", "product_id", "sales"]].copy()
    return (
        df.groupby("product_id", group_keys=False)
          .apply(preprocess_one_product)
          .reset_index(drop=True)
    )

def preprocess_data(df):
    """
    Backwards compatible:
    - If df has product_id -> preprocess each product separately
    - Else -> treat df as a single product time series
    """
    if "product_id" in df.columns:
        return preprocess_all_products(df)
    else:
        return preprocess_one_product(df)