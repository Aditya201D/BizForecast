import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df["product_id"] = df["product_id"].astype(str)
    return df
