import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_loader import load_data
from preprocessing import preprocess_data
from arima_model import train_sarima, forecast_sarima
from inventory import safety_stock, reorder_point, recommended_order_quantity, inventory_decision
from db_manager import get_connection, get_inventory_settings, ensure_inventory_row

split_ratio = 0.8

lead_time_days = 7
service_level = 0.95
target_days = 14

DEFAULT_CURRENT_INVENTORY = 120 # Temporary pplaceholder

# SARIMA Model is expensive and heavy to use for multiple products with no useful result.
# So, we use it to demonstrate its result for only ONE product, then set it to None.

SARIMA_product_id = "P003"
sarima_order = (2,1,2)
seasonal_order = (1,0,0,7)

SELECTED_PRODUCT_ID = None

def _eval (y_true, y_prediction):
    mae = mean_absolute_error(y_true, y_prediction)
    rmse = np.sqrt(mean_squared_error(y_true , y_prediction))
    return mae, rmse

# Loading the raw, generated data
df_raw = load_data("../data/sales_data.csv")

# If input dataset has only one product, add product_id tag to it to keep the code running
if "product_id" not in df_raw.columns:
    df_raw["product_id"] = "P001"

all_product_ids = sorted(df_raw["product_id"].unique())

if SELECTED_PRODUCT_ID is not None:
    if SELECTED_PRODUCT_ID not in all_product_ids:
        raise ValueError(f"Product {SELECTED_PRODUCT_ID} not found in dataset.")
    product_ids = [SELECTED_PRODUCT_ID]
else:
    product_ids = all_product_ids

print("\n===== MULTI-PRODUCT RUN =====")
print(f"Products found: {product_ids}\n")

summary_rows = []

def main():
    for pid in product_ids:
        print(f"\n==================== Product: {pid} ====================")
        
        #Slicing out one single product
        df_raw_p = df_raw[df_raw["product_id"] == pid].copy()

        #Regression has its own dataframe to include feature
        #engineering, ARIMA dataframe won't need it
        df_reg = preprocess_data(df_raw_p.copy())

        if len(df_reg) < 30:
            print("Not enough data after preprocessing. Skipping...")
            continue

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

        reg_mae, reg_rmse = _eval(Y_test, reg_predictions)

        # INVENTORY SYSTEM USING REGRESSION MODEL:

        # Pull per-product settings from DB (fallback to defaults)
        conn = get_connection()
        try:
            ensure_inventory_row(conn, pid, current_inventory=DEFAULT_CURRENT_INVENTORY,
                                lead_time_days=lead_time_days,
                                service_level=service_level,
                                target_days=target_days)
            inv_row = get_inventory_settings(conn, pid)
        finally:
            conn.close()

        if inv_row:
            current_inventory = int(inv_row["current_inventory"])
            lead_time_days_local = int(inv_row["lead_time_days"])
            service_level_local = float(inv_row["service_level"])
            target_days_local = int(inv_row["target_days"])
        else:
            current_inventory = DEFAULT_CURRENT_INVENTORY
            lead_time_days_local = lead_time_days
            service_level_local = service_level
            target_days_local = target_days

        avg_daily_demand = float(np.mean(reg_predictions))
        residuals = Y_test.values - reg_predictions
        demand_std = float(np.std(residuals))

        ss = safety_stock(demand_std, lead_time_days_local, service_level_local)
        rop = reorder_point(avg_daily_demand, lead_time_days_local, ss)
        order_qty = recommended_order_quantity(target_days_local, avg_daily_demand, current_inventory, ss)
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
        naive_mae, naive_rmse = _eval(Y_test, naive_predictions)


        # SARIMA Model Implementation
        #we use the same date range as the regression model

        sarima_prediction = None
        sarima_mae, sarima_rmse = None, None
        min_len = None

        if SARIMA_product_id is not None and pid == SARIMA_product_id:
            series = (
                df_raw_p[["date", "sales"]]
                .groupby("date", as_index=True)["sales"]
                .sum()                 
                .sort_index()
            )
            series = series.asfreq("D")

            split_date = test_dates.iloc[0]

            train_series = series[series.index < split_date]
            test_series = series[series.index >= split_date]

            if len(train_series) >= 50 and len(test_series) >=5:
                sarima_model = train_sarima(
                    train_series,
                    order = sarima_order,
                    seasonal_order= seasonal_order
                )

                sarima_prediction = forecast_sarima(sarima_model, steps = len(test_series))

                min_len = min(len(test_series), len(sarima_prediction))
                sarima_true = test_series.values[:min_len]
                sarima_prediction = sarima_prediction[:min_len]

                sarima_mae, sarima_rmse = _eval(sarima_true, sarima_prediction)
            else:
                print("\n SARIMA skipped. Not enough history for stable evaluation.")

        print("\n +++++ Model Comparison +++++")
        print("------Regression-----")
        print("MAE: ", round(reg_mae, 2))
        print("RMSE: ", round(reg_rmse, 2))

        print("------Naive Baseline Model------")
        print("MAE: ", round(naive_mae, 2))
        print("RMSE: ", round(naive_rmse, 2))

        if sarima_mae is not None:
            print("------SARIMA Model--------")
            print("MAE: ", round(sarima_mae, 2))
            print("RMSE: ", round(sarima_rmse, 2))

        # PLOTTING:

        if SELECTED_PRODUCT_ID is not None and pid == SELECTED_PRODUCT_ID:
            # Actual values from csv file used to plot
            actual = Y_test.values

            plt.figure(figsize=(12 , 6))

            plt.plot(test_dates, actual, label = "Actual")
            plt.plot(test_dates, reg_predictions, label = "Regression Model Predictions")
            plt.plot(test_dates, naive_predictions, label = "Baseline Model Predictions")
            
            if sarima_prediction is not None and min_len is not None:
                plt.plot(test_dates.values[:min_len], sarima_prediction, label = f"ARIMA {sarima_order} x {seasonal_order} prediction")

            
            plt.title(f"Model Comparison for Product {pid}")
            plt.xlabel("Date")
            plt.ylabel("Sales")
            plt.xticks(rotation = 45)
            plt.legend()
            plt.tight_layout()
            plt.show()

        summary_rows.append({
            "product_id": pid,
            "reg_mae": reg_mae,
            "naive_mae": naive_mae,
            "status": status,
            "rop": rop,
            "order_qty": order_qty
        })

if __name__ == "__main__":
    main()

# FINAL SUMMARY

print("\n---------SUMMARY (Sorted by Regression MAE)---------")
summary_rows = sorted(summary_rows, key=lambda r: r["reg_mae"])
for r in summary_rows:
    print(
        f"{r['product_id']} | reg_MAE={r['reg_mae']:.2f} | naive_MAE={r['naive_mae']:.2f} "
        f"| {r['status']} | ROP={r['rop']:.1f} | order={r['order_qty']:.1f}"
    )


# DASHBOARD

def run_forecast_for_product(product_id):
    """
    Runs forecasting + inventory pipeline for a single product.
    Returns dictionary with results.
    """

    df = load_data("../data/sales_data.csv")
    df = df[df["product_id"] == product_id].copy()

    df_reg = preprocess_data(df)

    features = ["day_of_week", "month", "lag_1", "lag_7", "rolling_mean_7"]

    X = df_reg[features]
    y = df_reg["sales"]

    split_index = int(len(df_reg) * split_ratio)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    test_dates = df_reg["date"][split_index:]

    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)

    reg_predictions = reg_model.predict(X_test)

    reg_mae = mean_absolute_error(y_test, reg_predictions)
    reg_rmse = np.sqrt(mean_squared_error(y_test, reg_predictions))

    naive_predictions = df_reg["lag_1"][split_index:].values
    naive_mae = mean_absolute_error(y_test, naive_predictions)
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_predictions))

    avg_daily_demand = float(np.mean(reg_predictions))
    residuals = y_test.values - reg_predictions
    demand_std = float(np.std(residuals))

    conn = get_connection()
    inv = get_inventory_settings(conn, product_id)
    conn.close()

    current_inventory = inv["current_inventory"]
    lead_time = inv["lead_time_days"]
    service_level = inv["service_level"]
    target_days = inv["target_days"]

    ss = safety_stock(demand_std, lead_time, service_level)
    rop = reorder_point(avg_daily_demand, lead_time, ss)
    order_qty = recommended_order_quantity(target_days, avg_daily_demand, current_inventory, ss)
    status = inventory_decision(current_inventory, rop)

    return {
        "processed_df": df_reg,
        "dates": test_dates,
        "actual": y_test,
        "prediction": reg_predictions,
        "mae": reg_mae,
        "rmse": reg_rmse,
        "avg_demand": avg_daily_demand,
        "safety_stock": ss,
        "rop": rop,
        "current_inventory": current_inventory,
        "order_qty": order_qty,
        "status": status
    }