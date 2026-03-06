import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from model import run_forecast_for_product
from data_loader import load_data
from db_manager import get_connection, get_inventory_settings, update_inventory_settings

st.title("BizForecast — Sales Demand Forecasting System")

df = load_data("../data/sales_data.csv")
products = sorted(df["product_id"].unique())

product = st.selectbox("Select Product", products)

conn = get_connection()
inv = get_inventory_settings(conn, product)

if inv is None:
    current_inventory = 120
    lead_time_days = 7
    service_level = 0.95
    target_days = 14
else:
    current_inventory = inv["current_inventory"]
    lead_time_days = inv["lead_time_days"]
    service_level = inv["service_level"]
    target_days = inv["target_days"]

st.subheader("Inventory Controls")

new_inventory = st.number_input("Current Inventory", min_value=0, value=int(current_inventory))
new_lead_time = st.number_input("Lead Time (days)", min_value=1, value=int(lead_time_days))
new_service_level = st.selectbox(
    "Service Level",
    [0.90, 0.95, 0.97, 0.99],
    index=[0.90, 0.95, 0.97, 0.99].index(float(service_level))
)
new_target_days = st.number_input("Target Days", min_value=1, value=int(target_days))

if st.button("Update Inventory Settings"):
    update_inventory_settings(
        conn,
        product,
        int(new_inventory),
        int(new_lead_time),
        float(new_service_level),
        int(new_target_days)
    )
    st.success("Inventory settings updated.")

if st.button("Run Forecast"):

    result = run_forecast_for_product(product)

    comparison_rows = [
        {
            "Model": "Regression",
            "MAE": result["mae"],
            "RMSE": result["rmse"]
        },
        {
            "Model": "Naive Baseline",
            "MAE": result["naive_mae"],
            "RMSE": result["naive_rmse"]
        }
    ]

    if result.get("sarima_mae") is not None and result.get("sarima_rmse") is not None:
        comparison_rows.append(
            {
                "Model": "SARIMA",
                "MAE": result["sarima_mae"],
                "RMSE": result["sarima_rmse"]
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    best_model_row = comparison_df.loc[comparison_df["MAE"].idxmin()]
    best_model_name = best_model_row["Model"]

    st.subheader("Model Performance")

    col1, col2 = st.columns(2)
    col1.metric("Regression MAE", f"{result['mae']:.2f}")
    col2.metric("Regression RMSE", f"{result['rmse']:.2f}")

    st.subheader("Model Comparison")

    st.dataframe(comparison_df, use_container_width=True)

    st.success(f"Best model for this product: {best_model_name}")

    st.subheader("Inventory Recommendation")

    col3, col4 = st.columns(2)
    col3.metric("Average Daily Demand", f"{result['avg_demand']:.2f}")
    col4.metric("Safety Stock", f"{result['safety_stock']:.2f}")

    col5, col6 = st.columns(2)
    col5.metric("Reorder Point", f"{result['rop']:.2f}")
    col6.metric("Recommended Order Qty", f"{result['order_qty']:.2f}")

    st.write(f"**Current Inventory:** {result['current_inventory']}")

    if result["status"] == "REORDER NOW":
        st.error("⚠ REORDER REQUIRED")
    else:
        st.success("✅ Inventory OK")

    st.subheader("Forecast Visualization")

    fig, ax = plt.subplots()

    ax.plot(result["dates"], result["actual"], label="Actual")
    ax.plot(result["dates"], result["prediction"], label="Forecast")

    ax.legend()
    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.subheader("Forecast Table")

    forecast_df = pd.DataFrame({
        "Date": result["dates"].values,
        "Actual Sales": result["actual"].values,
        "Predicted Sales": result["prediction"]
    })

    st.dataframe(forecast_df, use_container_width=True)

    st.subheader("Historical Sales Data")

    history_df = result["processed_df"][["date", "sales"]].copy()
    history_df.columns = ["Date", "Sales"]

    st.dataframe(history_df, use_container_width=True)

    st.subheader("Download Report")

    report_df = forecast_df.copy()
    csv = report_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Forecast Report as CSV",
        data=csv,
        file_name=f"{product}_forecast_report.csv",
        mime="text/csv"
    )

conn.close()