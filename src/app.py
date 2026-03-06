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
new_service_level = st.selectbox("Service Level", [0.90, 0.95, 0.97, 0.99],
                                 index=[0.90, 0.95, 0.97, 0.99].index(float(service_level)))
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

    st.subheader("Model Performance")
    st.write(f"MAE: {result['mae']:.2f}")
    st.write(f"RMSE: {result['rmse']:.2f}")

    st.subheader("Inventory Recommendation")

    st.write(f"Average Daily Demand: {result['avg_demand']:.2f}")
    st.write(f"Safety Stock: {result['safety_stock']:.2f}")
    st.write(f"Reorder Point: {result['rop']:.2f}")
    st.write(f"Current Inventory: {result['current_inventory']}")
    st.write(f"Recommended Order Quantity: {result['order_qty']:.2f}")

    if result["status"] == "REORDER NOW":
        st.error("⚠ REORDER REQUIRED")
    else:
        st.success("Inventory OK")

    st.subheader("Forecast Visualization")

    fig, ax = plt.subplots()

    ax.plot(result["dates"], result["actual"], label="Actual")
    ax.plot(result["dates"], result["prediction"], label="Forecast")

    ax.legend()
    plt.xticks(rotation=45)

    st.pyplot(fig)

conn.close()