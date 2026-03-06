import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from model import run_forecast_for_product
from data_loader import load_data

st.title("BizForecast — Sales Demand Forecasting System")

df = load_data("../data/sales_data.csv")

products = sorted(df["product_id"].unique())

product = st.selectbox("Select Product", products)

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