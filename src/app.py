import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from model import run_forecast_for_product
from data_loader import load_data
from db_manager import get_connection, get_inventory_settings, update_inventory_settings

from db_manager import (
    get_connection,
    get_inventory_settings,
    update_inventory_settings,
    insert_forecast_result,
    get_recent_forecast_results
)

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="BizForecast",
    page_icon="📈",
    layout="wide"
)


# -----------------------------
# Load custom CSS
# -----------------------------
def load_css():
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>BizForecast</h1>
        <p>AI-based demand forecasting and inventory recommendation for small businesses</p>
    </div>
    """,
    unsafe_allow_html=True
)

df = load_data("../data/sales_data.csv")
products = sorted(df["product_id"].unique())

top_left, top_right = st.columns([2, 1])

with top_left:
    st.markdown("### Product Selection")
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

with top_right:
    st.markdown("### ⚙ Inventory Controls")
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

run_clicked = st.button("Run Forecast", use_container_width=True)

if run_clicked:
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

    # -----------------------------
    # Save forecast result to DB
    # -----------------------------
    conn_save = get_connection()
    insert_forecast_result(
        conn_save,
        product_id=product,
        regression_mae=float(result["mae"]),
        regression_rmse=float(result["rmse"]),
        naive_mae=float(result["naive_mae"]),
        naive_rmse=float(result["naive_rmse"]),
        sarima_mae=float(result["sarima_mae"]) if result.get("sarima_mae") is not None else None,
        sarima_rmse=float(result["sarima_rmse"]) if result.get("sarima_rmse") is not None else None,
        best_model=best_model_name,
        avg_demand=float(result["avg_demand"]),
        safety_stock=float(result["safety_stock"]),
        reorder_point=float(result["rop"]),
        current_inventory=int(result["current_inventory"]),
        recommended_order_qty=float(result["order_qty"]),
        status=result["status"]
    )
    conn_save.close()

    # -----------------------------
    # Recent Forecast History
    # -----------------------------
    st.markdown("## 🕘 Recent Forecast History")

    conn_hist = get_connection()
    history_rows = get_recent_forecast_results(conn_hist, product, limit=5)
    conn_hist.close()

    if history_rows:
        history_runs_df = pd.DataFrame(history_rows)
        st.dataframe(history_runs_df, use_container_width=True)
    else:
        st.info("No saved forecast history yet for this product.")

    # -----------------------------
    # KPI cards
    # -----------------------------
    st.markdown("---")
    st.markdown("## Key Metrics")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best Model", best_model_name)
    k2.metric("Regression MAE", f"{result['mae']:.2f}")
    k3.metric("Current Inventory", f"{result['current_inventory']}")
    k4.metric("Reorder Point", f"{result['rop']:.2f}")

    # -----------------------------
    # Chart + Inventory Summary
    # -----------------------------
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.markdown("##  Forecast Visualization")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            result["dates"],
            result["actual"],
            label="Actual",
            color="#22c55e",
            linewidth=2.2
        )
        ax.plot(
            result["dates"],
            result["prediction"],
            label="Forecast",
            color="#f59e0b",
            linewidth=2.2
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig)

    with right_col:
        st.markdown("##  Inventory Recommendation")

        c1, c2 = st.columns(2)
        c1.metric("Avg Daily Demand", f"{result['avg_demand']:.2f}")
        c2.metric("Safety Stock", f"{result['safety_stock']:.2f}")

        c3, c4 = st.columns(2)
        c3.metric("Reorder Point", f"{result['rop']:.2f}")
        c4.metric("Order Qty", f"{result['order_qty']:.2f}")

        st.write(f"**Current Inventory:** {result['current_inventory']}")

        if result["status"] == "REORDER NOW":
            st.error("⚠ Reorder is recommended immediately.")
        else:
            st.success("✅ Inventory level is currently safe.")

    # -----------------------------
    # Comparison + Business Summary
    # -----------------------------
    st.markdown("---")
    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.markdown("## 🤖 Model Comparison")
        st.dataframe(comparison_df, use_container_width=True)

    with col_b:
        st.markdown("## 🧾 Business Summary")

        inventory_gap = result["current_inventory"] - result["rop"]

        st.info(
            f"""
**Selected Product:** {product}

**Best-performing model:** {best_model_name}

**Forecasted average demand:** {result['avg_demand']:.2f} units/day

**Inventory position:** {'Below reorder point' if result['status'] == 'REORDER NOW' else 'Above reorder point'}

**Recommended action:** {'Place a replenishment order' if result['status'] == 'REORDER NOW' else 'Monitor inventory and continue operations'}
"""
        )

        if inventory_gap < 0:
            st.warning(f"Inventory is below the reorder point by {abs(inventory_gap):.2f} units.")
        else:
            st.success(f"Inventory is above the reorder point by {inventory_gap:.2f} units.")

    # -----------------------------
    # Forecast table
    # -----------------------------
    st.markdown("---")
    st.markdown("## 📋 Forecast Table")

    forecast_df = pd.DataFrame({
        "Date": result["dates"].values,
        "Actual Sales": result["actual"].values,
        "Predicted Sales": result["prediction"]
    })

    st.dataframe(forecast_df, use_container_width=True)

    # -----------------------------
    # Historical sales table
    # -----------------------------
    st.markdown("## 🗂 Historical Sales Data")

    history_df = result["processed_df"][["date", "sales"]].copy()
    history_df.columns = ["Date", "Sales"]

    st.dataframe(history_df, use_container_width=True)

    # -----------------------------
    # Download report
    # -----------------------------
    st.markdown("## ⬇ Download Report")

    report_df = forecast_df.copy()
    csv = report_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Forecast Report as CSV",
        data=csv,
        file_name=f"{product}_forecast_report.csv",
        mime="text/csv",
        use_container_width=True
    )

conn.close()