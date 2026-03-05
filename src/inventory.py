import numpy as np

# INPUTS :
# Forecasted daily demand for the next N days (from regression)
# Lead time L (days) — time it takes to restock
# Current inventory I
# Service level (e.g., 95%)
# Demand uncertainty σ (we’ll approximate it from forecast errors)

# OUTPUTS:
# Safety stock
# Reorder point (ROP)
# Recommended reorder quantity (simple version)
# Status: OK / REORDER NOW

def z_value (service_level : float) -> float:
    mapping = {
        0.90: 1.28,
        0.95: 1.65,
        0.97: 1.88,
        0.99: 2.33
    }

    return mapping.get(service_level, 1.65) # Returns 95% as default service level

def safety_stock (demand_std: float, lead_time_days: int, service_level: float = 0.95) -> float:
    # Safety stock = Z * sigma * sqrt(L)

    Z = z_value(service_level)
    return Z * demand_std * np.sqrt(lead_time_days)

def reorder_point (avg_daily_demand: float, lead_time_days: int, safety_stock_units: float) -> float:
    # Reorder point = avg_demand * L + safety_stock
    return (avg_daily_demand * lead_time_days) + safety_stock_units

# POLICY : Order enough to cover target_days of demand + safety stock.
def recommended_order_quantity(
        target_days: int,
        avg_daily_demand: float,
        current_inventory: float,
        safety_stock_units: float
) -> float:
    target_stock_level = (avg_daily_demand * target_days) + safety_stock_units
    qty = target_stock_level - current_inventory
    return max(0.0, qty)

def inventory_decision( current_inventory: float, rop: float) -> str:
    return "REORDER NOW" if current_inventory <= rop else "OK"