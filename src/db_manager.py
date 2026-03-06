import os
import mysql.connector
from mysql.connector import Error


def get_connection():
    return mysql.connector.connect(
        host=os.getenv("BIZ_DB_HOST", "localhost"),
        user=os.getenv("BIZ_DB_USER", "aditya"),
        password=os.getenv("BIZ_DB_PASSWORD", ""),
        database=os.getenv("BIZ_DB_NAME", "bizforecast"),
        port=int(os.getenv("BIZ_DB_PORT", "3306")),
    )


def upsert_product(conn, product_id: str, product_name: str | None = None):
    q = """
    INSERT INTO products (product_id, product_name)
    VALUES (%s, %s)
    ON DUPLICATE KEY UPDATE product_name = COALESCE(VALUES(product_name), product_name)
    """
    with conn.cursor() as cur:
        cur.execute(q, (product_id, product_name))


def upsert_sale(conn, product_id: str, sale_date, sales: int):
    q = """
    INSERT INTO sales (product_id, sale_date, sales)
    VALUES (%s, %s, %s)
    ON DUPLICATE KEY UPDATE sales = VALUES(sales)
    """
    with conn.cursor() as cur:
        cur.execute(q, (product_id, sale_date, sales))


def ensure_inventory_row(conn, product_id: str,
                         current_inventory: int = 120,
                         lead_time_days: int = 7,
                         service_level: float = 0.95,
                         target_days: int = 14):

    q = """
    INSERT INTO inventory (product_id, current_inventory, lead_time_days, service_level, target_days)
    VALUES (%s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE product_id = product_id
    """
    with conn.cursor() as cur:
        cur.execute(q, (product_id, current_inventory, lead_time_days, service_level, target_days))


def get_inventory_settings(conn, product_id: str):
    
    q = """
    SELECT current_inventory, lead_time_days, service_level, target_days
    FROM inventory
    WHERE product_id = %s
    """
    with conn.cursor(dictionary=True) as cur:
        cur.execute(q, (product_id,))
        row = cur.fetchone()
    return row


def import_sales_csv(csv_path: str):
    
    import pandas as pd

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if "product_id" not in df.columns:
        df["product_id"] = "P001"

    conn = get_connection()
    try:
        conn.start_transaction()

        for _, r in df.iterrows():
            pid = str(r["product_id"])
            upsert_product(conn, pid)
            ensure_inventory_row(conn, pid)  # default row if missing
            upsert_sale(conn, pid, r["date"], int(r["sales"]))

        conn.commit()
        print(f"Imported {len(df)} rows into MySQL.")
    except Error as e:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    import_sales_csv("../data/sales_data.csv")