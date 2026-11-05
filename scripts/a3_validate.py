"""Part A, task 5: assert the reference checkpoints against BOTH the clean CSVs and the SQLite database.
Fails loudly on the first broken checkpoint; the last line printed on success is exactly
ALL CHECKPOINTS PASSED
"""
import re
import sqlite3

import pandas as pd

from common import CLEAN, DB


def main():
    cust = pd.read_csv(f"{CLEAN}/customers.csv")
    orders = pd.read_csv(f"{CLEAN}/orders.csv")
    products = pd.read_csv(f"{CLEAN}/products.csv")
    completed = orders[orders["status"] == "completed"]
    cancelled = orders[orders["status"] == "cancelled"]

    # Task 3: customers
    assert cust["city"].value_counts().to_dict() == {
        "Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}, "city counts"
    assert cust["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}, "gender counts"
    assert cust["age"].isna().sum() == 0, "missing ages remain"
    assert cust["age"].between(18, 60).all(), "age outside 18-60"
    assert pd.to_datetime(cust["signup_date"], format="%Y-%m-%d").notna().all(), "signup_date not parseable"

    # Task 4: orders
    assert len(orders) == 12309, "order row count"
    assert orders["order_id"].is_unique, "order_id not unique"
    assert len(completed) == 11110, "completed count"
    assert len(cancelled) == 1199, "cancelled count"
    assert not (orders["status"] == "pending").any(), "pending rows remain"
    assert orders["order_amount"].isna().sum() == 0, "missing order_amount"
    assert completed["delivery_minutes"].isna().sum() == 0, "completed order missing delivery"
    assert cancelled["delivery_minutes"].isna().sum() == 1199, "cancelled order was given a delivery time"
    assert (completed["delivery_minutes"] > 0).all(), "non-positive delivery time remains"
    assert round(completed["order_amount"].sum(), 2) == 2998668.80, "completed revenue"

    # Task 5: the database agrees with the CSVs and uses the required timestamp text format
    con = sqlite3.connect(DB)
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"customers", "products", "orders"} <= tables, "missing database table"
        for table, n in [("customers", len(cust)), ("products", len(products)), ("orders", len(orders))]:
            assert con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == n, f"{table} row count in database"
        revenue = con.execute("SELECT ROUND(SUM(order_amount), 2) FROM orders WHERE status='completed'").fetchone()[0]
        assert revenue == 2998668.80, "completed revenue in database"
        sample = [r[0] for r in con.execute("SELECT order_ts FROM orders")]
        assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", ts) for ts in sample), "order_ts text format"
    finally:
        con.close()

    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
