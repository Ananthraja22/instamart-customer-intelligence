"""Run summary analyses against the project SQLite database."""

from pathlib import Path
import sqlite3
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "state" / "instamart.db"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

QUERIES = {
    "order_status_summary": """
        SELECT status,
               COUNT(*) AS order_count,
               ROUND(SUM(order_amount), 2) AS total_order_amount,
               ROUND(AVG(order_amount), 2) AS average_order_amount
        FROM orders
        GROUP BY status
        ORDER BY order_count DESC
    """,
    "monthly_orders": """
        SELECT substr(order_ts, 1, 7) AS order_month,
               COUNT(*) AS order_count,
               ROUND(SUM(order_amount), 2) AS total_order_amount,
               ROUND(AVG(order_amount), 2) AS average_order_amount
        FROM orders
        GROUP BY order_month
        ORDER BY order_month
    """,
    "top_product_categories": """
        SELECT p.category,
               COUNT(*) AS completed_order_count,
               SUM(o.quantity) AS units_sold,
               ROUND(SUM(o.order_amount), 2) AS sales_inr
        FROM orders AS o
        JOIN products AS p ON p.product_id = o.product_id
        WHERE lower(o.status) = 'completed'
        GROUP BY p.category
        ORDER BY sales_inr DESC
    """,
    "top_customers": """
        SELECT c.customer_id,
               c.city,
               COUNT(o.order_id) AS order_count,
               ROUND(SUM(o.order_amount), 2) AS spend_inr
        FROM customers AS c
        JOIN orders AS o ON o.customer_id = c.customer_id
        WHERE lower(o.status) = 'completed'
        GROUP BY c.customer_id, c.city
        ORDER BY spend_inr DESC
        LIMIT 20
    """,
    "late_completed_orders": """
        SELECT COUNT(*) AS eligible_late_orders,
               ROUND(SUM(order_amount), 2) AS potential_refund_inr,
               ROUND(AVG(delivery_minutes), 1) AS average_delivery_minutes
        FROM orders
        WHERE lower(status) = 'completed'
          AND delivery_minutes > 45
    """,
    "ticket_categories": """
        SELECT lower(true_category) AS category,
               COUNT(*) AS ticket_count
        FROM tickets
        GROUP BY lower(true_category)
        ORDER BY ticket_count DESC
    """,
}


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"{DB_PATH} not found. Run build_database.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as connection:
        for name, query in QUERIES.items():
            result = pd.read_sql_query(query, connection)
            output_path = OUTPUT_DIR / f"{name}.csv"
            result.to_csv(output_path, index=False)

            print(f"\n{name.replace('_', ' ').title()}")
            if result.empty:
                print("(no rows)")
            else:
                print(result.to_string(index=False))
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
