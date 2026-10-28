"""Build the project SQLite database from cleaned CSV files."""

from pathlib import Path
import sqlite3
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
STATE_DIR = PROJECT_ROOT / "state"
DB_PATH = STATE_DIR / "instamart.db"

TABLES = [
    "customers",
    "products",
    "orders",
    "tickets",
    "policy_eval",
    "agent_scenarios",
]


def main() -> None:
    missing = [name for name in TABLES if not (CLEAN_DIR / f"{name}.csv").exists()]
    if missing:
        raise FileNotFoundError(
            "Missing cleaned CSV files: "
            + ", ".join(missing)
            + ". Run clean_data.py first."
        )

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("PRAGMA foreign_keys = ON")

        # Rebuild tables on each run so the database matches the current CSVs.
        for table in TABLES:
            frame = pd.read_csv(CLEAN_DIR / f"{table}.csv")
            frame.to_sql(table, connection, if_exists="replace", index=False)
            print(f"Loaded {len(frame):,} rows into {table}")

        connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_orders_customer
                ON orders(customer_id);
            CREATE INDEX IF NOT EXISTS idx_orders_product
                ON orders(product_id);
            CREATE INDEX IF NOT EXISTS idx_orders_timestamp
                ON orders(order_ts);
            CREATE INDEX IF NOT EXISTS idx_tickets_customer
                ON tickets(customer_id);
            CREATE INDEX IF NOT EXISTS idx_tickets_order
                ON tickets(order_id);
            """
        )

        tables = connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        ).fetchall()

    print(f"\nDatabase created: {DB_PATH}")
    print("Tables:", ", ".join(row[0] for row in tables))


if __name__ == "__main__":
    main()