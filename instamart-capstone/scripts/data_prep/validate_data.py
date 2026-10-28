"""Validate cleaned data and report key integrity issues."""

from pathlib import Path
import sqlite3
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
DB_PATH = PROJECT_ROOT / "state" / "instamart.db"


def load(name: str) -> pd.DataFrame:
    path = CLEAN_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run clean_data.py before validation."
        )
    return pd.read_csv(path)


def check_unique(df: pd.DataFrame, column: str, label: str) -> list[str]:
    issues = []
    duplicates = int(df[column].duplicated().sum())
    if duplicates:
        issues.append(f"{label}: {duplicates} duplicate {column} values")
    return issues


def check_foreign_key(
    child: pd.DataFrame,
    child_column: str,
    parent: pd.DataFrame,
    parent_column: str,
    label: str,
) -> list[str]:
    child_values = set(child[child_column].dropna().astype(str))
    parent_values = set(parent[parent_column].dropna().astype(str))
    missing = child_values - parent_values
    if missing:
        examples = ", ".join(sorted(missing)[:5])
        return [f"{label}: {len(missing)} unmatched values (examples: {examples})"]
    return []


def main() -> None:
    customers = load("customers.csv")
    products = load("products.csv")
    orders = load("orders.csv")
    tickets = load("tickets.csv")

    issues = []
    issues += check_unique(customers, "customer_id", "Customers")
    issues += check_unique(products, "product_id", "Products")
    issues += check_unique(orders, "order_id", "Orders")
    issues += check_unique(tickets, "ticket_id", "Tickets")

    issues += check_foreign_key(
        orders, "customer_id", customers, "customer_id", "Orders → customers"
    )
    issues += check_foreign_key(
        orders, "product_id", products, "product_id", "Orders → products"
    )
    issues += check_foreign_key(
        tickets, "customer_id", customers, "customer_id", "Tickets → customers"
    )
    issues += check_foreign_key(
        tickets, "order_id", orders, "order_id", "Tickets → orders"
    )

    if (orders["quantity"] <= 0).any():
        issues.append("Orders: quantity must be greater than zero")
    if (orders["order_amount"] < 0).any():
        issues.append("Orders: order_amount contains negative values")
    if not orders["discount_pct"].dropna().between(0, 100).all():
        issues.append("Orders: discount_pct must be between 0 and 100")
    if not products["price_inr"].gt(0).all():
        issues.append("Products: price_inr must be greater than zero")

    if len(customers) < 500 or len(customers.columns) < 5:
        issues.append("Part A minimum not met: customers must have >=500 rows and >=5 columns")
    if not set(customers["gender"].dropna().unique()).issubset(
        {"Male", "Female", "Unknown"}
    ):
        issues.append("Customers: gender values are not standardized")
    if not set(customers["city"].dropna().unique()).issubset(
        {"Chennai", "Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Pune", "Unknown"}
    ):
        issues.append("Customers: city values are not standardized")

    # Validate the persisted SQLite copy independently of the Pandas checks.
    if not DB_PATH.exists():
        issues.append(f"SQLite database missing: {DB_PATH}; run build_database.py first")
    else:
        with sqlite3.connect(DB_PATH) as connection:
            expected_counts = {
                "customers": len(customers),
                "products": len(products),
                "orders": len(orders),
                "tickets": len(tickets),
            }
            for table, expected in expected_counts.items():
                actual = connection.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
                if actual != expected:
                    issues.append(
                        f"SQLite {table}: {actual} rows, expected {expected}"
                    )

            sql_checks = {
                "orders with unknown customer": """
                    SELECT COUNT(*) FROM orders o
                    LEFT JOIN customers c ON c.customer_id = o.customer_id
                    WHERE c.customer_id IS NULL
                """,
                "orders with unknown product": """
                    SELECT COUNT(*) FROM orders o
                    LEFT JOIN products p ON p.product_id = o.product_id
                    WHERE p.product_id IS NULL
                """,
                "tickets with unknown customer": """
                    SELECT COUNT(*) FROM tickets t
                    LEFT JOIN customers c ON c.customer_id = t.customer_id
                    WHERE c.customer_id IS NULL
                """,
            }
            for label, query in sql_checks.items():
                count = connection.execute(query).fetchone()[0]
                if count:
                    issues.append(f"SQLite: {count} {label}")

    print("Cleaned data validation")
    print(f"Customers: {len(customers):,}")
    print(f"Products:  {len(products):,}")
    print(f"Orders:    {len(orders):,}")
    print(f"Tickets:   {len(tickets):,}")
    print("Pandas checks: completed")
    print("SQLite row-count and join checks: completed" if DB_PATH.exists() else "SQLite checks: database missing")

    if issues:
        print(f"\nFound {len(issues)} issue(s):")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)

    print("\nAll validation checks passed.")


if __name__ == "__main__":
    main()
