"""Create cleaned CSV files while preserving the original raw data."""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"

CITY_ALIASES = {
    "chennai": "Chennai",
    "madras": "Chennai",
    "mumbai": "Mumbai",
    "bombay": "Mumbai",
    "bengaluru": "Bengaluru",
    "bangalore": "Bengaluru",
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "hyderabad": "Hyderabad",
    "pune": "Pune",
}


def normalize_city(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    key = str(value).strip().casefold()
    return CITY_ALIASES.get(key, "Unknown")


def normalize_gender(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    key = str(value).strip().casefold()
    if key in {"m", "male", "man"}:
        return "Male"
    if key in {"f", "female", "woman"}:
        return "Female"
    return "Unknown"


def read_csv(name: str) -> pd.DataFrame:
    path = RAW_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return pd.read_csv(path)


def save_csv(frame: pd.DataFrame, name: str) -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    path = CLEAN_DIR / name
    frame.to_csv(path, index=False)
    print(f"Saved {len(frame):,} rows to {path}")


def clean_customers() -> None:
    df = read_csv("customers.csv")
    df["customer_id"] = df["customer_id"].astype("string").str.strip()
    df["city"] = df["city"].map(normalize_city)
    df["gender"] = df["gender"].map(normalize_gender)
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    if df["age"].notna().any():
        df["age"] = df["age"].fillna(df["age"].median())

    df = df.dropna(subset=["customer_id"]).drop_duplicates("customer_id", keep="first")
    df["signup_date"] = df["signup_date"].dt.strftime("%Y-%m-%d")
    save_csv(df, "customers.csv")


def clean_products() -> None:
    df = read_csv("products.csv")
    df["product_id"] = df["product_id"].astype("string").str.strip()
    df["product_name"] = df["product_name"].astype("string").str.strip()
    df["category"] = df["category"].astype("string").str.strip().str.title()
    df["price_inr"] = pd.to_numeric(df["price_inr"], errors="coerce")
    df = df.dropna(subset=["product_id", "price_inr"])
    df = df[df["price_inr"] > 0]
    df = df.drop_duplicates("product_id", keep="first")
    save_csv(df, "products.csv")


def clean_orders() -> None:
    df = read_csv("orders.csv")

    for column in ["order_id", "customer_id", "product_id"]:
        df[column] = df[column].astype("string").str.strip()

    df["status"] = df["status"].astype("string").str.strip().str.lower()
    df["order_ts"] = pd.to_datetime(df["order_ts"], errors="coerce")

    for column in ["quantity", "discount_pct", "order_amount", "delivery_minutes"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Confirm that repeated rows describe status updates for one product order.
    varying_products = (
        df.groupby("order_id")["product_id"]
        .nunique(dropna=False)
        .gt(1)
    )
    if varying_products.any():
        examples = ", ".join(varying_products[varying_products].index.astype(str)[:5])
        raise ValueError(
            "Some order IDs contain multiple product IDs. "
            f"Inspect these before collapsing snapshots: {examples}"
        )

    # Remove only exact duplicate snapshots.
    df = df.dropna(
        subset=["order_id", "customer_id", "product_id", "order_ts", "quantity", "order_amount"]
    )
    df = df.drop_duplicates()

    # Keep the latest snapshot's status and delivery details, but retain the
    # earliest timestamp as the order placement time.
    df = df.sort_values(["order_id", "order_ts"], kind="stable")
    first_order_ts = df.groupby("order_id")["order_ts"].min()
    df = df.drop_duplicates("order_id", keep="last").set_index("order_id")
    df["order_ts"] = first_order_ts
    df = df.reset_index()

    df["order_ts"] = df["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    save_csv(df, "orders.csv")

def clean_tickets() -> None:
    df = read_csv("tickets.csv")
    for column in ["ticket_id", "customer_id", "order_id"]:
        df[column] = df[column].astype("string").str.strip()
    df["text"] = df["text"].astype("string").str.strip()
    df["true_category"] = df["true_category"].astype("string").str.strip().str.lower()
    df = df.dropna(subset=["ticket_id", "customer_id"])
    df = df.drop_duplicates("ticket_id", keep="first")
    # Keep blank tickets here so Part C can explicitly record and skip them.
    df["text"] = df["text"].fillna("")
    df["true_category"] = df["true_category"].fillna("unknown")
    save_csv(df, "tickets.csv")


def clean_reference_file(filename: str, required_columns: list[str]) -> None:
    df = read_csv(filename)
    df = df.dropna(subset=required_columns)
    df = df.drop_duplicates()
    save_csv(df, filename)


def main() -> None:
    clean_customers()
    clean_products()
    clean_orders()
    clean_tickets()
    clean_reference_file(
        "policy_eval.csv",
        ["question", "expected_doc", "expected_fact"],
    )
    clean_reference_file(
        "agent_scenarios.csv",
        ["scenario_id", "customer_id", "message", "expected_outcome"],
    )
    print("Cleaning complete. Raw files were not changed.")


if __name__ == "__main__":
    main()
