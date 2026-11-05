"""Part A, task 2: audit the raw exports without changing them.

Two views of each table are needed. pandas' default reader silently turns the planted text 'N/A' into
NaN, so the type/missing table uses the default read (what an analyst would see first), while the
specific audit figures read every cell as literal text so nothing is hidden.
"""
import pandas as pd

from common import RAW, md_table, path, section


def describe(name):
    df = pd.read_csv(path("data", "raw", f"{name}.csv"))
    section(f"{name}.csv: shape {df.shape[0]:,} rows x {df.shape[1]} columns")
    rows = [{"column": c, "dtype": str(df[c].dtype), "missing": int(df[c].isna().sum()),
             "missing_pct": f"{df[c].isna().mean() * 100:.2f}"} for c in df.columns]
    print(md_table(pd.DataFrame(rows)))


def main():
    for name in ["customers", "products", "orders"]:
        describe(name)

    orders = pd.read_csv(f"{RAW}/orders.csv", dtype=str, keep_default_na=False)
    cust = pd.read_csv(f"{RAW}/customers.csv", dtype=str, keep_default_na=False)

    amount_numeric = pd.to_numeric(orders["order_amount"], errors="coerce")
    delivery_numeric = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    age_numeric = pd.to_numeric(cust["age"], errors="coerce")

    figures = [
        ("orders: raw rows", len(orders), 13084),
        ("orders: exact duplicate rows", int(orders.duplicated().sum()), 119),
        ("orders: rows with status pending", int((orders["status"] == "pending").sum()), 656),
        ("orders: distinct order_id", int(orders["order_id"].nunique()), 12309),
        ("orders: order_amount not convertible to a number", int(amount_numeric.isna().sum()), 120),
        ("orders: negative delivery_minutes", int((delivery_numeric < 0).sum()), 106),
        ("orders: blank delivery_minutes", int((orders["delivery_minutes"] == "").sum()), 2541),
        ("customers: blank age", int((cust["age"] == "").sum()), 61),
        ("customers: age outside 10-100", int(((age_numeric < 10) | (age_numeric > 100)).sum()), 6),
        ("customers: distinct raw city spellings", int(cust["city"].nunique()), 24),
        ("customers: distinct non-blank raw gender spellings", int(cust.loc[cust["gender"] != "", "gender"].nunique()), 10),
        ("customers: blank gender", int((cust["gender"] == "").sum()), 30),
    ]
    section("Audit figures")
    print(md_table(pd.DataFrame(figures, columns=["figure", "value", "reference"])))
    for label, value, reference in figures:
        assert value == reference, f"{label}: {value} != {reference}"

    section("Why pandas' default read understates the amount problem")
    default_missing = int(pd.read_csv(f"{RAW}/orders.csv")["order_amount"].isna().sum())
    print(f"Default read shows {default_missing} missing order_amount ('N/A' becomes NaN); "
          f"the literal read finds {int(amount_numeric.isna().sum())} unparseable entries "
          f"({int((orders['order_amount'] == 'N/A').sum())} 'N/A' + {int((orders['order_amount'] == 'error').sum())} 'error').")


if __name__ == "__main__":
    main()
