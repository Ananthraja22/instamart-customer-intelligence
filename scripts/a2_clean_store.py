"""Part A, tasks 3-5: clean customers and orders with explicit rules, then store CSVs and SQLite.

Rules run in the brief's order and each prints how many rows or values it changed.
data/clean/orders.csv adds one column beyond the raw file, delivery_imputed (1 = the delivery time was
filled from a category median, not measured). Part B and Part D both need to tell the two apart.
"""
import sqlite3

import pandas as pd

from common import CLEAN, DB, RAW, ensure_dirs, md_table, path, section

GENDER_MAP = {"male": "male", "m": "male", "female": "female", "f": "female", "": "unknown"}
ORDER_COLUMNS = ["order_id", "customer_id", "product_id", "quantity", "order_ts", "status",
                 "discount_pct", "order_amount", "delivery_minutes", "delivery_imputed"]


def clean_customers():
    log = []
    cust = pd.read_csv(f"{RAW}/customers.csv", dtype=str, keep_default_na=False)

    raw_city = cust["city"]
    cust["city"] = raw_city.str.strip().str.title()
    log.append(("city: strip and title-case", int((raw_city != cust["city"]).sum())))

    raw_gender = cust["gender"]
    cust["gender"] = raw_gender.str.strip().str.lower().map(GENDER_MAP)
    assert cust["gender"].notna().all(), "unmapped gender spelling"
    log.append(("gender: map spellings to male/female/unknown", int((raw_gender != cust["gender"]).sum())))
    log.append(("gender: blank mapped to unknown (subset of the row above)", int((raw_gender == "").sum())))

    age = pd.to_numeric(cust["age"], errors="coerce")
    impossible = (age < 10) | (age > 100)
    log.append(("age: outside 10-100 set to missing", int(impossible.sum())))
    age = age.mask(impossible)
    log.append(("age: missing filled with the city median", int(age.isna().sum())))
    cust["age"] = age.fillna(age.groupby(cust["city"]).transform("median"))

    cust["signup_date"] = pd.to_datetime(cust["signup_date"], format="%Y-%m-%d")
    log.append(("signup_date: converted to datetime", len(cust)))
    return cust, log


def clean_orders(products):
    log = []
    orders = pd.read_csv(f"{RAW}/orders.csv", dtype=str, keep_default_na=False)
    log.append(("raw rows", len(orders)))

    # 1. Timestamps
    orders["order_ts"] = pd.to_datetime(orders["order_ts"], format="%Y-%m-%d %H:%M:%S")
    log.append(("1. order_ts converted to datetime", len(orders)))

    # 2. Exact duplicates
    before = len(orders)
    orders = orders.drop_duplicates()
    log.append(("2. exact duplicate rows removed", before - len(orders)))

    # 3. Latest record per order_id. A tie would make "latest" ambiguous, so refuse to guess.
    latest = orders.groupby("order_id")["order_ts"].transform("max")
    assert not orders[orders["order_ts"] == latest].duplicated("order_id").any(), "tie on latest order_ts"
    before = len(orders)
    orders = orders[orders["order_ts"] == latest].copy()
    log.append(("3. earlier records of the same order_id removed", before - len(orders)))

    for col in ["quantity", "discount_pct"]:
        orders[col] = pd.to_numeric(orders[col])
    orders = orders.merge(products[["product_id", "category", "price_inr"]], on="product_id",
                          how="left", validate="many_to_one")
    assert orders["price_inr"].notna().all(), "order references an unknown product"

    # 4. Amounts: rebuild unparseable values from the product price.
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    log.append(("4. unparseable order_amount filled from price, quantity, discount", int(orders["order_amount"].isna().sum())))
    rebuilt = (orders["price_inr"] * orders["quantity"] * (1 - orders["discount_pct"] / 100)).round(2)
    orders["order_amount"] = orders["order_amount"].fillna(rebuilt)

    # 5. Delivery times: negative is impossible; only completed orders may be imputed.
    orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    negative = orders["delivery_minutes"] < 0
    log.append(("5a. negative delivery_minutes set to missing", int(negative.sum())))
    orders["delivery_minutes"] = orders["delivery_minutes"].mask(negative)

    completed = orders["status"] == "completed"
    to_fill = completed & orders["delivery_minutes"].isna()
    log.append(("5b. completed orders: missing delivery filled with category median", int(to_fill.sum())))
    medians = orders.loc[completed].groupby("category")["delivery_minutes"].transform("median")
    orders.loc[completed, "delivery_minutes"] = orders.loc[completed, "delivery_minutes"].fillna(medians)
    orders["delivery_imputed"] = to_fill.astype(int)
    log.append(("5c. cancelled orders: delivery_minutes left missing", int((orders["status"] == "cancelled").sum())))

    medians_used = orders.loc[completed].groupby("category")["delivery_minutes"].median()
    return orders[ORDER_COLUMNS].sort_values("order_id").reset_index(drop=True), log, medians_used


def store(customers, products, orders):
    """Dates go to disk as text so the CSVs and the SQLite tables use one exact format."""
    ensure_dirs(CLEAN)
    customers = customers.assign(signup_date=customers["signup_date"].dt.strftime("%Y-%m-%d"))
    orders = orders.assign(order_ts=orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S"))
    customers.to_csv(path("data", "clean", "customers.csv"), index=False)
    products.to_csv(path("data", "clean", "products.csv"), index=False)
    orders.to_csv(path("data", "clean", "orders.csv"), index=False)
    with sqlite3.connect(DB) as con:
        customers.to_sql("customers", con, if_exists="replace", index=False)
        products.to_sql("products", con, if_exists="replace", index=False)
        orders.to_sql("orders", con, if_exists="replace", index=False)
        con.execute("CREATE UNIQUE INDEX idx_orders_order_id ON orders(order_id)")
        con.execute("CREATE INDEX idx_orders_customer ON orders(customer_id)")
    con.close()


def main():
    customers, customer_log = clean_customers()
    products = pd.read_csv(f"{RAW}/products.csv")
    orders, order_log, medians = clean_orders(products)

    section("Customer cleaning rules (rows or values changed)")
    print(md_table(pd.DataFrame(customer_log, columns=["rule", "changed"])))
    section("Order cleaning rules (rows or values changed)")
    print(md_table(pd.DataFrame(order_log, columns=["rule", "changed"])))
    section("Delivery medians used per category (minutes)")
    print(md_table(medians.rename("median_minutes").reset_index()))

    store(customers, products, orders)
    section("Result")
    print(f"customers={len(customers)} products={len(products)} orders={len(orders)}")
    print("cities:", customers["city"].value_counts().to_dict())
    print("genders:", customers["gender"].value_counts().to_dict())
    print(f"age range {customers['age'].min()} to {customers['age'].max()}, "
          f"non-integer ages after imputation: {int((customers['age'] % 1 != 0).sum())}")
    print(f"written: data/clean/*.csv and data/instamart.db")


if __name__ == "__main__":
    main()
