"""Part A, task 7: pandas cross-check of Q2 and exploratory statistics for completed order_amount."""
import pandas as pd

from common import CLEAN, md_table, section


def main():
    orders = pd.read_csv(f"{CLEAN}/orders.csv")
    customers = pd.read_csv(f"{CLEAN}/customers.csv")
    products = pd.read_csv(f"{CLEAN}/products.csv")
    done = (orders[orders["status"] == "completed"]
            .merge(customers[["customer_id", "city"]], on="customer_id")
            .merge(products[["product_id", "category"]], on="product_id"))

    section("(a) Q2 rebuilt with pivot_table(margins=True)")
    pivot = pd.pivot_table(done, values="order_amount", index="city", columns="category",
                           aggfunc="sum", margins=True, margins_name="Total")
    print(md_table(pivot.round(2).reset_index()))
    grand = round(pivot.loc["Total", "Total"], 2)
    print(f"pivot grand total INR {grand:,.2f}; SQL Q2 total INR 2,998,668.80; equal: {grand == 2998668.80}")
    assert grand == 2998668.80

    section("(a) Named aggregation per city")
    per_city = done.groupby("city").agg(total_inr=("order_amount", "sum"), mean_inr=("order_amount", "mean"),
                                        orders=("order_amount", "count")).round(2).reset_index()
    print(md_table(per_city))
    check = per_city.set_index("city")
    assert tuple(check.loc["Bengaluru"]) == (523643.40, 276.47, 1894)
    assert tuple(check.loc["Mumbai"]) == (479058.50, 297.00, 1613)

    section("(b) Completed order_amount distribution")
    a = done["order_amount"]
    q1, q3 = a.quantile(0.25), a.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    stats = [("mean", a.mean()), ("median", a.median()), ("skewness", a.skew()), ("Q1", q1), ("Q3", q3), ("IQR", iqr),
             ("lower fence", low), ("upper fence", high)]
    print(md_table(pd.DataFrame(stats, columns=["statistic", "value"])))
    below, above = int((a < low).sum()), int((a > high).sum())
    print(f"outliers below lower fence: {below}; above upper fence: {above}")
    assert (below, above) == (0, 897) and round(high, 2) == 645.30
    print("upper-fence outliers by category:", done.loc[a > high, "category"].value_counts().to_dict())

    section("(b) Correlation matrix over completed orders")
    cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
    corr = done[cols].corr()
    print(md_table(corr.round(3).reset_index(), floatfmt="{:.3f}"))
    stacked = corr.where(~pd.DataFrame(
        [[i >= j for j in range(len(cols))] for i in range(len(cols))], index=cols, columns=cols)).stack()
    strongest = stacked.abs().sort_values(ascending=False)
    pair = strongest.index[0]
    print(f"largest |r|: {pair[0]} with {pair[1]} = {corr.loc[pair]:.3f}; "
          f"order_amount with quantity = {corr.loc['order_amount', 'quantity']:.3f}")


if __name__ == "__main__":
    main()
