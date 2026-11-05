"""Part A, task 6: run every query in sql/queries.sql with pd.read_sql_query and check the references."""
import re
import sqlite3

import pandas as pd

from common import DB, md_table, path, section

TOP_PER_CITY = {"Bengaluru": ("C0082", 12856.20), "Chennai": ("C0930", 10151.40), "Delhi": ("C0291", 13742.60),
                "Hyderabad": ("C0743", 12730.05), "Mumbai": ("C0693", 14766.55), "Pune": ("C0426", 10377.10)}


def load_queries():
    with open(path("sql", "queries.sql"), encoding="utf-8") as f:
        text = f.read()
    parts = re.split(r"^-- (Q\d):.*$", text, flags=re.M)
    return dict(zip(parts[1::2], (p.strip() for p in parts[2::2])))


def main():
    queries = load_queries()
    assert sorted(queries) == [f"Q{i}" for i in range(1, 8)], "expected seven queries"
    con = sqlite3.connect(DB)
    try:
        r = {name: pd.read_sql_query(sql, con) for name, sql in queries.items()}
    finally:
        con.close()

    section("Q1 WHERE then HAVING")
    print(f"{len(r['Q1'])} customers")
    assert len(r["Q1"]) == 30

    section("Q2 three-table join")
    q2 = r["Q2"]
    total = round(q2["revenue_inr"].sum(), 2)
    top = q2.iloc[0]
    print(f"{len(q2)} rows, total INR {total:,.2f}, largest {top['city']} / {top['category']} INR {top['revenue_inr']:,.2f}")
    assert len(q2) == 30 and total == 2998668.80 and (top["city"], top["category"], top["revenue_inr"]) == ("Bengaluru", "Electronics", 233909.45)

    section("Q3 scalar subquery")
    print(f"{len(r['Q3'])} products")
    assert len(r["Q3"]) == 11

    section("Q4 LEFT JOIN")
    print(f"{len(r['Q4'])} customers")
    assert len(r["Q4"]) == 25

    section("Q5 common table expressions")
    q5 = r["Q5"]
    print(f"{len(q5)} customers, average INR {q5['avg_spend'].iloc[0]:,.2f}")
    assert len(q5) == 550 and q5["avg_spend"].iloc[0] == 2552.06

    section("Q6 ranking window functions")
    q6 = r["Q6"]
    tops = q6[q6["spend_row_number"] == 1][["city", "customer_id", "total_spend"]]
    print(md_table(tops))
    for row in tops.itertuples(index=False):
        assert (row.customer_id, row.total_spend) == TOP_PER_CITY[row.city], row
    differ = q6[q6["spend_rank"] != q6["spend_row_number"]]
    print(f"rows where RANK() differs from ROW_NUMBER(): {len(differ)}")
    assert len(differ) == 8
    print(md_table(differ))

    section("Q7 running total")
    q7 = r["Q7"]
    print(f"{len(q7)} daily rows, last cumulative INR {q7['cumulative_revenue_inr'].iloc[-1]:,.2f}")
    assert len(q7) == 183 and q7["cumulative_revenue_inr"].iloc[-1] == 2998668.80

    print("\nAll seven queries match the references.")


if __name__ == "__main__":
    main()
