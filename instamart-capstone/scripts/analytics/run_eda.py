"""Create auditable quality summaries and SVG charts using Pandas."""

from __future__ import annotations

from html import escape
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def save_bar_chart(
    title: str,
    labels: list[str],
    values: list[float],
    destination: Path,
    value_label: str,
) -> None:
    width = 920
    row_height = 38
    height = 90 + row_height * max(len(labels), 1)
    label_width = 210
    chart_width = 570
    max_value = max(values, default=0) or 1

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="24" y="34" font-size="22" font-family="Arial" '
        f'font-weight="bold" fill="#0f172a">{escape(title)}</text>',
    ]

    for index, (label, value) in enumerate(zip(labels, values)):
        y = 62 + index * row_height
        bar_width = chart_width * max(value, 0) / max_value
        parts.extend(
            [
                f'<text x="24" y="{y + 19}" font-size="14" font-family="Arial" '
                f'fill="#334155">{escape(label)}</text>',
                f'<rect x="{label_width}" y="{y}" width="{bar_width:.1f}" '
                f'height="24" rx="4" fill="#2563eb"/>',
                f'<text x="{label_width + bar_width + 9:.1f}" y="{y + 18}" '
                f'font-size="13" font-family="Arial" fill="#0f172a">'
                f'{value:,.0f} {escape(value_label)}</text>',
            ]
        )

    parts.append("</svg>")
    destination.write_text("\n".join(parts), encoding="utf-8")


def outlier_summary(frame: pd.DataFrame, table: str, column: str) -> dict:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return {
            "table": table,
            "column": column,
            "non_null_count": 0,
            "q1": None,
            "q3": None,
            "iqr_lower_bound": None,
            "iqr_upper_bound": None,
            "outlier_count": 0,
            "cleaning_action": "No numeric values to assess",
        }

    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((values < lower) | (values > upper)).sum())
    return {
        "table": table,
        "column": column,
        "non_null_count": int(len(values)),
        "q1": round(q1, 3),
        "q3": round(q3, 3),
        "iqr_lower_bound": round(lower, 3),
        "iqr_upper_bound": round(upper, 3),
        "outlier_count": count,
        "cleaning_action": "Flagged for review; retained unless invalid by business rule",
    }


def identify_outlier_rows(
    frame: pd.DataFrame, table: str, id_column: str, column: str
) -> list[dict]:
    values = pd.to_numeric(frame[column], errors="coerce")
    non_null = values.dropna()
    if non_null.empty:
        return []
    q1 = float(non_null.quantile(0.25))
    q3 = float(non_null.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (values < lower) | (values > upper)
    return [
        {
            "table": table,
            "record_id": str(frame.loc[index, id_column]),
            "column": column,
            "value": float(values.loc[index]),
            "iqr_lower_bound": round(lower, 3),
            "iqr_upper_bound": round(upper, 3),
        }
        for index in frame.index[mask.fillna(False)]
    ]


def main() -> None:
    customers_path = CLEAN_DIR / "customers.csv"
    orders_path = CLEAN_DIR / "orders.csv"
    if not customers_path.exists() or not orders_path.exists():
        raise FileNotFoundError("Run clean_data.py before the EDA step.")

    customers = pd.read_csv(customers_path)
    orders = pd.read_csv(orders_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outliers = []
    outlier_records = []
    if "age" in customers.columns:
        outliers.append(outlier_summary(customers, "customers", "age"))
        outlier_records.extend(
            identify_outlier_rows(customers, "customers", "customer_id", "age")
        )
    for column in ["quantity", "order_amount", "delivery_minutes"]:
        if column in orders.columns:
            outliers.append(outlier_summary(orders, "orders", column))
            outlier_records.extend(
                identify_outlier_rows(orders, "orders", "order_id", column)
            )
    quality_path = OUTPUT_DIR / "data_quality_outliers.csv"
    pd.DataFrame(outliers).to_csv(quality_path, index=False)
    outlier_rows_path = OUTPUT_DIR / "identified_outlier_rows.csv"
    pd.DataFrame(outlier_records).to_csv(outlier_rows_path, index=False)

    city_counts = customers["city"].fillna("Unknown").value_counts()
    save_bar_chart(
        "Customers by standardized city",
        city_counts.index.astype(str).tolist(),
        city_counts.astype(float).tolist(),
        OUTPUT_DIR / "customers_by_city.svg",
        "customers",
    )

    status_counts = orders["status"].fillna("unknown").value_counts()
    save_bar_chart(
        "Orders by final status",
        status_counts.index.astype(str).tolist(),
        status_counts.astype(float).tolist(),
        OUTPUT_DIR / "orders_by_status.svg",
        "orders",
    )

    completed = orders[orders["status"].astype(str).str.lower() == "completed"].copy()
    completed["order_ts"] = pd.to_datetime(completed["order_ts"], errors="coerce")
    completed["order_amount"] = pd.to_numeric(
        completed["order_amount"], errors="coerce"
    )
    monthly = (
        completed.dropna(subset=["order_ts"])
        .assign(month=lambda df: df["order_ts"].dt.to_period("M").astype(str))
        .groupby("month")["order_amount"]
        .sum()
        .sort_index()
    )
    save_bar_chart(
        "Completed order value by month",
        monthly.index.astype(str).tolist(),
        monthly.fillna(0).astype(float).tolist(),
        OUTPUT_DIR / "completed_sales_by_month.svg",
        "INR",
    )

    print(f"Outlier report: {quality_path}")
    print(f"Flagged outlier rows: {outlier_rows_path}")
    print(f"Charts saved under: {OUTPUT_DIR}")
    print("IQR outliers are reported for review, not automatically discarded.")


if __name__ == "__main__":
    main()
