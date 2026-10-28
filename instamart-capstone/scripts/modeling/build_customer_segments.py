"""Choose a K-Means RFM model and derive customer/churn-risk segments."""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
STATE_DIR = PROJECT_ROOT / "state"

RFM_COLUMNS = ["recency_days", "frequency", "monetary"]


def write_metric_chart(frame: pd.DataFrame, filename: str, metric: str, title: str) -> None:
    width, height = 760, 420
    left, right, top, bottom = 72, 24, 60, 58
    plot_w, plot_h = width - left - right, height - top - bottom
    values = frame[metric].astype(float).to_numpy()
    ks = frame["k"].astype(int).to_numpy()
    low, high = float(np.min(values)), float(np.max(values))
    if low == high:
        high = low + 1.0

    points = []
    for k, value in zip(ks, values):
        x = left + (k - ks.min()) / max(ks.max() - ks.min(), 1) * plot_w
        y = top + (high - value) / (high - low) * plot_h
        points.append((x, y, k, value))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="{left}" y="32" font-family="Arial" font-size="20" '
        f'font-weight="bold" fill="#0f172a">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#64748b"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
        f'y2="{top + plot_h}" stroke="#64748b"/>',
    ]
    for x, y, k, value in points:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#2563eb"/>')
        parts.append(
            f'<text x="{x - 12:.1f}" y="{top + plot_h + 24}" '
            f'font-family="Arial" font-size="13" fill="#334155">{k}</text>'
        )
        parts.append(
            f'<text x="{x - 19:.1f}" y="{y - 11:.1f}" '
            f'font-family="Arial" font-size="11" fill="#0f172a">{value:.3f}</text>'
        )
    parts.extend(
        [
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 12}" '
            'font-family="Arial" font-size="13" fill="#334155" text-anchor="middle">K</text>',
            "</svg>",
        ]
    )
    (OUTPUT_DIR / filename).write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    customers_path = CLEAN_DIR / "customers.csv"
    orders_path = CLEAN_DIR / "orders.csv"
    if not customers_path.exists() or not orders_path.exists():
        raise FileNotFoundError("Run clean_data.py before customer clustering.")

    customers = pd.read_csv(customers_path)
    orders = pd.read_csv(orders_path)
    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    completed = orders[
        (orders["status"].astype(str).str.lower() == "completed")
        & orders["order_ts"].notna()
    ].copy()
    if completed.empty:
        raise ValueError("No completed orders are available for RFM clustering.")

    anchor_date = completed["order_ts"].max().normalize() + pd.Timedelta(days=1)
    rfm = completed.groupby("customer_id").agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    rfm["recency_days"] = (anchor_date - rfm["last_order"]).dt.days
    rfm = rfm.drop(columns="last_order")

    result = customers.set_index("customer_id").join(rfm)
    result["frequency"] = result["frequency"].fillna(0).astype(int)
    result["monetary"] = result["monetary"].fillna(0.0)
    result["recency_days"] = result["recency_days"].fillna(
        (anchor_date - pd.to_datetime(result["signup_date"], errors="coerce"))
        .dt.days.fillna(180)
    )

    active = result.loc[result["frequency"] > 0, RFM_COLUMNS].copy()
    if len(active) < 9:
        raise ValueError("At least nine active customers are required to test K=2..8.")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(active)
    evaluation_rows = []
    models = {}
    for k in range(2, 9):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(scaled)
        score = silhouette_score(scaled, labels)
        evaluation_rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette_score": float(score),
            }
        )
        models[k] = (model, labels)

    evaluation = pd.DataFrame(evaluation_rows)
    best_k = int(evaluation.loc[evaluation["silhouette_score"].idxmax(), "k"])
    best_model, best_labels = models[best_k]

    result["cluster"] = -1
    result.loc[active.index, "cluster"] = best_labels
    active_profiles = (
        result.loc[active.index]
        .groupby("cluster")[RFM_COLUMNS]
        .mean()
    )

    # Higher recency and lower frequency/monetary value indicate greater
    # disengagement risk. This is an unsupervised segment-based risk flag,
    # not a supervised probability or a ground-truth churn label.
    standardized_profiles = scaler.transform(active_profiles[RFM_COLUMNS])
    risk_score = (
        standardized_profiles[:, 0]
        - 0.5 * standardized_profiles[:, 1]
        - 0.5 * standardized_profiles[:, 2]
    )
    risk_cutoff = float(np.quantile(risk_score, 0.75))
    high_risk_clusters = active_profiles.index[risk_score >= risk_cutoff].tolist()

    result["segment"] = "No completed orders"
    result.loc[active.index, "segment"] = [
        f"Cluster {int(label)}" for label in best_labels
    ]
    result["churn_risk"] = "No completed orders"
    result.loc[active.index, "churn_risk"] = [
        "High" if label in high_risk_clusters else "Standard"
        for label in best_labels
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(OUTPUT_DIR / "kmeans_k_evaluation.csv", index=False)
    result.reset_index().to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)
    result.reset_index().to_csv(OUTPUT_DIR / "churn_risk_predictions.csv", index=False)
    active_profiles.assign(
        churn_risk_score=risk_score,
        high_risk=active_profiles.index.isin(high_risk_clusters),
    ).to_csv(OUTPUT_DIR / "cluster_profiles.csv")
    write_metric_chart(evaluation, "kmeans_inertia.svg", "inertia", "K-Means inertia by K")
    write_metric_chart(
        evaluation,
        "kmeans_silhouette.svg",
        "silhouette_score",
        "K-Means silhouette score by K",
    )
    joblib.dump(
        {
            "model": best_model,
            "scaler": scaler,
            "features": RFM_COLUMNS,
            "selected_k": best_k,
            "high_risk_clusters": high_risk_clusters,
        },
        STATE_DIR / "customer_rfm_kmeans.joblib",
    )

    print(f"RFM anchor date: {anchor_date.date()}")
    print("K evaluation (2 through 8):")
    print(evaluation.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Selected K={best_k} by maximum silhouette score.")
    print(f"High-risk clusters by RFM profile: {high_risk_clusters}")
    print(f"Saved cluster and churn-risk outputs under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
