"""Predict customer segments and relative churn-risk flags from RFM features."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "state" / "customer_rfm_kmeans.joblib"
REQUIRED_COLUMNS = ["recency_days", "frequency", "monetary"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="CSV with one customer per row and recency_days, frequency, monetary columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination CSV (default: <input stem>_predictions.csv).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help=f"Saved model bundle (default: {MODEL_PATH}).",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")
    if not args.model.is_file():
        raise FileNotFoundError(
            f"Model not found: {args.model}. Run run_pipeline.cmd first."
        )

    frame = pd.read_csv(args.input)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Input CSV must contain at least one customer row.")

    features = frame[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("RFM features must contain only finite numeric values.")
    if (features["recency_days"] < 0).any() or (features["frequency"] < 0).any():
        raise ValueError("recency_days and frequency must be zero or greater.")

    bundle = joblib.load(args.model)
    model = bundle["model"]
    scaler = bundle["scaler"]
    expected = list(bundle.get("features", REQUIRED_COLUMNS))
    if expected != REQUIRED_COLUMNS:
        raise ValueError(f"Saved model expects unsupported features: {expected}")

    # Customers with no completed orders were excluded from training. Keep that
    # business rule when scoring new rows rather than inventing a cluster.
    active = features["frequency"] > 0
    clusters = np.full(len(frame), -1, dtype=int)
    if active.any():
        scaled = scaler.transform(features.loc[active, REQUIRED_COLUMNS])
        clusters[active.to_numpy()] = model.predict(scaled)

    high_risk_clusters = {int(value) for value in bundle["high_risk_clusters"]}
    result = frame.copy()
    result["cluster"] = clusters
    result["segment"] = [
        f"Cluster {cluster}" if cluster >= 0 else "No completed orders"
        for cluster in clusters
    ]
    result["churn_risk"] = [
        ("High" if cluster in high_risk_clusters else "Standard")
        if cluster >= 0
        else "No completed orders"
        for cluster in clusters
    ]

    output = args.output or args.input.with_name(f"{args.input.stem}_predictions.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Predicted {len(result):,} customers; saved results to {output}")
    print(
        "Note: churn_risk is a relative cluster-based flag, not a trained churn "
        "probability."
    )


if __name__ == "__main__":
    main()
