"""Check the raw CSV files and report their basic shape and columns."""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

EXPECTED_FILES = [
    "customers.csv",
    "products.csv",
    "orders.csv",
    "tickets.csv",
    "policy_eval.csv",
    "agent_scenarios.csv",
]


def main() -> None:
    missing = [name for name in EXPECTED_FILES if not (RAW_DIR / name).exists()]
    if missing:
        print("Missing input files:")
        for name in missing:
            print(f"  - {name}")
        raise SystemExit(1)

    print(f"Raw data directory: {RAW_DIR}")
    for name in EXPECTED_FILES:
        path = RAW_DIR / name
        frame = pd.read_csv(path)
        print(f"\n{name}: {len(frame):,} rows, {len(frame.columns)} columns")
        print("  Columns:", ", ".join(frame.columns))
        print(f"  Duplicate rows: {frame.duplicated().sum():,}")
        print(f"  Missing cells: {int(frame.isna().sum().sum()):,}")


if __name__ == "__main__":
    main()