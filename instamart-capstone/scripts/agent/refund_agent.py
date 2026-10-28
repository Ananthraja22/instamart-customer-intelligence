"""Evaluate the supplied synthetic refund scenarios against policy rules."""

from pathlib import Path
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

ORDER_PATTERN = re.compile(r"\bORD-\d+\b", re.IGNORECASE)


def decide(message: str, customer_id: str, orders: pd.DataFrame) -> dict:
    if "[LOOP]" in message.upper():
        return {
            "outcome": "FALLBACK",
            "refund_inr": 0.0,
            "reason": "Loop guard triggered; hand off to a human agent.",
        }

    match = ORDER_PATTERN.search(message)
    if not match:
        return {
            "outcome": "NEED_ORDER_ID",
            "refund_inr": 0.0,
            "reason": "Ask the customer for the order identifier.",
        }

    order_id = match.group(0).upper()
    matches = orders[
        (orders["order_id"].str.upper() == order_id)
        & (orders["customer_id"].astype(str) == str(customer_id))
    ]
    if matches.empty:
        return {
            "outcome": "NOT_FOUND",
            "refund_inr": 0.0,
            "reason": "Order was not found for this customer.",
        }

    order = matches.iloc[0]
    if str(order["status"]).lower() != "completed":
        return {
            "outcome": "DENIED",
            "refund_inr": 0.0,
            "reason": "Policy allows refunds only for completed orders.",
        }

    delivery_minutes = pd.to_numeric(order["delivery_minutes"], errors="coerce")
    if pd.isna(delivery_minutes) or delivery_minutes <= 45:
        return {
            "outcome": "DENIED",
            "refund_inr": 0.0,
            "reason": "Completed delivery was not more than 45 minutes.",
        }

    amount = float(order["order_amount"])
    if amount > 1000:
        return {
            "outcome": "ESCALATE",
            "refund_inr": amount,
            "reason": "Refund exceeds INR 1000 and requires human support lead approval.",
        }

    return {
        "outcome": "REFUND",
        "refund_inr": amount,
        "reason": "Completed order was delivered after 45 minutes; within auto-approval limit.",
    }


def main() -> None:
    orders_path = CLEAN_DIR / "orders.csv"
    scenarios_path = CLEAN_DIR / "agent_scenarios.csv"
    if not orders_path.exists() or not scenarios_path.exists():
        raise FileNotFoundError("Run clean_data.py before evaluating scenarios.")

    orders = pd.read_csv(orders_path).fillna("")
    scenarios = pd.read_csv(scenarios_path).fillna("")
    rows = []

    for scenario in scenarios.to_dict(orient="records"):
        decision = decide(
            str(scenario["message"]),
            str(scenario["customer_id"]),
            orders,
        )
        expected = str(scenario["expected_outcome"]).upper()
        actual = decision["outcome"]
        rows.append(
            {
                "scenario_id": scenario["scenario_id"],
                "customer_id": scenario["customer_id"],
                "actual_outcome": actual,
                "expected_outcome": expected,
                "outcome_match": actual == expected,
                "refund_inr": round(decision["refund_inr"], 2),
                "expected_refund_inr": float(scenario["expected_refund_inr"]),
                "reason": decision["reason"],
            }
        )

    result = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "refund_scenario_results.csv"
    result.to_csv(output_path, index=False)

    print(result.to_string(index=False))
    print(
        f"\nOutcome matches: {result['outcome_match'].sum()}/{len(result)}"
    )
    print(f"Saved results to {output_path}")
    print(
        "\nNote: The dataset has no refund-request timestamp, so the 7-day "
        "request-window rule cannot be verified by this demo."
    )


if __name__ == "__main__":
    main()