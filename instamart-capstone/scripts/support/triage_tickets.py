"""Guarded ticket triage with rules or optional local Ollama zero/few-shot prompts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MAX_PROMPT_TOKENS = 2048

CATEGORIES = {
    "fraud": "unauthorized transactions, account misuse, or charges the customer did not make",
    "billing": "incorrect charges, invoices, refunds, or payment problems",
    "delivery": "late, missing, damaged, or incorrectly delivered orders",
    "technical": "application, website, login, or checkout failures",
    "other": "requests that do not fit another category",
}

CATEGORY_KEYWORDS = {
    "fraud": [
        "fraud", "unauthorized", "did not place", "never made", "not mine",
        "unknown transaction", "stolen",
    ],
    "billing": [
        "billing", "bill", "charged", "charge", "payment", "invoice",
        "wrong amount", "double", "deducted", "refund",
    ],
    "delivery": [
        "delivery", "delivered", "late", "rider", "missing", "arrived",
        "order not received", "never received",
    ],
    "technical": [
        "app", "crash", "error", "login", "technical", "not working",
        "website", "freezes", "coupon",
    ],
}

FEW_SHOT_EXAMPLES = [
    ("The rider has not delivered my groceries and the order is late.", "delivery"),
    ("I was charged twice and the invoice shows the wrong amount.", "billing"),
    ("I did not make this card transaction; please secure my account.", "fraud"),
    ("The app crashes when I try to sign in.", "technical"),
]

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
ORDER_ID_RE = re.compile(r"\bORD[-_ ]?\d+\b", re.I)
CUSTOMER_ID_RE = re.compile(r"\bC\d{3,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d .()/-]{7,}\d)(?!\w)")


def redact_sensitive(text: str) -> str:
    """Remove contact details and synthetic order/customer identifiers."""
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = ORDER_ID_RE.sub("[ORDER_ID]", text)
    text = CUSTOMER_ID_RE.sub("[CUSTOMER_ID]", text)
    return PHONE_RE.sub("[PHONE]", text)


def contains_sensitive(text: str) -> bool:
    return bool(
        EMAIL_RE.search(text)
        or ORDER_ID_RE.search(text)
        or CUSTOMER_ID_RE.search(text)
        or PHONE_RE.search(text)
    )


def token_upper_bound(text: str) -> int:
    """UTF-8 byte count is a conservative upper bound for tokenizer tokens."""
    return len(text.encode("utf-8"))


def rule_classify(text: str) -> str:
    normalized = text.casefold()
    for category in ("fraud", "billing", "delivery", "technical"):
        if any(term in normalized for term in CATEGORY_KEYWORDS[category]):
            return category
    return "other"


def build_prompt(sanitized_ticket: str, mode: str) -> tuple[str, str]:
    category_list = "\n".join(
        f"- {name}: {description}" for name, description in CATEGORIES.items()
    )
    system = (
        "Classify one customer-support ticket. Return only valid JSON with keys "
        '"category" and "reason". category must be one of fraud, billing, delivery, '
        "technical, other. Do not infer or repeat personal data."
    )
    prompt = f"Categories:\n{category_list}\n"
    if mode == "few-shot":
        prompt += "\nExamples:\n"
        for example, label in FEW_SHOT_EXAMPLES:
            prompt += f"Ticket: {example}\nCategory: {label}\n"
    prompt += f"\nTicket to classify:\n{sanitized_ticket}"
    return system, prompt


def call_local_model(system: str, prompt: str) -> dict:
    model = os.environ.get("OLLAMA_MODEL", "").strip()
    if not model:
        raise RuntimeError(
            "Set OLLAMA_MODEL to a locally installed Ollama model to use "
            "zero-shot or few-shot mode. The default rules mode needs no model."
        )

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0},
    }
    request = Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(
            "Could not reach the local Ollama service. Start Ollama and check "
            "OLLAMA_BASE_URL."
        ) from exc

    content = body.get("message", {}).get("content", "")
    result = json.loads(content)
    category = str(result.get("category", "other")).casefold()
    if category not in CATEGORIES:
        category = "other"
    return {"category": category, "reason": str(result.get("reason", ""))[:240]}


def normalize_truth(value: object) -> str:
    category = str(value or "").strip().casefold()
    return "other" if category in {"", "none", "nan", "unknown"} else category


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["rules", "zero-shot", "few-shot"],
        default="rules",
        help="Use rules by default; zero/few-shot calls a local Ollama model.",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=int(os.environ.get("MAX_PROMPT_TOKENS", DEFAULT_MAX_PROMPT_TOKENS)),
        help="Conservative UTF-8 byte upper bound for the complete model prompt.",
    )
    args = parser.parse_args()

    tickets_path = CLEAN_DIR / "tickets.csv"
    if not tickets_path.exists():
        raise FileNotFoundError("Run clean_data.py before ticket triage.")
    if args.max_prompt_tokens <= 0:
        raise ValueError("--max-prompt-tokens must be a positive integer.")

    tickets = pd.read_csv(tickets_path).fillna("")
    output_rows = []
    guardrail_rows = []

    for row in tickets.to_dict(orient="records"):
        ticket_id = str(row.get("ticket_id", ""))
        raw_text = str(row.get("text", ""))
        safe_text = redact_sensitive(raw_text)
        estimated_tokens = token_upper_bound(safe_text)

        if not raw_text.strip():
            guardrail_rows.append(
                {
                    "ticket_id": ticket_id,
                    "guardrail_status": "SKIPPED_EMPTY",
                    "estimated_token_upper_bound": 0,
                    "personal_data_redacted": safe_text != raw_text,
                }
            )
            continue

        if args.mode == "rules":
            system, prompt = "", safe_text
        else:
            system, prompt = build_prompt(safe_text, args.mode)

        full_prompt = f"{system}\n{prompt}".strip()
        estimated_tokens = token_upper_bound(full_prompt)
        if estimated_tokens > args.max_prompt_tokens:
            guardrail_rows.append(
                {
                    "ticket_id": ticket_id,
                    "guardrail_status": "BLOCKED_TOO_LONG",
                    "estimated_token_upper_bound": estimated_tokens,
                    "personal_data_redacted": safe_text != raw_text,
                }
            )
            continue

        if contains_sensitive(full_prompt):
            guardrail_rows.append(
                {
                    "ticket_id": ticket_id,
                    "guardrail_status": "BLOCKED_SENSITIVE_DATA",
                    "estimated_token_upper_bound": estimated_tokens,
                    "personal_data_redacted": safe_text != raw_text,
                }
            )
            continue

        if args.mode == "rules":
            prediction = {"category": rule_classify(safe_text), "reason": "keyword rules"}
        else:
            prediction = call_local_model(system, prompt)

        category = prediction["category"]
        truth = normalize_truth(row.get("true_category", ""))
        high_priority = category == "fraud"
        output_rows.append(
            {
                "ticket_id": ticket_id,
                "customer_id": row.get("customer_id", ""),
                "order_id": row.get("order_id", ""),
                "predicted_category": category,
                "true_category": truth,
                "category_match": category == truth,
                "priority": "high" if high_priority else "normal",
                "team": "security" if high_priority else "customer_support",
                "response_sla": "within 2 hours" if high_priority else "within 24 hours",
                "escalate_to_human": high_priority or category == "other",
                "classification_mode": args.mode,
            }
        )
        guardrail_rows.append(
            {
                "ticket_id": ticket_id,
                "guardrail_status": "CLASSIFIED",
                "estimated_token_upper_bound": estimated_tokens,
                "personal_data_redacted": safe_text != raw_text,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(output_rows)
    audit = pd.DataFrame(guardrail_rows)
    result.to_csv(OUTPUT_DIR / "ticket_triage.csv", index=False)
    audit.to_csv(OUTPUT_DIR / "ticket_guardrail_audit.csv", index=False)

    print(f"Input tickets: {len(tickets):,}; classified: {len(result):,}")
    print("Guardrail decisions:")
    print(audit["guardrail_status"].value_counts().to_string())
    if not result.empty:
        print(f"Classification match rate: {result['category_match'].mean():.1%}")
    print(f"Mode: {args.mode}")
    print(f"Saved reports under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
