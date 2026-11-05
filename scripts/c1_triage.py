"""Part C: support-ticket triage around mock_llm.py with guardrails, retries, validation, budget, routing.

Pipeline for each ticket, in file order:
    input guardrails -> PII redaction -> model call with retry -> output validation -> routing

The model only ever receives the redacted ticket text. Output files hold no ticket text at all.
"""
import json
import random
import re
import time

import pandas as pd

import common  # noqa: F401  (puts the repo root on sys.path for mock_llm)
from common import OUTPUTS, RAW, ensure_dirs, md_table, section
import mock_llm
from mock_llm import TransientLLMError, chat, est_tokens

MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
ALLOWED = ("billing", "delivery", "technical", "fraud")
REDACTIONS = (("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
              ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
              ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"))

ZERO_SHOT = """You are the support desk triage assistant for a grocery delivery app.
CLASSIFY the customer ticket into exactly one of these categories: billing, delivery, technical, fraud.
Reply with JSON only, in exactly this form: {"category": "...", "reason": "..."}"""

FEW_SHOT = ZERO_SHOT + """
EXAMPLES:
Ticket: I was charged for an order I never placed and my card shows a transaction I do not recognise.
Reply: {"category": "fraud", "reason": "unrecognised charge on the customer's card"}
Ticket: My invoice shows a higher amount than the price in the app.
Reply: {"category": "billing", "reason": "wrong invoice amount"}
Ticket: The app freezes on the payment page and shows an error.
Reply: {"category": "technical", "reason": "app fault during checkout"}
Ticket: The rider says the package was delivered but nothing reached me.
Reply: {"category": "delivery", "reason": "package not received"}"""


def redact(text):
    """Apply the three patterns in order; returns the masked text and a count per type."""
    counts = {}
    for name, pattern, placeholder in REDACTIONS:
        text, counts[name] = pattern.subn(placeholder, text)
    return text, counts


def input_guard(text):
    """Returns a status when the ticket must not reach the model, otherwise None."""
    if not text.strip():
        return "skipped_empty"
    if est_tokens(text) > MAX_INPUT_TOKENS:
        return "flagged_long"
    return None


def validate(reply):
    """Returns (status, category). Accept only what can be proven; everything else goes to a human."""
    try:
        parsed = json.loads(reply or "")
    except ValueError:
        found = set(re.findall(r"[a-z]+", (reply or "").lower())) & set(ALLOWED)
        if len(found) == 1:
            return "classified_keyword", found.pop()
        return "flagged_invalid_output", ""
    category = parsed.get("category") if isinstance(parsed, dict) else None
    if isinstance(category, str) and category in ALLOWED:
        return "classified_json", category
    return "flagged_invalid_output", ""


def call_with_retry(ticket_id, messages, jitter):
    """Exponential backoff with jitter. Returns (response, number_of_transient_errors)."""
    errors = 0
    for retry in range(MAX_RETRIES + 1):
        try:
            return chat(messages), errors
        except TransientLLMError:
            errors += 1
            if retry == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** retry * jitter.uniform(0.9, 1.15)
            print(f"[retry] ticket={ticket_id} retry={retry} wait={wait:.3f}s")
            time.sleep(wait)


def run_pipeline(tickets, system_prompt, token_budget=None):
    """One full run. With a budget, a call that could overrun it is not made and the ticket is deferred."""
    mock_llm.reset()
    jitter = random.Random(42)
    used = 0
    rows = []
    for t in tickets.itertuples(index=False):
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "true_category": t.true_category,
               "status": "", "category": "", "tokens": 0, "transient_errors": 0, "cumulative_tokens": used}
        blocked = input_guard(t.text)
        if blocked:
            row["status"] = blocked
            rows.append(row)
            continue
        safe_text, _ = redact(t.text)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": safe_text}]
        if token_budget is not None:
            # Prompt size is known up front; reserve the largest reply the validator can accept.
            needed = est_tokens(system_prompt) + est_tokens(safe_text) + est_tokens('{"category": "technical", "reason": "keyword match"}')
            if used + needed > token_budget:
                row["status"] = "deferred_budget"
                rows.append(row)
                continue
        response, errors = call_with_retry(t.ticket_id, messages, jitter)
        row["status"], row["category"] = validate(response.choices[0].message.content)
        row["tokens"] = response.usage.total_tokens
        row["transient_errors"] = errors
        used += response.usage.total_tokens
        row["cumulative_tokens"] = used
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(result, label):
    classified = result[result["status"].isin(["classified_json", "classified_keyword"])]
    correct = int((classified["category"] == classified["true_category"]).sum())
    total_tokens = int(result["tokens"].sum())
    return {"prompt": label, "classified": len(classified),
            "classified_json": int((result["status"] == "classified_json").sum()),
            "classified_keyword": int((result["status"] == "classified_keyword").sum()),
            "flagged_invalid_output": int((result["status"] == "flagged_invalid_output").sum()),
            "correct": correct, "accuracy_pct": round(correct / len(classified) * 100, 2),
            "total_tokens": total_tokens, "tokens_per_classified": round(total_tokens / len(classified), 2),
            "transient_errors": int(result["transient_errors"].sum())}


def route(result):
    """Apply the escalation policy using the Part B segments. Tightest deadline wins when two rules apply."""
    segments = pd.read_csv(f"{OUTPUTS}/customer_segments.csv")[["customer_id", "segment_name", "is_premium"]]
    routed = result.merge(segments, on="customer_id", how="left", validate="many_to_one")
    routed["segment_name"] = routed["segment_name"].fillna("No completed orders")
    routed["is_premium"] = routed["is_premium"].fillna(False).astype(bool)

    def decide(r):
        base_sla = 1 if r.is_premium else 24
        if r.status == "skipped_empty":
            return "ask_customer_for_details", "low", base_sla
        if r.status in ("flagged_long", "flagged_invalid_output", "deferred_budget"):
            return "human_review", "high" if r.is_premium else "normal", base_sla
        if r.category == "fraud":
            return "security_team", "high", min(2, base_sla)
        return f"{r.category}_team", "high" if r.is_premium else "normal", base_sla

    decisions = [decide(r) for r in routed.itertuples(index=False)]
    routed["queue"], routed["priority"], routed["sla_hours"] = zip(*decisions)
    return routed


def assert_no_pii(frame):
    """No output file may contain anything the redaction patterns would catch."""
    blob = frame.to_csv(index=False)
    for name, pattern, _ in REDACTIONS:
        assert not pattern.search(blob), f"{name} found in an output file"


def main():
    ensure_dirs(OUTPUTS)
    tickets = pd.read_csv(f"{RAW}/tickets.csv", keep_default_na=False)

    section("Task 1: input guardrails")
    guards = tickets["text"].map(input_guard)
    long_tokens = sorted({est_tokens(t) for t in tickets.loc[guards == "flagged_long", "text"]})
    print(f"empty {int((guards == 'skipped_empty').sum())}; long {int((guards == 'flagged_long').sum())} "
          f"(estimated tokens {long_tokens}); eligible {int(guards.isna().sum())}")

    section("Task 2: redaction over all 164 tickets")
    totals = {name: 0 for name, _, _ in REDACTIONS}
    redacted = []
    for text in tickets["text"]:
        masked, counts = redact(text)
        redacted.append(masked)
        for k, v in counts.items():
            totals[k] += v
    print(f"{totals}; total {sum(totals.values())}")
    left = sum(len(p.findall(t)) for t in redacted for _, p, _ in REDACTIONS)
    print(f"matches left after re-applying the patterns to the redacted texts: {left}")
    assert totals == {"email": 55, "phone": 91, "order_id": 132} and left == 0

    section("Task 3: prompt checks")
    for name, prompt in [("zero-shot", ZERO_SHOT), ("few-shot", FEW_SHOT)]:
        assert "CLASSIFY" in prompt and all(c in prompt for c in ALLOWED) and '{"category": "...", "reason": "..."}' in prompt
        print(f"{name}: {est_tokens(prompt)} estimated tokens")
    assert "EXAMPLES:" not in ZERO_SHOT and "\nEXAMPLES:\n" in FEW_SHOT

    section("Tasks 4-5: zero-shot run (one retry line per transient error)")
    zero = run_pipeline(tickets, ZERO_SHOT)
    section("Tasks 4-5: few-shot run (one retry line per transient error)")
    few = run_pipeline(tickets, FEW_SHOT)

    section("Task 6: zero-shot versus few-shot")
    summary = pd.DataFrame([summarise(zero, "zero-shot"), summarise(few, "few-shot")])
    print(md_table(summary))
    for s in summary.itertuples(index=False):
        assert (s.classified_json, s.classified_keyword, s.flagged_invalid_output) == (135, 12, 9), s
        assert s.transient_errors == 19, s

    for label, frame in [("zero-shot", zero), ("few-shot", few)]:
        classified = frame[frame["status"].isin(["classified_json", "classified_keyword"])]
        table = pd.crosstab(classified["true_category"], classified["category"]).reindex(
            index=list(ALLOWED), columns=list(ALLOWED), fill_value=0)
        section(f"Confusion table, {label} (rows: true; columns: predicted)")
        print(md_table(table.rename_axis(None, axis=1).reset_index().rename(columns={"true_category": "true"})))

    section("Routing the few-shot run (the more accurate prompt)")
    routed = route(few)
    columns = ["ticket_id", "customer_id", "status", "category", "true_category", "segment_name", "is_premium",
               "queue", "priority", "sla_hours", "tokens", "transient_errors"]
    results = routed[columns]
    assert_no_pii(results)
    results.to_csv(f"{OUTPUTS}/triage_results.csv", index=False)
    print(md_table(results.groupby(["queue", "priority", "sla_hours"]).size().rename("tickets").reset_index()))
    print(f"status counts: {results['status'].value_counts().to_dict()}")
    print(f"premium tickets: {int(results['is_premium'].sum())}; "
          f"tickets from customers with no segment: {int((results['segment_name'] == 'No completed orders').sum())}")

    section("Token budget demonstration")
    full_tokens = int(zero["tokens"].sum())
    budget = 2000
    demo = run_pipeline(tickets, ZERO_SHOT, token_budget=budget)
    demo["budget"] = budget
    demo_columns = ["ticket_id", "status", "category", "tokens", "cumulative_tokens", "budget"]
    assert_no_pii(demo[demo_columns])
    demo[demo_columns].to_csv(f"{OUTPUTS}/triage_budget_demo.csv", index=False)
    spent = int(demo["tokens"].sum())
    print(f"budget {budget} of {full_tokens} needed for a full run; spent {spent}; "
          f"status counts {demo['status'].value_counts().to_dict()}")
    assert spent <= budget

    print("\nwritten: outputs/triage_results.csv, outputs/triage_budget_demo.csv")


if __name__ == "__main__":
    main()
