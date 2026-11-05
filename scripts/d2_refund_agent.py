"""Part D: refund agent. mock_llm.chat proposes tool calls; this host decides what actually happens.

Safety lives in the host, not the model:
  * the customer id comes from the session, so a model cannot read or refund another customer's order;
  * issue_refund re-reads the order from SQLite and recomputes the amount, ignoring what the model claims;
  * refunds above INR 1000 are refused here even if a model asks for them;
  * the ledger is keyed by order_id, so replays cannot pay twice;
  * every step is checkpointed to state/<run_id>.json, so a crashed run resumes without repeating work;
  * a final message is only believed if the ledger or escalation record agrees with it.
"""
import glob
import json
import os
import re
import sqlite3

import pandas as pd

import common  # noqa: F401  (puts the repo root on sys.path for mock_llm)
from common import DB, OUTPUTS, RAW, STATE, ensure_dirs, md_table, section
from d1_policy_rag import PolicyIndex
from mock_llm import chat, est_tokens

MAX_STEPS = 6
AUTO_APPROVE_LIMIT_INR = 1000
LATE_AFTER_MINUTES = 45
RUNS_PATH = f"{OUTPUTS}/agent_runs.jsonl"
LEDGER_PATH = f"{OUTPUTS}/refund_ledger.jsonl"
OUTCOMES = ("REFUND", "ESCALATE", "DENIED", "NOT_FOUND", "NEEDS_INFO")
SYSTEM_PROMPT = ("You are the refund AGENT for a grocery delivery app. Use the tools to look up the order, check "
                 "the refund policy, compute the refund and either issue it or send it to a human. Finish with one "
                 "line that starts with REFUND, ESCALATE, DENIED, NOT_FOUND or NEEDS_INFO.")
TOOL_SCHEMAS = [
    {"name": "get_order", "description": "Look up one of this customer's orders.", "parameters": {"order_id": "string"}},
    {"name": "search_policy", "description": "Search the policy documents.", "parameters": {"query": "string"}},
    {"name": "calculate_refund", "description": "Refund due under the late-delivery rule.",
     "parameters": {"order_amount": "number", "delivery_minutes": "number"}},
    {"name": "issue_refund", "description": "Pay a refund of at most INR 1000.",
     "parameters": {"order_id": "string", "refund_inr": "number"}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human lead.",
     "parameters": {"order_id": "string", "refund_inr": "number"}},
]


class SimulatedCrash(BaseException):
    """Stands in for the process dying. BaseException, so no ordinary handler can absorb it."""


def read_ledger():
    if not os.path.exists(LEDGER_PATH):
        return []
    with open(LEDGER_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(file_path, record):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


class Tools:
    """The only code that touches the database and the ledger. customer_id is bound at construction."""

    def __init__(self, customer_id, run_id, index):
        self.customer_id, self.run_id, self.index = customer_id, run_id, index

    def _load_order(self, order_id):
        con = sqlite3.connect(DB)
        con.row_factory = sqlite3.Row
        try:
            row = con.execute(
                "SELECT order_id, customer_id, status, order_amount, delivery_minutes, delivery_imputed "
                "FROM orders WHERE order_id = ? AND customer_id = ?", (order_id, self.customer_id)).fetchone()
        finally:
            con.close()
        return dict(row) if row else None

    def get_order(self, order_id):
        # A wrong owner and a missing order look identical, so ids cannot be probed.
        return self._load_order(order_id) or {"error": "order not found"}

    def search_policy(self, query):
        hits = self.index.search(query)
        return {"results": [{"source": h["source"], "text": h["text"]} for h in hits]}

    @staticmethod
    def calculate_refund(order_amount, delivery_minutes):
        amount = float(order_amount)
        late = delivery_minutes is not None and float(delivery_minutes) > LATE_AFTER_MINUTES
        return {"refund_inr": round(amount, 2) if late else 0, "rule": "full refund if delivered in more than 45 minutes"}

    def issue_refund(self, order_id, refund_inr):
        order = self._load_order(order_id)
        refused = {"error": None, "refund_inr": 0, "refused": True}
        if order is None:  # Ownership comes first, so nobody can read another customer's refund.
            return {**refused, "error": "order not found"}
        for row in read_ledger():
            if row["order_id"] == order_id:  # Idempotent: a replay returns the original payment.
                return {"refund_inr": row["refund_inr"], "refund_id": row["refund_id"], "duplicate": True}
        if order["status"] != "completed":
            return {**refused, "error": "only completed orders can be refunded"}
        due = self.calculate_refund(order["order_amount"], order["delivery_minutes"])["refund_inr"]
        if due == 0 or abs(float(refund_inr) - due) > 0.005:
            return {**refused, "error": f"requested amount does not match the INR {due} due under policy"}
        if due > AUTO_APPROVE_LIMIT_INR:
            return {**refused, "error": "refunds above INR 1000 need a human support lead"}
        row = {"refund_id": f"RF-{order_id}", "order_id": order_id, "customer_id": self.customer_id,
               "refund_inr": due, "run_id": self.run_id, "status": "issued"}
        append_jsonl(LEDGER_PATH, row)
        return {"refund_inr": due, "refund_id": row["refund_id"], "duplicate": False}

    def escalate_to_human(self, order_id, refund_inr):
        order = self._load_order(order_id)
        if order is None:
            return {"error": "order not found", "refund_inr": 0}
        return {"status": "escalated", "order_id": order_id, "refund_inr": float(refund_inr), "queue": "human_support_lead"}

    def execute(self, name, arguments):
        if name not in {t["name"] for t in TOOL_SCHEMAS}:
            return {"error": f"unknown tool {name}"}
        return getattr(self, name)(**arguments)


def state_path(run_id):
    return os.path.join(STATE, f"{run_id}.json")


def save_state(state):
    """Write then rename, so a crash can never leave a half-written checkpoint behind."""
    temporary = state_path(state["run_id"]) + ".tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, state_path(state["run_id"]))


def new_state(run_id, customer_id, message):
    return {"run_id": run_id, "customer_id": customer_id, "status": "running", "steps": 0, "total_tokens": 0,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": message}],
            "tools_started": [], "tools_completed": [], "outcome": None, "refund_inr": 0.0, "final_message": None,
            "summary": None}


def stop_run(state, outcome, final_message, summary=None):
    state.update(status="finished", outcome=outcome, final_message=final_message, summary=summary)
    save_state(state)
    return state


def human_summary(state, reason):
    tools = ", ".join(state["tools_completed"]) or "none"
    return (f"{reason} after {state['steps']} model steps. Tools run: {tools}. No refund was issued by this run; "
            f"a human agent should review the request.")


def prompt_tokens(messages):
    return sum(est_tokens(m.get("content") or json.dumps(m.get("tool_calls", ""))) for m in messages)


def run_tool_call(state, tools, call, crash_after_tool):
    """Record intent, execute, optionally crash before saving the result, then record the result."""
    state["tools_started"].append(call["name"])
    save_state(state)
    result = tools.execute(call["name"], call["arguments"])
    if crash_after_tool == call["name"]:
        raise SimulatedCrash(call["name"])
    state["messages"].append({"role": "tool", "tool_call_id": call["id"], "name": call["name"],
                              "content": json.dumps(result)})
    state["tools_completed"].append(call["name"])
    save_state(state)


def finish_from_reply(state, text):
    """Believe the model's outcome only when the host's own records back it up."""
    match = re.match(r"([A-Z_]+):", text or "")
    outcome = match.group(1) if match and match.group(1) in OUTCOMES else None
    results = {m["name"]: json.loads(m["content"]) for m in state["messages"] if m["role"] == "tool"}
    if outcome == "REFUND":
        paid = results.get("issue_refund", {})
        if not paid.get("refund_id"):
            return stop_run(state, "FALLBACK", text, human_summary(state, "Reply claimed a refund the ledger does not hold"))
        state["refund_inr"] = paid["refund_inr"]
    elif outcome == "ESCALATE":
        sent = results.get("escalate_to_human", {})
        if sent.get("status") != "escalated":
            return stop_run(state, "FALLBACK", text, human_summary(state, "Reply claimed an escalation that did not happen"))
        state["refund_inr"] = sent["refund_inr"]
    elif outcome is None:
        return stop_run(state, "FALLBACK", text, human_summary(state, "Unrecognised final reply"))
    return stop_run(state, outcome, text)


def advance(state, tools, max_steps, token_budget, crash_after_tool):
    # Resume path: the checkpoint may hold tool requests whose results were lost in the crash.
    for call in pending_calls(state):
        run_tool_call(state, tools, call, None)

    while state["status"] == "running":
        if state["steps"] >= max_steps:
            return stop_run(state, "FALLBACK", None, human_summary(state, f"Stopped at the {max_steps}-step limit"))
        if token_budget is not None and state["total_tokens"] + prompt_tokens(state["messages"]) > token_budget:
            return stop_run(state, "STOPPED_BUDGET", None, human_summary(state, f"Stopped at the {token_budget}-token budget"))

        response = chat(state["messages"], tools=TOOL_SCHEMAS)
        state["steps"] += 1
        state["total_tokens"] += response.usage.total_tokens
        message = response.choices[0].message
        if not message.tool_calls:
            return finish_from_reply(state, message.content)

        calls = [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in message.tool_calls]
        state["messages"].append({"role": "assistant", "content": None, "tool_calls": calls})
        save_state(state)
        for call in calls:
            run_tool_call(state, tools, call, crash_after_tool)
    return state


def pending_calls(state):
    """Tool requests from the latest assistant turn that have no saved result."""
    answered = {m["tool_call_id"] for m in state["messages"] if m["role"] == "tool"}
    assistant_turns = [m for m in state["messages"] if m["role"] == "assistant"]
    if not assistant_turns:
        return []
    return [c for c in assistant_turns[-1]["tool_calls"] if c["id"] not in answered]


def run_agent(run_id, customer_id, message, index, max_steps=MAX_STEPS, token_budget=None, crash_after_tool=None):
    state = new_state(run_id, customer_id, message)
    save_state(state)
    return advance(state, Tools(customer_id, run_id, index), max_steps, token_budget, crash_after_tool)


def resume_agent(run_id, index, max_steps=MAX_STEPS):
    """Rebuilds everything from the checkpoint on disk; nothing is carried over in memory."""
    with open(state_path(run_id), encoding="utf-8") as f:
        state = json.load(f)
    return advance(state, Tools(state["customer_id"], run_id, index), max_steps, None, None)


def record_run(state, kind, **extra):
    fields = ["run_id", "customer_id", "outcome", "refund_inr", "steps", "total_tokens", "tools_completed",
              "final_message", "summary"]
    append_jsonl(RUNS_PATH, {"kind": kind, **{k: state[k] for k in fields}, **extra})


def demo_orders(scenarios, n):
    """Refund-eligible orders that no scenario uses, with measured (not imputed) delivery times."""
    used = set(re.findall(r"ORD-\d{6}", " ".join(scenarios["message"])))
    con = sqlite3.connect(DB)
    try:
        frame = pd.read_sql_query(
            "SELECT order_id, customer_id, order_amount, delivery_minutes FROM orders "
            "WHERE status = 'completed' AND delivery_imputed = 0 AND delivery_minutes > ? AND order_amount <= ? "
            "ORDER BY order_id", con, params=(LATE_AFTER_MINUTES, AUTO_APPROVE_LIMIT_INR))
    finally:
        con.close()
    return frame[~frame["order_id"].isin(used)].head(n).to_dict("records")


def reset_outputs():
    """Each run starts from an empty ledger so the published results are reproducible."""
    ensure_dirs(OUTPUTS, STATE)
    for file_path in [RUNS_PATH, LEDGER_PATH] + glob.glob(os.path.join(STATE, "*.json")):
        if os.path.exists(file_path):
            os.remove(file_path)


def main():
    reset_outputs()
    index = PolicyIndex()
    scenarios = pd.read_csv(f"{RAW}/agent_scenarios.csv", keep_default_na=False)

    section("Seven scenarios")
    rows = []
    for s in scenarios.itertuples(index=False):
        state = run_agent(s.scenario_id, s.customer_id, s.message, index)
        record_run(state, "scenario", expected_outcome=s.expected_outcome, expected_refund_inr=s.expected_refund_inr)
        rows.append({"scenario": s.scenario_id, "customer": s.customer_id, "expected": s.expected_outcome,
                     "outcome": state["outcome"], "expected_inr": s.expected_refund_inr, "refund_inr": float(state["refund_inr"]),
                     "steps": state["steps"], "tokens": state["total_tokens"], "tools": " > ".join(state["tools_completed"])})
    table = pd.DataFrame(rows)
    print(md_table(table))
    table["passed"] = (table["expected"] == table["outcome"]) & ((table["expected_inr"] - table["refund_inr"]).abs() < 0.005)
    print(f"passed: {int(table['passed'].sum())}/7")
    assert table["passed"].all()

    section("Idempotency: replay every scenario, ledger must not grow")
    before = read_ledger()
    for s in scenarios.itertuples(index=False):
        state = run_agent(f"{s.scenario_id}-replay", s.customer_id, s.message, index)
        record_run(state, "idempotency_replay")
    after = read_ledger()
    print(f"ledger rows before replay {len(before)}; after replay {len(after)}; "
          f"INR {sum(r['refund_inr'] for r in after):,.2f} paid in total")
    assert before == after

    step_order, budget_order, crash_order = demo_orders(scenarios, 3)

    section("Stop-safely demonstration 1: step limit")
    state = run_agent("STOP-steps", step_order["customer_id"], f"My order {step_order['order_id']} was late, refund it.",
                      index, max_steps=3)
    record_run(state, "stop_safely")
    print(f"order {step_order['order_id']}: status {state['outcome']}; steps {state['steps']}; refund INR {state['refund_inr']}")
    print(f"summary: {state['summary']}")
    assert state["outcome"] == "FALLBACK" and step_order["order_id"] not in {r["order_id"] for r in read_ledger()}

    section("Stop-safely demonstration 2: token budget")
    state = run_agent("STOP-budget", budget_order["customer_id"], f"My order {budget_order['order_id']} was late, refund it.",
                      index, token_budget=400)
    record_run(state, "stop_safely")
    print(f"order {budget_order['order_id']}: status {state['outcome']}; tokens {state['total_tokens']}; refund INR {state['refund_inr']}")
    print(f"summary: {state['summary']}")
    assert state["outcome"] == "STOPPED_BUDGET" and budget_order["order_id"] not in {r["order_id"] for r in read_ledger()}

    section("Crash and resume: process dies right after the ledger write, before the checkpoint")
    run_id = "CRASH-resume"
    message = f"My order {crash_order['order_id']} was late, refund it."
    try:
        run_agent(run_id, crash_order["customer_id"], message, index, crash_after_tool="issue_refund")
        raise AssertionError("the crash did not happen")
    except SimulatedCrash:
        pass
    with open(state_path(run_id), encoding="utf-8") as f:
        crashed = json.load(f)
    paid_before = [r for r in read_ledger() if r["order_id"] == crash_order["order_id"]]
    print(f"tools started before the crash: {crashed['tools_started']}")
    print(f"tool results saved before the crash: {crashed['tools_completed']}")
    print(f"ledger rows for this order at crash time: {len(paid_before)}")

    state = resume_agent(run_id, index)
    record_run(state, "crash_resume", tools_started=state["tools_started"])
    resumed_tools = state["tools_started"][len(crashed["tools_started"]):]
    paid_after = [r for r in read_ledger() if r["order_id"] == crash_order["order_id"]]
    print(f"tools executed after resume: {resumed_tools}")
    print(f"outcome {state['outcome']}; refund INR {state['refund_inr']}; ledger rows for this order: {len(paid_after)}")
    assert state["outcome"] == "REFUND" and len(paid_before) == len(paid_after) == 1
    assert resumed_tools == ["issue_refund"], "resume repeated work it should have skipped"

    section("Host guards: requests a misbehaving model might make")
    big = scenarios[scenarios["expected_outcome"] == "ESCALATE"].iloc[0]
    big_id = re.search(r"ORD-\d{6}", big["message"]).group(0)
    already_paid = read_ledger()[0]
    outsider = Tools("C0001", "GUARD", index)
    checks = [
        ("another customer's unpaid order", outsider.issue_refund(big_id, float(big["expected_refund_inr"]))),
        ("another customer's already-refunded order", outsider.issue_refund(already_paid["order_id"], already_paid["refund_inr"])),
        ("refund above INR 1000 by the owner", Tools(big["customer_id"], "GUARD", index).issue_refund(big_id, float(big["expected_refund_inr"]))),
        ("wrong amount on an eligible order", Tools(step_order["customer_id"], "GUARD", index).issue_refund(step_order["order_id"], 1.0)),
    ]
    size_before = len(read_ledger())
    print(md_table(pd.DataFrame([{"attempt": label, "result": r.get("error")} for label, r in checks])))
    assert all(r.get("refused") for _, r in checks) and len(read_ledger()) == size_before

    section("Ledger")
    ledger = pd.DataFrame(read_ledger())
    print(md_table(ledger))
    assert ledger["order_id"].is_unique and (ledger["refund_inr"] <= AUTO_APPROVE_LIMIT_INR).all()
    print(f"\nwritten: outputs/agent_runs.jsonl, outputs/refund_ledger.jsonl, state/*.json ({len(glob.glob(os.path.join(STATE, '*.json')))} checkpoints)")


if __name__ == "__main__":
    main()
