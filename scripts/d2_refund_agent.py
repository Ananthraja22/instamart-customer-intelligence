"""Part D, tasks 5-8: refund agent. mock_llm.chat proposes tool calls; this code decides what actually happens.

Safety lives in the tools, not the model:
  * issue_refund re-reads the order and recomputes the amount, so a model cannot pay a different amount;
  * refunds above INR 1000 are refused by issue_refund even if a model asks for one;
  * a repeated refund for the same order pays nothing new, so a replay after a crash cannot pay twice;
  * when the caller supplies customer_id, a customer cannot read or refund another customer's order;
  * a final reply claiming a refund or escalation is believed only if the matching tool call succeeded.
"""
import glob
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone

import pandas as pd

import common  # noqa: F401  (puts the repo root on sys.path for mock_llm)
from common import DB, OUTPUTS, RAW, STATE, ensure_dirs, md_table, section
from d1_policy_rag import retrieve
from mock_llm import chat

AUTO_APPROVE_LIMIT_INR = 1000
LATE_AFTER_MINUTES = 45
RUNS_PATH = f"{OUTPUTS}/agent_runs.jsonl"
LEDGER_PATH = f"{OUTPUTS}/refund_ledger.jsonl"
NORMAL_STATUSES = ("REFUND", "ESCALATE", "DENIED", "NOT_FOUND", "NEEDS_INFO")
SYSTEM_PROMPT = ("You are the refund AGENT for a grocery delivery app. Use the tools to look up the order, check "
                 "the refund policy, compute the refund and either issue it or send it to a human. Finish with one "
                 "line that starts with REFUND, ESCALATE, DENIED, NOT_FOUND or NEEDS_INFO, followed by a colon.")


class SimulatedCrash(Exception):
    """Stands in for the process dying."""


def _read_order(order_id):
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    try:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                          (order_id,)).fetchone()
    finally:
        con.close()
    return dict(row) if row else None


def _read_ledger():
    if not os.path.exists(LEDGER_PATH):
        return []
    with open(LEDGER_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_jsonl(file_path, record):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _record_action(tool, order_id, amount):
    """Every call appends one ledger line. A repeat is marked duplicate and moves no new money."""
    earlier = [r for r in _read_ledger() if r["tool"] == tool and r["order_id"] == order_id]
    _append_jsonl(LEDGER_PATH, {"timestamp": datetime.now(timezone.utc).isoformat(), "tool": tool, "order_id": order_id,
                                "refund_inr": amount, "duplicate": bool(earlier)})
    return {"refund_inr": earlier[0]["refund_inr"] if earlier else amount}


def get_order(order_id):
    return _read_order(order_id) or {"error": "order not found"}


def search_policy(query):
    return {"results": retrieve(query)}


def calculate_refund(order_amount, delivery_minutes):
    late = delivery_minutes is not None and float(delivery_minutes) > LATE_AFTER_MINUTES
    return {"refund_inr": round(float(order_amount), 2) if late else 0}


def issue_refund(order_id, refund_inr):
    order = _read_order(order_id)
    if order is None:
        return {"error": "order not found", "refund_inr": 0}
    if order["status"] != "completed":
        return {"error": "only completed orders can be refunded", "refund_inr": 0}
    due = calculate_refund(order["order_amount"], order["delivery_minutes"])["refund_inr"]
    if due == 0 or not abs(float(refund_inr) - due) <= 0.005:
        return {"error": f"requested amount does not match the INR {due} due under policy", "refund_inr": 0}
    if due > AUTO_APPROVE_LIMIT_INR:
        return {"error": "refunds above INR 1000 need a human support lead", "refund_inr": 0}
    return _record_action("issue_refund", order_id, due)


def escalate_to_human(order_id, refund_inr):
    if _read_order(order_id) is None:
        return {"error": "order not found", "refund_inr": 0}
    return _record_action("escalate_to_human", order_id, float(refund_inr))


TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order's status, amount and delivery time in the database.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Search the policy documents for passages relevant to a query.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Compute the refund due under the late-delivery rule.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": ["number", "null"]}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Pay a refund of at most INR 1000 and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human lead and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}
ORDER_TOOLS = {"get_order", "issue_refund", "escalate_to_human"}


def _refusal(name, message):
    return {"error": message} if name == "get_order" else {"error": message, "refund_inr": 0}


def _order_belongs_to(order_id, customer_id):
    con = sqlite3.connect(DB)
    try:
        return con.execute("SELECT 1 FROM orders WHERE order_id = ? AND customer_id = ?", (order_id, customer_id)).fetchone() is not None
    finally:
        con.close()


def execute_tool(name, arguments, customer_id=None):
    """Whatever a model requests, a bad request becomes an error observation and never crashes the run."""
    if name not in TOOL_MAP:
        return _refusal(name, f"unknown tool {name}")
    try:
        if customer_id and name in ORDER_TOOLS and not _order_belongs_to(arguments.get("order_id"), customer_id):
            return _refusal(name, "order not found")  # A wrong owner looks like a missing order, so ids cannot be probed.
        return TOOL_MAP[name](**arguments)
    except (TypeError, ValueError, AttributeError, sqlite3.Error):
        return _refusal(name, f"invalid arguments for {name}")


def _checkpoint_path(run_id):
    return os.path.join(STATE, f"{run_id}.json")


def _save_checkpoint(state):
    """Write then rename, so a crash can never leave a half-written checkpoint behind."""
    temporary = _checkpoint_path(state["run_id"]) + ".tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, _checkpoint_path(state["run_id"]))


def _successful_action(state, tool):
    """The observation of the last successful call to this tool in the run, or None."""
    found = [s["observation"] for s in state["steps"] if s["tool"] == tool and "error" not in s["observation"]]
    return found[-1] if found else None


def _result(state):
    paid = _successful_action(state, "issue_refund")
    sent = _successful_action(state, "escalate_to_human")
    action = paid or sent
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"], "status": state["status"],
            "final_answer": state["final_answer"], "steps": state["steps"], "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"], "refund_inr": action["refund_inr"] if action else 0,
            "tools_executed": state["tools_executed"]}


def _finish(state, status, final_answer):
    state.update(status=status, final_answer=final_answer)
    _save_checkpoint(state)
    return _result(state)


def _handoff(state, status, reason):
    """Stop safely with a short summary a human can act on."""
    tools = ", ".join(state["tools_executed"]) or "none"
    paid = _successful_action(state, "issue_refund")
    money = f"INR {paid['refund_inr']:.2f} is already refunded, so do not pay it again" if paid else "No refund was issued"
    return _finish(state, status, f"Handing to a human: {reason} after {state['llm_calls']} model calls. Tools run: {tools}. {money}.")


def _finish_from_reply(state, text):
    """The normal status is the word before the colon, believed only if the tool records agree."""
    match = re.match(r"([A-Z_]+):", text or "")
    status = match.group(1) if match and match.group(1) in NORMAL_STATUSES else None
    if status is None:
        return _handoff(state, "FALLBACK_UNVERIFIED_REPLY", "the model's final reply was not recognised")
    if status == "REFUND" and not _successful_action(state, "issue_refund"):
        return _handoff(state, "FALLBACK_UNVERIFIED_REPLY", "the model claimed a refund no tool issued")
    if status == "ESCALATE" and not _successful_action(state, "escalate_to_human"):
        return _handoff(state, "FALLBACK_UNVERIFIED_REPLY", "the model claimed an escalation no tool made")
    return _finish(state, status, text)


def _loop(state, crash_after_tools=None):
    started = time.monotonic()
    while state["llm_calls"] < state["max_iterations"]:
        if time.monotonic() - started >= state["time_limit_seconds"]:
            return _handoff(state, "FALLBACK_TIMEOUT", f"the {state['time_limit_seconds']}-second time limit was reached")

        response = chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += response.usage.total_tokens
        if state["tokens_used"] > state["token_limit"]:  # Stop before running the tool just requested.
            return _handoff(state, "FALLBACK_TOKEN_LIMIT", f"the {state['token_limit']}-token limit was exceeded")

        message = response.choices[0].message
        if not message.tool_calls:
            return _finish_from_reply(state, message.content)

        calls = [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in message.tool_calls]
        state["messages"].append({"role": "assistant", "content": None, "tool_calls": calls})
        for call in calls:
            observation = execute_tool(call["name"], call["arguments"], state["customer_id"])
            state["messages"].append({"role": "tool", "tool_call_id": call["id"], "name": call["name"],
                                      "content": json.dumps(observation)})
            state["tools_executed"].append(call["name"])
            state["step"] += 1
            state["steps"].append({"step": state["step"], "tool": call["name"], "arguments": call["arguments"],
                                   "observation": observation})
            _save_checkpoint(state)
            if crash_after_tools == len(state["tools_executed"]):
                raise SimulatedCrash(f"after tool {len(state['tools_executed'])}")
    return _handoff(state, "FALLBACK_MAX_ITERATIONS", f"the limit of {state['max_iterations']} model calls was reached")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              customer_id=None, run_id=None, crash_after_tools=None):
    """Returns status, final_answer, steps, llm_calls and tokens_used, plus refund_inr and tools_executed."""
    state = {"run_id": run_id or scenario_id, "scenario_id": scenario_id, "customer_id": customer_id, "status": "running",
             "final_answer": None, "max_iterations": max_iterations, "token_limit": token_limit,
             "time_limit_seconds": time_limit_seconds, "tokens_used": 0, "llm_calls": 0, "step": 0, "steps": [],
             "tools_executed": [], "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}]}
    return _loop(state, crash_after_tools)


def resume_agent(run_id):
    """Rebuilds the run from its checkpoint on disk; nothing is carried over in memory."""
    with open(_checkpoint_path(run_id), encoding="utf-8") as f:
        state = json.load(f)
    return _result(state) if state["status"] != "running" else _loop(state)


def record_run(result, **extra):
    fields = ["run_id", "scenario_id", "status", "final_answer", "llm_calls", "tokens_used", "refund_inr", "steps"]
    _append_jsonl(RUNS_PATH, {**{k: result[k] for k in fields}, **extra})


def reset_outputs():
    """Each full run starts from an empty ledger, so the published results are reproducible."""
    ensure_dirs(OUTPUTS, STATE)
    for file_path in [RUNS_PATH, LEDGER_PATH] + glob.glob(os.path.join(STATE, "*.json")):
        if os.path.exists(file_path):
            os.remove(file_path)


def main():
    reset_outputs()
    scenarios = pd.read_csv(f"{RAW}/agent_scenarios.csv", keep_default_na=False)
    by_id = {s.scenario_id: s for s in scenarios.itertuples(index=False)}

    section("Task 6: seven scenarios")
    rows = []
    for s in scenarios.itertuples(index=False):
        result = run_agent(s.message, s.scenario_id, customer_id=s.customer_id)
        record_run(result)
        rows.append({"scenario": s.scenario_id, "status": result["status"], "refund_inr": float(result["refund_inr"]),
                     "expected_inr": s.expected_refund_inr, "model_calls": result["llm_calls"], "tokens_used": result["tokens_used"],
                     "tools": ", ".join(result["tools_executed"])})
        assert result["status"].startswith(s.expected_outcome), (s.scenario_id, result["status"])
        assert abs(float(result["refund_inr"]) - s.expected_refund_inr) <= 0.01, s.scenario_id
    table = pd.DataFrame(rows).set_index("scenario")
    print(md_table(table.reset_index()))
    assert table.loc["S1", "model_calls"] == 5 and table.loc["S5", "model_calls"] == 3 and table.loc["S6", "model_calls"] == 2
    assert table.loc["S7", "status"] == "FALLBACK_MAX_ITERATIONS" and table.loc["S7", "model_calls"] == 6

    section("Task 7: stop safely")
    s1 = by_id["S1"]
    token_demo = run_agent(s1.message, "S1_token_limit_demo", token_limit=200, customer_id=s1.customer_id)
    record_run(token_demo)
    print(f"S1 with token_limit=200: {token_demo['status']}; model calls {token_demo['llm_calls']}; tokens {token_demo['tokens_used']}; "
          f"tools run {token_demo['tools_executed']}")
    print(f"  {token_demo['final_answer']}")
    assert token_demo["status"] == "FALLBACK_TOKEN_LIMIT"
    time_demo = run_agent(s1.message, "S1_timeout_demo", time_limit_seconds=0, customer_id=s1.customer_id)
    record_run(time_demo)
    print(f"S1 with time_limit_seconds=0: {time_demo['status']}; model calls {time_demo['llm_calls']}; tools run {time_demo['tools_executed']}")
    print(f"  {time_demo['final_answer']}")
    assert time_demo["status"] == "FALLBACK_TIMEOUT" and time_demo["llm_calls"] == 0

    section("Task 8: crash and resume with S1")
    run_id = "S1_crash_resume_demo"
    try:
        run_agent(s1.message, run_id, customer_id=s1.customer_id, crash_after_tools=2)
        raise AssertionError("the crash did not happen")
    except SimulatedCrash as crash:
        print(f"simulated crash: {crash}")
    with open(_checkpoint_path(run_id), encoding="utf-8") as f:
        checkpoint = json.load(f)
    print(f"tools in the checkpoint at the crash: {checkpoint['tools_executed']}")
    resumed = resume_agent(run_id)
    record_run(resumed)
    print(f"tools executed across the crash and resume: {resumed['tools_executed']}")
    print(f"final status {resumed['status']}; refund INR {resumed['refund_inr']}; model calls {resumed['llm_calls']}")
    assert resumed["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert resumed["status"] == "REFUND" and abs(resumed["refund_inr"] - 128.25) <= 0.01

    section("Guards against requests a misbehaving model might make")
    big = by_id["S3"]
    big_order = re.search(r"ORD-\d{6}", big.message).group(0)
    paid_order = re.search(r"ORD-\d{6}", s1.message).group(0)
    size_before = len(_read_ledger())
    checks = [("another customer's order", execute_tool("issue_refund", {"order_id": big_order, "refund_inr": big.expected_refund_inr}, "C0001")),
              ("another customer's refunded order", execute_tool("get_order", {"order_id": paid_order}, "C0001")),
              ("refund above INR 1000", execute_tool("issue_refund", {"order_id": big_order, "refund_inr": big.expected_refund_inr}, big.customer_id)),
              ("wrong amount", execute_tool("issue_refund", {"order_id": paid_order, "refund_inr": 1.0}, s1.customer_id)),
              ("unknown tool", execute_tool("delete_everything", {})),
              ("missing arguments", execute_tool("issue_refund", {}))]
    print(md_table(pd.DataFrame([{"attempt": label, "result": r.get("error")} for label, r in checks])))
    assert all("error" in r for _, r in checks) and len(_read_ledger()) == size_before

    section("Ledger")
    ledger = pd.DataFrame(_read_ledger())
    print(md_table(ledger.drop(columns="timestamp")))
    issued = ledger[(ledger["tool"] == "issue_refund") & ~ledger["duplicate"]]
    print(f"money actually refunded: INR {issued['refund_inr'].sum():,.2f} across {len(issued)} orders")
    checkpoints = len(glob.glob(os.path.join(STATE, "*.json")))
    print(f"\nwritten: outputs/agent_runs.jsonl, outputs/refund_ledger.jsonl, state/*.json ({checkpoints} checkpoints)")


if __name__ == "__main__":
    main()
