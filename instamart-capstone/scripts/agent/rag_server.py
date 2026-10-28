"""Local HTTP API for the React support copilot and policy RAG workflow."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

from policy_assistant import load_policy_chunks, retrieve_policy_chunks
from refund_agent import decide as decide_refund

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "state" / "instamart.db"
CLEAN_ORDERS = PROJECT_ROOT / "data" / "clean" / "orders.csv"
POLICY_CHUNKS = load_policy_chunks()

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
ORDER_ID_RE = re.compile(r"\bORD[-_ ]?\d+\b", re.I)
CUSTOMER_ID_RE = re.compile(r"\bC\d{3,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d .()/-]{7,}\d)(?!\w)")
ORDER_LOOKUP_RE = re.compile(r"\bORD[-_ ]?\d+\b", re.I)


def redact_for_model(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = ORDER_ID_RE.sub("[ORDER_ID]", text)
    return CUSTOMER_ID_RE.sub("[CUSTOMER_ID]", text)


def ollama_grounded_answer(question: str, retrieved: list[dict]) -> str | None:
    model = os.environ.get("OLLAMA_MODEL", "").strip()
    if not model:
        return None
    context = "\n\n".join(
        f"Source {chunk['document']} ({chunk['chunk_id']}): {chunk['text']}"
        for chunk in retrieved
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Answer only from the provided synthetic policy context. If the "
                "context does not answer the question, say so. Cite source file "
                "names. Do not invent policy or claim an action was executed."
            ),
        },
        {
            "role": "user",
            "content": f"Policy context:\n{context}\n\nQuestion:\n{question}",
        },
    ]
    prompt_bytes = sum(len(message["content"].encode("utf-8")) for message in messages)
    if prompt_bytes > 8192:
        raise ValueError("Policy prompt exceeded the configured 8192-byte safety cap.")

    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    request = Request(
        f"{base}/api/chat",
        data=json.dumps(
            {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0},
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return str(payload.get("message", {}).get("content", "")).strip() or None
    except (URLError, TimeoutError, json.JSONDecodeError):
        return None


def process_question(body: dict) -> dict:
    question = str(body.get("question", "")).strip()
    if not question:
        raise ValueError("Enter a question.")
    if len(question.encode("utf-8")) > 2048:
        raise ValueError("Question is too long; maximum accepted size is 2048 bytes.")

    # IDs are extracted for local tool lookup only. The model question is
    # redacted before it is sent to the optional local Ollama service.
    order_match = ORDER_LOOKUP_RE.search(question)
    order_id = str(body.get("order_id") or (order_match.group(0) if order_match else ""))
    customer_id = str(body.get("customer_id", "")).strip()
    sanitized_question = redact_for_model(question)

    retrieved = retrieve_policy_chunks(sanitized_question, POLICY_CHUNKS, top_k=3)
    if not retrieved:
        answer = "I could not find a relevant fact in the supplied policies."
        generation_mode = "no-match"
    else:
        answer = ollama_grounded_answer(sanitized_question, retrieved)
        generation_mode = "local-ollama-rag" if answer else "extractive-rag"
        if not answer:
            answer = retrieved[0]["text"]

    refund_decision = None
    lowered = question.casefold()
    asks_refund = any(term in lowered for term in ("refund", "money back", "reimburse"))
    if asks_refund:
        if not CLEAN_ORDERS.exists():
            raise RuntimeError("Clean order data is missing; run the project pipeline first.")
        if not order_id or not customer_id:
            refund_decision = {
                "outcome": "NEED_VERIFICATION",
                "refund_inr": 0.0,
                "reason": "Provide the synthetic customer ID and order ID for local ownership checks.",
            }
        else:
            orders = pd.read_csv(CLEAN_ORDERS).fillna("")
            refund_decision = decide_refund(
                f"Refund request for {order_id}",
                customer_id,
                orders,
            )

    return {
        "answer": answer,
        "sources": [
            {
                "document": item["document"],
                "chunk_id": item["chunk_id"],
                "score": round(item["score"], 3),
            }
            for item in retrieved
        ],
        "generation_mode": generation_mode,
        "refund_decision": refund_decision,
        "privacy": {
            "identifiers_redacted_before_model": True,
            "order_lookup_performed_locally": bool(refund_decision),
        },
    }


class CopilotHandler(BaseHTTPRequestHandler):
    server_version = "InstamartSupportCopilot/1.0"

    def _send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "http://localhost:5173")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:
        self._send_json(200, {"ok": True})

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self._send_json(200, {"status": "ok", "policy_chunks": len(POLICY_CHUNKS)})
        elif self.path == "/openapi.json":
            self._send_json(
                200,
                {
                    "openapi": "3.0.0",
                    "info": {
                        "title": "Instamart Support Copilot API",
                        "version": "1.0.0",
                    },
                    "paths": {
                        "/api/chat": {
                            "post": {
                                "summary": "Retrieve policy context and evaluate synthetic refund requests",
                                "requestBody": {
                                    "required": True,
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "required": ["question"],
                                                "properties": {
                                                    "question": {"type": "string"},
                                                    "customer_id": {"type": "string"},
                                                    "order_id": {"type": "string"},
                                                },
                                            }
                                        }
                                    },
                                },
                                "responses": {"200": {"description": "Grounded answer and sources"}},
                            }
                        }
                    },
                },
            )
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self._send_json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 8192:
                raise ValueError("Request body must be between 1 and 8192 bytes.")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("Request body must be a JSON object.")
            self._send_json(200, process_question(body))
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def log_message(self, format: str, *args: object) -> None:
        # Avoid logging raw request bodies or personal information.
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    port = int(os.environ.get("COPILOT_API_PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), CopilotHandler)
    print(f"Support Copilot API listening on http://127.0.0.1:{port}")
    print(f"Health check: http://127.0.0.1:{port}/api/health")
    server.serve_forever()


if __name__ == "__main__":
    main()
