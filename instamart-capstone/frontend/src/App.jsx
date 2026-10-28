import { useEffect, useState } from "react";

const API = "http://127.0.0.1:8000";

export default function App() {
  const [question, setQuestion] = useState("");
  const [customerId, setCustomerId] = useState("");
  const [orderId, setOrderId] = useState("");
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const [apiStatus, setApiStatus] = useState("checking");

  useEffect(() => {
    fetch(API + "/api/health")
      .then((response) => response.json())
      .then(() => setApiStatus("online"))
      .catch(() => setApiStatus("offline"));
  }, []);

  async function submit(event) {
    event.preventDefault();
    const prompt = question.trim();
    if (!prompt || busy) return;

    setMessages((current) => [...current, { role: "user", text: prompt }]);
    setQuestion("");
    setBusy(true);

    try {
      const response = await fetch(API + "/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: prompt,
          customer_id: customerId.trim(),
          order_id: orderId.trim(),
        }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "The request failed.");
      setMessages((current) => [...current, { role: "assistant", result }]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        { role: "error", text: error.message },
      ]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand-mark" aria-hidden="true">i</div>
        <div>
          <p className="eyebrow">SYNTHETIC DATA DEMO</p>
          <h1>Instamart Support Copilot</h1>
        </div>
        <span className={"status status-" + apiStatus}>
          <span className="status-dot" />
          API {apiStatus}
        </span>
      </header>

      <section className="intro">
        <p className="eyebrow">POLICY RAG | REFUND TOOL</p>
        <h2>How can I help?</h2>
        <p>
          Ask about a support policy, or enter synthetic order details to check
          the refund rules. The assistant returns policy sources with its answer.
        </p>
      </section>

      <section className="conversation" aria-live="polite">
        {messages.length === 0 && (
          <div className="suggestions">
            <button onClick={() => setQuestion("How long does a refund take?")}>
              How long does a refund take?
            </button>
            <button onClick={() => setQuestion("When is a late delivery refundable?")}>
              When is a late delivery refundable?
            </button>
          </div>
        )}
        {messages.map((message, index) => (
          <article className={"message message-" + message.role} key={index}>
            <div className="message-label">
              {message.role === "user"
                ? "You"
                : message.role === "error"
                  ? "Error"
                  : "Copilot"}
            </div>
            {message.result ? (
              <>
                <p>{message.result.answer}</p>
                {message.result.sources?.length > 0 && (
                  <div className="sources">
                    <span>Policy sources</span>
                    {message.result.sources.map((source) => (
                      <span className="source-chip" key={source.chunk_id}>
                        {source.document} - {source.chunk_id}
                      </span>
                    ))}
                  </div>
                )}
                {message.result.refund_decision && (
                  <div className="refund-card">
                    <strong>Refund check: {message.result.refund_decision.outcome}</strong>
                    <span>
                      INR {Number(message.result.refund_decision.refund_inr || 0).toFixed(2)}
                    </span>
                    <small>{message.result.refund_decision.reason}</small>
                  </div>
                )}
                <small className="mode">
                  Answer mode: {message.result.generation_mode}
                </small>
              </>
            ) : (
              <p>{message.text}</p>
            )}
          </article>
        ))}
        {busy && <div className="typing">Searching policy chunks...</div>}
      </section>

      <form className="composer" onSubmit={submit}>
        <div className="tool-fields">
          <label>
            Synthetic customer ID
            <input
              value={customerId}
              onChange={(event) => setCustomerId(event.target.value)}
              placeholder="C0007"
              autoComplete="off"
            />
          </label>
          <label>
            Synthetic order ID
            <input
              value={orderId}
              onChange={(event) => setOrderId(event.target.value)}
              placeholder="ORD-100048"
              autoComplete="off"
            />
          </label>
        </div>
        <div className="question-row">
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Type a policy or refund question..."
            maxLength={2000}
            rows={2}
          />
          <button className="send-button" type="submit" disabled={busy || !question.trim()}>
            {busy ? "Working..." : "Ask"}
          </button>
        </div>
        <p className="privacy-note">
          Demo only. Order/customer IDs are used by local tools and removed from
          model prompts. Do not enter real customer information.
        </p>
      </form>
    </main>
  );
}
