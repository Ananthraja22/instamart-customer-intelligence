"""Part D, tasks 1-4: chunk the policies, retrieve with TF-IDF, answer through mock_llm.chat, and evaluate.
The refund agent (d2_refund_agent.py) imports retrieve() from here, so both share one retrieval path.
"""
import glob
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import common  # noqa: F401  (puts the repo root on sys.path for mock_llm)
from common import POLICIES, RAW, md_table, section
from mock_llm import chat

CHUNK_WORDS = 40
OVERLAP_WORDS = 10
EXPECTED_CHUNKS = {"delivery_policy.txt": 3, "escalation_policy.txt": 3, "payment_policy.txt": 3,
                   "privacy_policy.txt": 2, "refund_policy.txt": 4}
SYSTEM_PROMPT = ("You are the POLICY ASSISTANT for a grocery delivery app. Answer only from the CONTEXT supplied. "
                 "If the CONTEXT does not contain the answer, say that you could not find the information.")


def chunk_document(source, text, overlap=OVERLAP_WORDS):
    """Windows of CHUNK_WORDS words, stopping after the first one that reaches the document's last word."""
    words = text.split()
    chunks = []
    start = 0
    while True:
        chunks.append({"source": source, "text": " ".join(words[start:start + CHUNK_WORDS])})
        if start + CHUNK_WORDS >= len(words):
            return chunks
        start += CHUNK_WORDS - overlap


class PolicyIndex:
    def __init__(self, overlap=OVERLAP_WORDS):
        self.chunks = []
        for file_path in sorted(glob.glob(os.path.join(POLICIES, "*.txt"))):
            with open(file_path, encoding="utf-8") as f:
                self.chunks.extend(chunk_document(os.path.basename(file_path), f.read(), overlap))
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(c["text"] for c in self.chunks)


_default_index = None


def retrieve(query, k=3, index=None):
    """The k chunks most similar to the query, as dictionaries with source, score and text."""
    global _default_index
    if index is None:
        if _default_index is None:
            _default_index = PolicyIndex()
        index = _default_index
    scores = cosine_similarity(index.vectorizer.transform([query]), index.matrix)[0]
    ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
    return [{"source": index.chunks[i]["source"], "score": round(float(scores[i]), 4), "text": index.chunks[i]["text"]}
            for i in ranked]


def answer(question, index=None):
    """Retrieve three chunks and ask the model. Returns the model's text and the sources used."""
    hits = retrieve(question, k=3, index=index)
    context = "\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    response = chat([{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}])
    return {"answer": response.choices[0].message.content, "sources": list(dict.fromkeys(h["source"] for h in hits))}


def evaluate(index):
    """Runs policy_eval.csv. Returns one row per question."""
    rows = []
    for q in pd.read_csv(f"{RAW}/policy_eval.csv", keep_default_na=False).itertuples(index=False):
        result = answer(q.question, index=index)
        answerable = q.expected_doc != "NONE"
        rows.append({"question": q.question, "expected_doc": q.expected_doc, "expected_fact": q.expected_fact,
                     "doc_in_top_3": answerable and q.expected_doc in result["sources"],
                     "fact_in_answer": answerable and q.expected_fact in result["answer"],
                     "fallback": result["answer"].startswith("I could not find"), "answer": result["answer"]})
    return pd.DataFrame(rows)


def scores_of(table):
    answerable = table[table["expected_doc"] != "NONE"]
    uncovered = table[table["expected_doc"] == "NONE"]
    return (int(answerable["doc_in_top_3"].sum()), int(answerable["fact_in_answer"].sum()), int(uncovered["fallback"].sum()))


def main():
    index = PolicyIndex()

    section("Task 1: chunks (40 words, overlap 10)")
    counts = pd.Series([c["source"] for c in index.chunks]).value_counts().sort_index().to_dict()
    print(f"{len(index.chunks)} chunks: {counts}")
    assert counts == EXPECTED_CHUNKS and len(index.chunks) == 15

    section("Task 2: retrieve('refund policy late delivery')")
    hits = retrieve("refund policy late delivery", k=3)
    print(md_table(pd.DataFrame([{"rank": i + 1, "source": h["source"], "score": h["score"]} for i, h in enumerate(hits)]), floatfmt="{:.4f}"))
    assert all(h["source"] == "refund_policy.txt" for h in hits)

    section("Task 4: evaluation with overlap 10")
    with_overlap = evaluate(index)
    print(md_table(with_overlap[["question", "doc_in_top_3", "fact_in_answer", "fallback"]]))
    assert scores_of(with_overlap) == (12, 12, 2)

    section("Task 4: evaluation with overlap 0")
    no_overlap_index = PolicyIndex(overlap=0)
    no_overlap = evaluate(no_overlap_index)
    print(f"{len(no_overlap_index.chunks)} chunks")
    print(md_table(pd.DataFrame([{"overlap": "10 words", "doc_in_top_3": f"{scores_of(with_overlap)[0]}/12",
                                  "fact_in_answer": f"{scores_of(with_overlap)[1]}/12", "fallback": f"{scores_of(with_overlap)[2]}/2"},
                                 {"overlap": "0 words", "doc_in_top_3": f"{scores_of(no_overlap)[0]}/12",
                                  "fact_in_answer": f"{scores_of(no_overlap)[1]}/12", "fallback": f"{scores_of(no_overlap)[2]}/2"}])))
    broken = no_overlap[(no_overlap["expected_doc"] != "NONE") & ~no_overlap["fact_in_answer"]]
    assert len(broken) >= 1, "the brief expects at least one answerable question to fail with overlap 0"
    for row in broken.itertuples(index=False):
        whole = [c["source"] for c in no_overlap_index.chunks if row.expected_fact in c["text"]]
        print(f"\nBROKEN: {row.question}\n  expected fact: {row.expected_fact!r}; chunks holding the whole fact: {len(whole)}")
        print(f"  answer: {row.answer}")
        for chunk in no_overlap_index.chunks:
            if chunk["source"] == row.expected_doc:
                print(f"  chunk ends '...{chunk['text'][-60:]}'")

    section("Limitation check: questions outside the evaluation set")
    for question in ["Can I get a refund for my pet insurance?", "What is the delivery fee on the moon?", "Which cities do you deliver to?"]:
        print(f"{question} -> {answer(question)['answer'][:110]}")


if __name__ == "__main__":
    main()
