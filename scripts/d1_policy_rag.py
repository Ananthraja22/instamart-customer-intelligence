"""Part D: policy question answering. TF-IDF retrieval over sentence-window chunks of the five policy
files, a grounded answer through mock_llm.chat, and a refusal when nothing relevant is retrieved.

The refund agent (d2_refund_agent.py) imports PolicyIndex, so both share one retrieval path.
"""
import glob
import os
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import common  # noqa: F401  (puts the repo root on sys.path for mock_llm)
from common import POLICIES, RAW, md_table, section
from mock_llm import chat

SENTENCES_PER_CHUNK = 2
SENTENCE_OVERLAP = 1
TOP_K = 3
MIN_SCORE = 0.08
REFUSAL = "I could not find that information in the policy documents."
SYSTEM_PROMPT = ("You are the POLICY ASSISTANT for a grocery delivery app. Answer the question using only the "
                 "CONTEXT supplied. If the CONTEXT does not contain the answer, say that you could not find it.")


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.])\s+", text.strip()) if s.strip()]


def chunk_document(source, text):
    """Windows of whole sentences, so a fact and its condition are never cut apart mid-sentence."""
    sentences = split_sentences(text)
    step = SENTENCES_PER_CHUNK - SENTENCE_OVERLAP
    chunks = []
    for start in range(0, max(len(sentences) - SENTENCE_OVERLAP, 1), step):
        chunks.append({"source": source, "chunk_id": f"{source}#{len(chunks)}",
                       "text": " ".join(sentences[start:start + SENTENCES_PER_CHUNK])})
    return chunks


class PolicyIndex:
    def __init__(self):
        self.chunks = []
        for file_path in sorted(glob.glob(os.path.join(POLICIES, "*.txt"))):
            with open(file_path, encoding="utf-8") as f:
                self.chunks.extend(chunk_document(os.path.basename(file_path), f.read()))
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(c["text"] for c in self.chunks)

    def search(self, query, k=TOP_K):
        """Top-k chunks scoring at least MIN_SCORE. An empty list means the policies do not cover the query."""
        scores = linear_kernel(self.vectorizer.transform([query]), self.matrix)[0]
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [{**self.chunks[i], "score": round(float(scores[i]), 4)} for i in ranked if scores[i] >= MIN_SCORE]


def answer_question(index, question):
    hits = index.search(question)
    if not hits:  # Out of scope: the model is never asked to improvise.
        return {"answer": REFUSAL, "sources": [], "refused": True, "tokens": 0}
    context = "\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    response = chat([{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION: {question}"}])
    text = response.choices[0].message.content
    grounded = any(text in h["text"] for h in hits)
    cited = sorted({h["source"] for h in hits if text in h["text"]})
    return {"answer": text if grounded else REFUSAL, "sources": cited, "refused": not grounded,
            "tokens": response.usage.total_tokens}


def main():
    index = PolicyIndex()
    section("Policy index")
    print(f"{len(index.chunks)} chunks from 5 documents; top_k={TOP_K}; min_score={MIN_SCORE}")

    questions = pd.read_csv(f"{RAW}/policy_eval.csv", keep_default_na=False)
    rows = []
    for q in questions.itertuples(index=False):
        hits = index.search(q.question)
        result = answer_question(index, q.question)
        if q.expected_doc == "NONE":
            retrieved_ok, fact_ok = (not hits), result["refused"]
        else:
            retrieved_ok = bool(hits) and hits[0]["source"] == q.expected_doc
            fact_ok = q.expected_fact.lower() in result["answer"].lower()
        rows.append({"question": q.question, "expected_doc": q.expected_doc,
                     "top_doc": hits[0]["source"] if hits else "-", "top_score": hits[0]["score"] if hits else 0.0,
                     "doc_ok": retrieved_ok, "answer_ok": fact_ok, "tokens": result["tokens"]})
    table = pd.DataFrame(rows)

    section("Evaluation on data/raw/policy_eval.csv")
    print(md_table(table, floatfmt="{:.3f}"))
    answerable = table[table["expected_doc"] != "NONE"]
    refused = table[table["expected_doc"] == "NONE"]
    print(f"\nanswerable: top-1 document correct {int(answerable['doc_ok'].sum())}/{len(answerable)}; "
          f"expected fact in answer {int(answerable['answer_ok'].sum())}/{len(answerable)}")
    print(f"out of scope: correctly refused {int(refused['answer_ok'].sum())}/{len(refused)}")
    lowest = answerable["top_score"].min()
    print(f"lowest top score on an answerable question: {lowest:.3f}; "
          f"highest on an out-of-scope question: {refused['top_score'].max():.3f}")
    print(f"total tokens: {int(table['tokens'].sum())}")


if __name__ == "__main__":
    main()
