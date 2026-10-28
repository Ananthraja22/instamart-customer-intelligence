"""Chunked TF-IDF policy retrieval and deterministic evaluation."""

from pathlib import Path
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_DIR = PROJECT_ROOT / "data" / "policies"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHUNK_WORDS = 80
CHUNK_OVERLAP_WORDS = 16


def load_policy_chunks() -> list[dict]:
    """Split each paragraph into fixed-size word chunks with overlap."""
    chunks = []
    for path in sorted(POLICY_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        chunk_number = 0
        for paragraph in paragraphs:
            words = paragraph.split()
            step = CHUNK_WORDS - CHUNK_OVERLAP_WORDS
            for start in range(0, max(len(words), 1), step):
                window = words[start : start + CHUNK_WORDS]
                if not window:
                    continue
                chunks.append(
                    {
                        "document": path.name,
                        "chunk_id": f"{path.stem}-{chunk_number:03d}",
                        "text": " ".join(window),
                        "word_start": start,
                        "word_end": start + len(window),
                    }
                )
                chunk_number += 1
                if start + CHUNK_WORDS >= len(words):
                    break
    if not chunks:
        raise FileNotFoundError(f"No policy documents found in {POLICY_DIR}")
    return chunks


def retrieve_policy_chunks(
    question: str, chunks: list[dict] | None = None, top_k: int = 3
) -> list[dict]:
    chunks = chunks or load_policy_chunks()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])
    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, matrix).ravel()
    ranked_indices = scores.argsort()[::-1][: max(1, min(top_k, len(chunks)))]
    return [
        {**chunks[index], "score": float(scores[index])}
        for index in ranked_indices
        if scores[index] > 0
    ]


def answer_question(question: str, chunks: list[dict]) -> dict:
    retrieved = retrieve_policy_chunks(question, chunks=chunks, top_k=1)
    if not retrieved:
        return {
            "document": "",
            "answer": "I could not find a relevant fact in the supplied policies.",
            "score": 0.0,
            "chunk_id": "",
        }
    best = retrieved[0]
    return {
        "document": best["document"],
        "answer": best["text"],
        "score": round(best["score"], 3),
        "chunk_id": best["chunk_id"],
    }


def main() -> None:
    eval_path = CLEAN_DIR / "policy_eval.csv"
    if not eval_path.exists():
        raise FileNotFoundError("Run clean_data.py before policy evaluation.")

    chunks = load_policy_chunks()
    evaluation = pd.read_csv(eval_path).fillna("")
    rows = []

    for row in evaluation.to_dict(orient="records"):
        result = answer_question(str(row["question"]), chunks)
        expected_doc = str(row["expected_doc"])
        expected_fact = str(row["expected_fact"]).strip().lower()
        rows.append(
            {
                "question": row["question"],
                "predicted_doc": result["document"],
                "expected_doc": expected_doc,
                "document_match": result["document"] == expected_doc,
                "answer_chunk": result["answer"],
                "expected_fact": row["expected_fact"],
                "fact_found": expected_fact in result["answer"].lower(),
                "chunk_id": result["chunk_id"],
                "retrieval_score": result["score"],
            }
        )

    result_frame = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "policy_eval_results.csv"
    result_frame.to_csv(output_path, index=False)

    print(f"Policy chunks: {len(chunks)} (80 words, 16-word overlap)")
    print(result_frame.to_string(index=False))
    print(
        f"\nDocument matches: "
        f"{result_frame['document_match'].sum()}/{len(result_frame)}"
    )
    print(
        "Expected facts found in retrieved chunks: "
        f"{result_frame['fact_found'].sum()}/{len(result_frame)}"
    )
    print(f"Saved evaluation to {output_path}")


if __name__ == "__main__":
    main()
