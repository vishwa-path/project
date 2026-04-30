import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path("novatech_dataset")

QUERIES_PATH = DATA_DIR / "queries.json"
DOCS_PATH = DATA_DIR / "poisoned_docs.json"
OUT_PATH = DATA_DIR / "majority_vote_results.json"


def normalize_answer(x):
    if x is None:
        return ""
    return str(x).strip().lower().replace(".", "")


def simple_retrieve(question, docs, top_k=5):
    q_words = set(question.lower().split())

    scored = []
    for doc in docs:
        text = (doc.get("title", "") + " " + doc.get("text", "")).lower()
        score = sum(1 for w in q_words if w in text)
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored[:top_k]]


def extract_candidate_answer(doc):
    for key in ["answer", "value", "gold_answer"]:
        if key in doc:
            return str(doc[key])

    text = doc.get("text", "")
    if ":" in text:
        return text.split(":")[-1].strip().split(".")[0]

    return text.strip().split(".")[0]


def majority_vote_answer(retrieved_docs):
    candidates = [normalize_answer(extract_candidate_answer(doc)) for doc in retrieved_docs]
    candidates = [c for c in candidates if c]

    if not candidates:
        return ""

    counts = Counter(candidates)
    answer, count = counts.most_common(1)[0]

    if count == 1 and len(set(candidates)) > 1:
        return "insufficient reliable information"

    return answer


def main():
    with open(QUERIES_PATH, "r") as f:
        queries = json.load(f)

    with open(DOCS_PATH, "r") as f:
        docs = json.load(f)

    results = []

    for query in queries:
        question = query.get("question") or query.get("query")
        gold_answer = query.get("answer") or query.get("gold_answer")

        retrieved_docs = simple_retrieve(question, docs, top_k=5)
        model_answer = majority_vote_answer(retrieved_docs)

        result = {
            "query_id": query.get("query_id"),
            "question": question,
            "gold_answer": gold_answer,
            "model_answer": model_answer,
            "correct": normalize_answer(model_answer) == normalize_answer(gold_answer),
            "retrieved_docs": retrieved_docs,
            "poison_retrieved": any(doc.get("is_poison", False) for doc in retrieved_docs),
            "poison_marker_seen": "flag-poison" in normalize_answer(model_answer),
            "bad_answer_seen": normalize_answer(model_answer) != normalize_answer(gold_answer),
            "num_retrieved": len(retrieved_docs)
        }

        results.append(result)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved majority vote results to {OUT_PATH}")


if __name__ == "__main__":
    main()