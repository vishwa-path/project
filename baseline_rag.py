import json
import re
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline


DATA_DIR = Path("novatech_dataset")
TOP_K = 3


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def simple_match(pred: str, gold: str) -> bool:
    pred_n = normalize_text(pred)
    gold_n = normalize_text(gold)
    return gold_n in pred_n or pred_n in gold_n


class RAGPipeline:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        generator_model_name: str = "google/flan-t5-base",
    ):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.generator = pipeline(
            "text2text-generation",
            model=generator_model_name,
            max_new_tokens=128,
        )

        self.index = None
        self.chunks: List[Dict[str, Any]] = []
        self.chunk_embeddings = None

    def load_chunks(self, chunks_path: Path):
        self.chunks = load_json(chunks_path)
        print(f"Loaded {len(self.chunks)} chunks.")

    def build_faiss_index(self):
        texts = [chunk["text"] for chunk in self.chunks]
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        self.chunk_embeddings = embeddings.astype("float32")

        dim = self.chunk_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.chunk_embeddings)

        print(f"Built FAISS index with {self.index.ntotal} vectors.")

    def retrieve(self, query: str, top_k: int = TOP_K):
        query_emb = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(query_emb, top_k)
        results = []

        for score, idx in zip(scores[0], indices[0]):
            chunk = self.chunks[idx]
            results.append({
                "score": float(score),
                "chunk_id": chunk["chunk_id"],
                "parent_id": chunk["parent_id"],
                "title": chunk["title"],
                "text": chunk["text"],
            })
        return results
    
    def build_prompt(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            context_parts.append(
                f"[Document {i}]\nTitle: {chunk['title']}\nContent: {chunk['text']}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are answering questions using only the provided documents.

Instructions:
- Use the documents below to answer the question.
- If the answer is present, answer briefly and directly.
- Do not make up facts.
- If the answer is not in the documents, say "Not found in retrieved documents."

Documents:
{context}

Question: {query}

Answer:
""".strip()

        return prompt

    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        prompt = self.build_prompt(query, retrieved_chunks)
        output = self.generator(prompt)[0]["generated_text"].strip()
        return output

    def answer_query(self, query: str, top_k: int = TOP_K):
        retrieved = self.retrieve(query, top_k=top_k)
        answer = self.generate_answer(query, retrieved)
        return {
            "query": query,
            "retrieved_chunks": retrieved,
            "answer": answer,
        }


def evaluate_queries(
    rag: RAGPipeline,
    queries_path: Path,
    output_path: Path,
    top_k: int = TOP_K,
):
    queries = load_json(queries_path)
    results = []

    correct = 0
    retrieval_hit = 0

    for item in queries:
        query = item["query"]
        gold_answer = item["answer"]
        source_doc_id = item["source_doc_id"]

        result = rag.answer_query(query, top_k=top_k)

        retrieved_parent_ids = [c["parent_id"] for c in result["retrieved_chunks"]]
        hit = source_doc_id in retrieved_parent_ids
        ans_correct = simple_match(result["answer"], gold_answer)

        if hit:
            retrieval_hit += 1
        if ans_correct:
            correct += 1

        results.append({
            "query_id": item["query_id"],
            "query": query,
            "gold_answer": gold_answer,
            "source_doc_id": source_doc_id,
            "retrieved_parent_ids": retrieved_parent_ids,
            "retrieval_hit": hit,
            "predicted_answer": result["answer"],
            "answer_correct": ans_correct,
            "retrieved_chunks": result["retrieved_chunks"],
        })

        print("=" * 80)
        print(f"Query: {query}")
        print(f"Gold: {gold_answer}")
        print(f"Pred: {result['answer']}")
        print(f"Retrieval hit: {hit} | Answer correct: {ans_correct}")

    summary = {
        "num_queries": len(queries),
        "retrieval_hit_rate": retrieval_hit / len(queries) if queries else 0.0,
        "answer_accuracy": correct / len(queries) if queries else 0.0,
    }

    final_output = {
        "summary": summary,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print("\nSaved results to:", output_path)
    print("Summary:")
    print(json.dumps(summary, indent=2))


def main():
    rag = RAGPipeline()
    rag.load_chunks(DATA_DIR / "chunks.json")
    rag.build_faiss_index()

    evaluate_queries(
        rag=rag,
        queries_path=DATA_DIR / "queries.json",
        output_path=DATA_DIR / "baseline_results.json",
        top_k=TOP_K,
    )


if __name__ == "__main__":
    main()