import sys
from retriever import retrieve
from generator import generate_answer
from evaluator import evaluate_single
from logger import log_query
from config import CONFIDENCE_THRESHOLD


def _check_confidence(top_chunks: list[dict]) -> float | None:
    """Return the top reranker score, or None if no chunks."""
    if not top_chunks:
        return None
    return max(c["reranker_score"] for c in top_chunks)


def query(question: str, ground_truth: str | None = None) -> str:
    """Run one question through the full pipeline and return the answer.

    If ground_truth is provided, RAGAS scores are computed and logged.
    """
    retrieval = retrieve(question)
    chunk_metas = [
        {"year": c["year"], "company": c["company"], "section_name": c["section_name"], "parent_id": c.get("parent_id", "")}
        for c in retrieval["top_child_chunks"]
    ]
    top_score = _check_confidence(retrieval["top_child_chunks"])
    if top_score is not None and top_score < CONFIDENCE_THRESHOLD:
        print(f"⚠️  Low confidence — best reranker score: {top_score:.2f}. Answer may not be accurate.\n")

    answer = generate_answer(question, retrieval["parent_texts"], chunk_metas=chunk_metas)

    ragas_scores: dict = {}
    if ground_truth:
        ragas_scores = evaluate_single(
            question=question,
            answer=answer,
            contexts=retrieval["parent_texts"],
            ground_truth=ground_truth,
        )

    log_query({
        "question": question,
        "route": retrieval.get("route", {}),
        "top_child_chunks": retrieval["top_child_chunks"],
        "parent_texts": retrieval["parent_texts"],
        "answer": answer,
        "ragas": ragas_scores,
        "dense_candidates":  len(retrieval["dense_ids"]),
        "bm25_candidates":   len(retrieval["bm25_ids"]),
        "total_candidates":  len(retrieval["all_candidates"]),
    })

    return answer


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 query.py 'your question here'")
        print("       python3 query.py 'question' 'expected answer'")
        print("       python3 query.py --all")
        sys.exit(1)

    if sys.argv[1] == "--all":
        from evaluator import parse_golden_dataset
        pairs = parse_golden_dataset()
        print(f"\nRunning all {len(pairs)} questions (no RAGAS)...\n")
        for i, pair in enumerate(pairs, 1):
            print(f"[{i:02d}/{len(pairs)}] {pair['question']}")
            query(pair["question"])
    else:
        q  = sys.argv[1]
        gt = sys.argv[2] if len(sys.argv) > 2 else None
        query(q, gt)
