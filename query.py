import sys
from retriever import retrieve
from generator import generate_answer
from evaluator import evaluate_single
from logger import log_query


def query(question: str, ground_truth: str | None = None) -> str:
    """Run one question through the full pipeline and return the answer.

    If ground_truth is provided, RAGAS scores are computed and logged.
    """
    retrieval = retrieve(question)
    answer = generate_answer(question, retrieval["parent_texts"])

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
        sys.exit(1)

    q  = sys.argv[1]
    gt = sys.argv[2] if len(sys.argv) > 2 else None
    query(q, gt)
