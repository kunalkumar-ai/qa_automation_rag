import json
import os
from datetime import datetime
from config import LOGS_DIR


def log_query(data: dict) -> str:
    """Log a query result to console and a JSON file.

    Returns the path to the JSON log file.
    """
    question = data.get("question", "")
    answer = data.get("answer", "")
    chunks = data.get("top_child_chunks", [])
    ragas = data.get("ragas", {})

    print("\n" + "=" * 60)
    print(f"QUESTION: {question}")
    print("=" * 60)

    print(f"\nTOP {len(chunks)} RETRIEVED CHUNKS:")
    for i, chunk in enumerate(chunks, 1):
        print(
            f"\n  [{i}] dense={chunk.get('dense_score', 0):.4f} | "
            f"bm25={chunk.get('bm25_score', 0):.4f} | "
            f"reranker={chunk.get('reranker_score', 0):.4f}"
        )
        print(f"      section: {chunk.get('section_name', 'unknown')}")
        print(f"      text: {chunk.get('text', '')[:120]}...")

    print(f"\nANSWER:\n{answer}")

    if ragas:
        print("\nRAGAS SCORES:")
        print(f"  faithfulness:       {ragas.get('faithfulness', 'n/a')}")
        print(f"  answer_correctness: {ragas.get('answer_correctness', 'n/a')}")
        print(f"  context_recall:     {ragas.get('context_recall', 'n/a')}")

    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGS_DIR, f"{timestamp}.json")
    with open(log_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\nFull trace → {log_path}")
    print("=" * 60 + "\n")
    return log_path
