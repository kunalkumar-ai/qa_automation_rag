import re
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness, context_recall
from config import GOLDEN_DATASET_PATH
from retriever import retrieve
from generator import generate_answer


def parse_golden_dataset(path: str = GOLDEN_DATASET_PATH) -> list[dict]:
    """Parse golden_dataset.md into a list of {question, ground_truth} dicts."""
    with open(path) as f:
        content = f.read()

    qa_pairs = []
    blocks = re.split(r'---+', content)
    for block in blocks:
        q_match = re.search(r'\*\*Q\d+\.\*\*\s+(.+?)(?=\n\*\*A:\*\*)', block, re.DOTALL)
        a_match = re.search(r'\*\*A:\*\*\s+(.+?)(?=\n\*\*Source:|$)', block, re.DOTALL)
        if q_match and a_match:
            qa_pairs.append({
                "question": q_match.group(1).strip(),
                "ground_truth": a_match.group(1).strip(),
            })
    return qa_pairs


def evaluate_single(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> dict:
    """Score one Q&A with RAGAS. Returns faithfulness, answer_correctness, context_recall."""
    dataset = Dataset.from_dict({
        "question":     [question],
        "answer":       [answer],
        "contexts":     [contexts],
        "ground_truth": [ground_truth],
    })
    result = evaluate(dataset, metrics=[faithfulness, answer_correctness, context_recall])
    return {
        "faithfulness":        float(result["faithfulness"]),
        "answer_correctness":  float(result["answer_correctness"]),
        "context_recall":      float(result["context_recall"]),
    }


def run_full_evaluation() -> None:
    """Run all 20 golden dataset questions and print per-question and aggregate RAGAS scores."""
    pairs = parse_golden_dataset()
    print(f"\nEvaluating {len(pairs)} questions...\n")

    # Phase 1: retrieve + generate for all questions (no RAGAS yet)
    results = []
    for i, pair in enumerate(pairs, 1):
        print(f"[{i:02d}/{len(pairs)}] retrieving + generating: {pair['question'][:55]}...")
        retrieval = retrieve(pair["question"])
        answer = generate_answer(pair["question"], retrieval["parent_texts"])
        results.append({
            "question":     pair["question"],
            "answer":       answer,
            "contexts":     retrieval["parent_texts"],
            "ground_truth": pair["ground_truth"],
        })

    # Phase 2: single RAGAS call over all results
    print(f"\nScoring {len(results)} answers with RAGAS (one batch)...\n")
    dataset = Dataset.from_dict({
        "question":     [r["question"]     for r in results],
        "answer":       [r["answer"]       for r in results],
        "contexts":     [r["contexts"]     for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })
    ragas_result = evaluate(dataset, metrics=[faithfulness, answer_correctness, context_recall])

    scores_by_metric = {
        "faithfulness":       list(ragas_result.scores["faithfulness"]),
        "answer_correctness": list(ragas_result.scores["answer_correctness"]),
        "context_recall":     list(ragas_result.scores["context_recall"]),
    }

    print("Per-question scores:")
    for i, r in enumerate(results):
        f  = scores_by_metric["faithfulness"][i]
        ac = scores_by_metric["answer_correctness"][i]
        cr = scores_by_metric["context_recall"][i]
        print(f"  [{i+1:02d}] faithfulness={f:.3f} | correctness={ac:.3f} | recall={cr:.3f}")
        print(f"       Q: {r['question'][:65]}")

    print("\n" + "=" * 55)
    print("AGGREGATE RAGAS SCORES (mean across all questions):")
    for metric, values in scores_by_metric.items():
        print(f"  {metric:<22}: {sum(values) / len(values):.3f}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    run_full_evaluation()
