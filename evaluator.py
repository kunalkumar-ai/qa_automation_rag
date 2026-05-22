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

    all_scores: dict[str, list[float]] = {
        "faithfulness": [], "answer_correctness": [], "context_recall": []
    }

    for i, pair in enumerate(pairs, 1):
        print(f"[{i:02d}/{len(pairs)}] {pair['question'][:65]}...")
        retrieval = retrieve(pair["question"])
        answer = generate_answer(pair["question"], retrieval["parent_texts"])
        scores = evaluate_single(
            question=pair["question"],
            answer=answer,
            contexts=retrieval["parent_texts"],
            ground_truth=pair["ground_truth"],
        )
        for metric, score in scores.items():
            all_scores[metric].append(score)
        print(
            f"         faithfulness={scores['faithfulness']:.3f} | "
            f"correctness={scores['answer_correctness']:.3f} | "
            f"recall={scores['context_recall']:.3f}"
        )

    print("\n" + "=" * 55)
    print("AGGREGATE RAGAS SCORES (mean across all questions):")
    for metric, values in all_scores.items():
        print(f"  {metric:<22}: {sum(values) / len(values):.3f}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    run_full_evaluation()
