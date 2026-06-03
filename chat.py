from collections import deque
from retriever import retrieve
from generator import generate_answer
from rewriter import rewrite_query
from logger import log_query

_MAX_HISTORY = 3


def chat():
    """Interactive conversational RAG loop with sliding window memory.

    Keeps the last 3 Q&A turns. Each new question is rewritten into a
    standalone question before retrieval so follow-ups resolve correctly.
    Type 'exit' or 'quit' to stop. Type 'clear' to reset conversation history.
    """
    history: deque[dict] = deque(maxlen=_MAX_HISTORY)

    print("Tesla 10-K Chat — ask anything about Tesla's 2022, 2023, or 2024 filings.")
    print("Commands: 'exit' to quit | 'clear' to reset conversation history\n")

    while True:
        try:
            raw_question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not raw_question:
            continue
        if raw_question.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        if raw_question.lower() == "clear":
            history.clear()
            print("[Conversation history cleared]\n")
            continue

        # Rewrite follow-up into standalone question
        history_list = list(history)
        print("Thinking...", flush=True)
        standalone = rewrite_query(raw_question, history_list)
        if standalone != raw_question:
            print(f"[Rewritten: {standalone}]", flush=True)

        # Full retrieval pipeline on the standalone question
        retrieval = retrieve(standalone)
        chunk_metas = [
            {
                "year": c["year"],
                "company": c["company"],
                "section_name": c["section_name"],
                "parent_id": c.get("parent_id", ""),
            }
            for c in retrieval["top_child_chunks"]
        ]
        answer = generate_answer(standalone, retrieval["parent_texts"], chunk_metas=chunk_metas)

        print(f"\nAssistant: {answer}\n")

        # Store the original question (not rewritten) in history so future
        # rewriting sees what the user actually said, not a system-generated version
        history.append({"question": raw_question, "answer": answer})

        log_query({
            "question": raw_question,
            "rewritten_question": standalone,
            "history_length": len(history_list),
            "history": [{"question": t["question"], "answer": t["answer"][:200]} for t in history_list],
            "route": retrieval.get("route", {}),
            "top_child_chunks": retrieval["top_child_chunks"],
            "parent_texts": retrieval["parent_texts"],
            "answer": answer,
            "ragas": {},
            "dense_candidates": len(retrieval["dense_ids"]),
            "bm25_candidates": len(retrieval["bm25_ids"]),
            "total_candidates": len(retrieval["all_candidates"]),
        })


if __name__ == "__main__":
    chat()
