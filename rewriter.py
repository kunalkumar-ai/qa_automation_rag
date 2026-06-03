import json
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

_SYSTEM_PROMPT = (
    "You are a query rewriter for a financial document RAG system. "
    "You will receive a conversation history and a new follow-up question. "
    "Your job is to rewrite the follow-up question into a complete, standalone question "
    "that can be understood and answered without reading the conversation history. "
    "Rules:\n"
    "- Resolve all pronouns and references (those, it, they, that, the same) to their actual subjects.\n"
    "- If the question already stands alone with no references to prior context, return it unchanged.\n"
    "- Return ONLY the rewritten question — no explanation, no preamble, no quotes.\n"
    "- Never add information that was not in the conversation."
)


def rewrite_query(question: str, history: list[dict]) -> str:
    """Rewrite a follow-up question into a standalone question using conversation history.

    history: list of {"question": str, "answer": str} dicts, most recent last.
    Returns the rewritten question string (or original if no history).
    """
    if not history:
        return question

    history_text = ""
    for i, turn in enumerate(history, 1):
        history_text += f"Turn {i}:\n"
        history_text += f"  Q: {turn['question']}\n"
        history_text += f"  A: {turn['answer'][:400]}\n\n"

    user_message = (
        f"Conversation history:\n{history_text}"
        f"Follow-up question: {question}\n\n"
        f"Rewrite the follow-up question as a standalone question:"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else question
    except Exception:
        return question
