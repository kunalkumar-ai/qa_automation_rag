from openai import OpenAI
from config import OPENAI_API_KEY, GENERATION_MODEL, GENERATION_TEMPERATURE, MAX_CONTEXT_CHARS

client = OpenAI(api_key=OPENAI_API_KEY)

_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. "
    "Answer the question using ONLY the provided context from Tesla's 2024 10-K filing. "
    "If the context does not contain enough information to answer, say so clearly. "
    "Be precise with numbers and facts."
)


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """Send the question and retrieved context to GPT-4o and return a grounded answer."""
    if not context_chunks:
        return "No relevant context found to answer this question."

    truncated = [chunk[:MAX_CONTEXT_CHARS] for chunk in context_chunks]
    context = "\n\n---\n\n".join(truncated)
    user_message = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=GENERATION_TEMPERATURE,
    )
    return response.choices[0].message.content.strip()
