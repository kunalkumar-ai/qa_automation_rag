from openai import OpenAI
from config import OPENAI_API_KEY, GENERATION_MODEL, GENERATION_TEMPERATURE, MAX_CONTEXT_CHARS

client = OpenAI(api_key=OPENAI_API_KEY)

_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. "
    "Answer the question using ONLY the provided context from Tesla's 10-K filings. "
    "Each context block is labeled with its source year and section — use these labels "
    "when referencing specific numbers or facts so the answer is clear about which year "
    "a data point comes from. "
    "If the context does not contain enough information to answer, say so clearly. "
    "Be precise with numbers and facts."
)


def generate_answer(question: str, context_chunks: list[str], chunk_metas: list[dict] | None = None) -> str:
    """Send the question and retrieved context to GPT-4o and return a grounded answer.

    chunk_metas: optional list of dicts with 'year', 'company', 'section_name' for each chunk.
    When provided, each context block is labeled so GPT-4o knows which year it came from.
    """
    if not context_chunks:
        return "No relevant context found to answer this question."

    labeled_blocks = []
    for i, chunk in enumerate(context_chunks):
        truncated = chunk[:MAX_CONTEXT_CHARS]
        if chunk_metas and i < len(chunk_metas):
            meta = chunk_metas[i]
            company = meta.get("company", "tesla").capitalize()
            year = meta.get("year", "")
            section = meta.get("section_name", "")
            label = f"[Source: {company} {year} 10-K — {section}]" if year else f"[Source: {section}]"
        else:
            label = f"[Context {i + 1}]"
        labeled_blocks.append(f"{label}\n{truncated}")

    context = "\n\n---\n\n".join(labeled_blocks)
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
