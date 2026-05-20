from openai import OpenAI
from config import OPENAI_API_KEY, CHAT_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

NO_INFO_RESPONSE = (
    "I don't have enough information to answer this question. "
    "Please contact our support team directly."
)


def generate_answer(question: str, chunks: list[str]) -> str:
    """Generates a professional customer support answer from retrieved chunks."""
    if not chunks:
        return NO_INFO_RESPONSE

    context = "\n\n---\n\n".join(chunks)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional customer support agent for ElectroTech, an electronics manufacturer. "
                    "Answer the customer's question using only the information provided in the context below. "
                    "Be clear, concise, and professional. Do not invent information not found in the context. "
                    "If the context does not contain enough information to fully answer, say so politely.\n\n"
                    f"Context:\n{context}"
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()
