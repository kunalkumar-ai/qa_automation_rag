from openai import OpenAI
from config import OPENAI_API_KEY, CHAT_MODEL, DEPARTMENTS, DEPARTMENT_DESCRIPTIONS

client = OpenAI(api_key=OPENAI_API_KEY)


def classify_department(question: str) -> str:
    """Returns a department name from DEPARTMENTS, or 'unknown' if not confident."""
    dept_list = "\n".join(
        f"- {dept}: {desc}" for dept, desc in DEPARTMENT_DESCRIPTIONS.items()
    )
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a department classifier for ElectroTech, an electronics manufacturer. "
                    "Given a customer question, identify which single department it belongs to. "
                    "Reply with ONLY the department name from the list below, exactly as written. "
                    "If the question does not clearly belong to any one department, reply with 'unknown'.\n\n"
                    f"Departments:\n{dept_list}"
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    result = response.choices[0].message.content.strip().lower()
    return result if result in DEPARTMENTS else "unknown"
