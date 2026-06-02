import json
from openai import OpenAI
from config import OPENAI_API_KEY, AVAILABLE_COMPANIES, AVAILABLE_YEARS

client = OpenAI(api_key=OPENAI_API_KEY)

_SYSTEM_PROMPT = """You are a query router for a financial document intelligence system.
Your job is to decide which company filings and which years are needed to answer a question.

You must return valid JSON only — no explanation, no markdown.

Format:
{{
  "companies": ["tesla"],
  "years": ["2022", "2023", "2024"],
  "reasoning": "one sentence explaining the decision"
}}

Rules:
- Only include companies and years that are actually needed to answer the question
- If the question asks about a specific year, include only that year
- If the question asks to compare across years (e.g. "how did X change", "growth", "trend", "over the years"), include all relevant years
- If no specific year is mentioned and the question is about a single point in time, default to the most recent available year which is {most_recent_year}
- Available companies: {companies}
- Available years: {years}
"""


def route_query(question: str) -> dict:
    """Use GPT-4o-mini to decide which companies and years are relevant for this question.

    Returns a dict with keys: companies (list), years (list), reasoning (str)
    Falls back to all available data if the LLM response is unparseable.
    """
    prompt = _SYSTEM_PROMPT.format(
        companies=", ".join(AVAILABLE_COMPANIES),
        years=", ".join(AVAILABLE_YEARS),
        most_recent_year=max(AVAILABLE_YEARS),
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=200,
        )
        result = json.loads(response.choices[0].message.content)

        # Validate — only allow companies and years we actually have
        companies = [c for c in result.get("companies", []) if c in AVAILABLE_COMPANIES]
        years = [y for y in result.get("years", []) if y in AVAILABLE_YEARS]
        reasoning = result.get("reasoning", "")

        # Fallback to all data if router returned empty lists
        if not companies:
            companies = AVAILABLE_COMPANIES
        if not years:
            years = AVAILABLE_YEARS

        return {"companies": companies, "years": years, "reasoning": reasoning}

    except Exception:
        # Safe fallback — search everything rather than return nothing
        return {
            "companies": AVAILABLE_COMPANIES,
            "years": AVAILABLE_YEARS,
            "reasoning": "fallback: router failed, searching all available data",
        }
