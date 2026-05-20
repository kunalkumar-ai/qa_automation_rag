import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
CHROMA_PATH = "chroma_db"
DOCS_PATH = "docs"
TOP_K_RESULTS = 3
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

DEPARTMENT_DESCRIPTIONS = {
    "legal": "warranty terms, terms of service, privacy policy, legal agreements",
    "quality_control": "product quality, certifications, ISO standards, defect returns, testing",
    "supply_chain": "delivery timelines, shipping, distributors, order fulfilment",
    "environment_sustainability": "RoHS, WEEE, recycling, environmental compliance, sustainability",
    "it": "firmware, software, drivers, technical support, downloads",
    "energy_water": "energy ratings, power consumption, energy star, specifications",
    "after_sales": "repairs, spare parts, service centres, post-purchase support",
    "product_safety": "CE marking, safety certifications, recalls, product safety standards",
    "finance": "payment terms, invoices, pricing, export finance",
    "hr": "careers, jobs, internships, working at the company",
    "rd": "product roadmap, new products, innovation, research",
    "marketing": "product catalog, brand, promotions, partner guidelines",
    "logistics": "import, export, customs, international shipping",
    "customer_support": "escalation, complaints, SLA, response times, contact",
}

DEPARTMENTS = list(DEPARTMENT_DESCRIPTIONS.keys())
