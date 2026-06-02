import os
from dotenv import load_dotenv

load_dotenv()

# API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Models
EMBEDDING_MODEL = "text-embedding-3-large"
GENERATION_MODEL = "gpt-4o"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Paths
CHROMA_PATH = "chroma_db"
BM25_INDEX_PATH = "bm25_index.pkl"
PARENTS_PATH = "parents.pkl"
GOLDEN_DATASET_PATH = "tesla_doc/golden_dataset.md"
LOGS_DIR = "logs"

# Document registry — each entry is one 10-K filing
# Add new companies or years here; ingest.py and router.py read from this list
DOCUMENTS = [
    {"company": "tesla", "year": "2022", "path": "tesla_doc/tsla-10k-2022.txt"},
    {"company": "tesla", "year": "2023", "path": "tesla_doc/tsla-10k-2023.txt"},
    {"company": "tesla", "year": "2024", "path": "tesla_doc/tsla-10k-2024.txt"},
]

AVAILABLE_COMPANIES = list({d["company"] for d in DOCUMENTS})
AVAILABLE_YEARS = sorted({d["year"] for d in DOCUMENTS})

# Retrieval parameters
TOP_K_DENSE = 30
TOP_K_BM25 = 30
TOP_K_RERANK = 5       # base value — retriever scales this up for multi-year queries
RRF_K = 60
GENERATION_TEMPERATURE = 0.3
MAX_CONTEXT_CHARS = 3000  # max chars per parent section sent to GPT-4o
