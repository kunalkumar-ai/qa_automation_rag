import os
from dotenv import load_dotenv

load_dotenv()

# API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Models
EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4o"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Paths
CHROMA_PATH = "chroma_db"
BM25_INDEX_PATH = "bm25_index.pkl"
PARENTS_PATH = "parents.pkl"
TESLA_DOC_PATH = "tesla_doc/tsla-10k-2024.txt"
GOLDEN_DATASET_PATH = "tesla_doc/golden_dataset.md"
LOGS_DIR = "logs"

# Retrieval parameters
TOP_K_DENSE = 20
TOP_K_BM25 = 20
TOP_K_RERANK = 5
RRF_K = 60
GENERATION_TEMPERATURE = 0.3
MAX_CONTEXT_CHARS = 3000  # max chars per parent section sent to GPT-4o
