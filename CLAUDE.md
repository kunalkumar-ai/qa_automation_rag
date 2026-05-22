# Tesla 10-K Financial RAG — Learning Project

## Why We Are Building This

This is a learning project, not a production tool. The goal is to understand RAG deeply — retrieval, reasoning, and evaluation — by building a system where **naive RAG visibly fails**, then fixing it. Each fix teaches one layer of the pipeline.

**The learning loop:**
```
Build → Measure with RAGAS → See where it breaks → Fix one thing → Measure again
```

## Why Tesla 10-K

The Tesla 2024 annual report (10-K) was chosen specifically because it breaks naive RAG:
- Long, dense document (142 pages, 473K characters) — naive chunking loses context
- Tables and numbers — standard text embeddings struggle with structured data
- Questions require multi-paragraph synthesis — top-3 retrieval often returns wrong chunks
- Answers are verifiable against the real filing — we can measure correctness

## What We Have So Far

| File/Folder | What it is |
|---|---|
| `tesla_doc/sec_tesla_2024.pdf` | Original Tesla 2024 10-K PDF from Tesla's website |
| `tesla_doc/tsla-10k-2024.txt` | Clean extracted text (142 pages, 473K chars) |
| `tesla_doc/golden_dataset.md` | 20 verified Q&As — our evaluation baseline |

## Golden Dataset

20 questions with verified answers drawn directly from the filing. Split across:
- 5 simple (Business section) — baseline sanity check
- 7 medium (Risk Factors) — tests retrieval precision
- 8 hard (MD&A) — tests multi-paragraph reasoning

Every answer in the golden dataset includes the exact source line so it can be verified.

## Architecture

**Approach: Hierarchical + Hybrid RAG**

Two phases:

**Ingestion (runs once):**
```
Tesla 10-K text
  → Hierarchical Chunker (parent sections + child paragraphs)
  → Embed child chunks (OpenAI text-embedding-3-small)
  → Store in ChromaDB (vectors + parent_id metadata)
  → Build BM25 keyword index (same child chunks, saved to disk)
```

**Query (runs every question):**
```
Question
  → Dense search (ChromaDB, top 20 by vector similarity)     ┐
  → Sparse search (BM25, top 20 by keyword match)            ┘ run in parallel
  → Merge with Reciprocal Rank Fusion → 40 candidates
  → Reranker (cross-encoder, picks best 5)
  → Parent lookup (swap child chunks for full parent sections)
  → GPT-4o (question + 5 parent sections → answer)
  → RAGAS evaluation (retrieval score + faithfulness + correctness)
```

## Key Decisions

**Hierarchical chunking over flat word-count chunking**
Current system chunks by 500 words regardless of document structure. For a 10-K, this splits sentences mid-way and merges different risk factors or financial line items into the same chunk. Hierarchical chunking respects section boundaries — parent chunks are full sections, child chunks are individual paragraphs.

**Hybrid search (dense + BM25) over dense-only**
Two distinct failure modes require two search strategies:
- Embeddings fail on exact terms: numbers (`$97.69B`), abbreviations (`NACS`, `FSD`), proper nouns (`Gigafactory Berlin-Brandenburg`). Embedding models don't reliably place these near related queries.
- BM25 fails on paraphrase: asking "why did income drop?" won't match a chunk saying "net income decreased" — BM25 has no concept of meaning.
Hybrid search covers both failure modes. Merging via Reciprocal Rank Fusion combines the ranked lists without needing to tune score thresholds.

**Reranker after hybrid search**
Search (dense or sparse) finds candidates by similarity. A cross-encoder reranker reads the question and each candidate chunk *together* and scores true relevance — much more accurate than similarity alone. We retrieve 40, rerank, keep top 5.

**Retrieve small, pass large (parent lookup)**
Child chunks (paragraphs) are retrieved for precision — small chunks are more likely to be an exact semantic match. But the LLM needs context to reason correctly. So after retrieval, we swap each child chunk for its parent section. Precision from child, context from parent.

**Local cross-encoder reranker over Cohere API**
Using `sentence-transformers` cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) rather than a paid API. Runs locally, free, and inspectable — since this is a learning project, seeing the reranker's exact scores for each candidate chunk is more valuable than marginal accuracy gains from a cloud API.

**Unit tests only for pure logic — not for RAG quality**
`test_chunker.py` tests hierarchy correctness (child-parent links, section boundary respect) — pure deterministic logic, no API calls. `test_retriever.py` tests RRF merge algorithm — does it rank correctly, does it handle one source returning zero results? We do NOT mock the reranker, ChromaDB, or GPT-4o — those mocks would give false confidence. The golden dataset run is the real integration test.

**Two-level logging: console + JSON**
Every query produces two logs:
- Console (human-readable): question → top-5 chunks with scores → answer → RAGAS scores. Immediate feedback while running.
- JSON file (machine-readable): full trace of every step — dense scores, BM25 scores, reranker scores for all 40 candidates, parent chunks used, full prompt, answer, RAGAS scores, timing. Stored in `logs/` by timestamp. Lets you query across runs later — e.g. "which questions had lowest retrieval scores?"

**No UI — terminal script only**
`query.py` runs from the command line. Prints answer + retrieval scores + reranker scores + RAGAS metrics alongside the answer. A UI would hide the internals that matter most for learning. Focus stays on the RAG pipeline, not the interface.

**RAGAS evaluation loop**
Without measurement, we can't know if a change improved anything. RAGAS scores three things independently: did retrieval find the right chunk, did the LLM stay faithful to the retrieved context, and is the final answer correct? The 20-question golden dataset is the ground truth for all scoring.

## Files

| File | Responsibility |
|---|---|
| `config.py` | All constants — model names, paths, chunk sizes, top-k values |
| `chunker.py` | Hierarchical chunking — splits 10-K into parent sections + child paragraphs, stores parent_id on each child |
| `ingest.py` | One-time script: chunk → embed child chunks → store in ChromaDB → build + save BM25 index |
| `retriever.py` | Hybrid search: dense (ChromaDB) + sparse (BM25) → merge via RRF → rerank → parent lookup |
| `generator.py` | GPT-4o call with question + retrieved parent chunks → answer |
| `evaluator.py` | RAGAS scoring — retrieval recall, faithfulness, answer correctness against golden dataset |
| `logger.py` | Two-level logging — console (human-readable) + JSON file (full trace per query) |
| `query.py` | Entry point — wires all components, run with: `python3 query.py "your question"` |
| `logs/` | JSON log files, one per query run, named by timestamp |
| `tests/test_chunker.py` | Unit tests for chunker logic — hierarchy correct, child-parent links valid, section boundaries respected |
| `tests/test_retriever.py` | Unit tests for RRF merge logic — correct ranking, handles zero results from one source |

## How to Run

_To be documented here once the system is built._
