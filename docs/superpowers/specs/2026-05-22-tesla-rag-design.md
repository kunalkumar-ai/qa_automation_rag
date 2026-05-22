# Tesla 10-K Financial RAG — System Design Spec

**Date:** 2026-05-22
**Status:** Approved for implementation

---

## 1. Purpose

This system is a **learning project**, not a production tool. The goal is to understand RAG deeply — retrieval, reasoning, and evaluation — by building a system where naive RAG visibly fails, then fixing it layer by layer. Each component is chosen because it teaches something specific about how RAG works.

**The learning loop:**
```
Build → Measure with RAGAS → See where it breaks → Fix one thing → Measure again
```

---

## 2. Why Tesla 10-K

The Tesla 2024 annual report (10-K) was chosen because it breaks naive RAG in specific, diagnosable ways:
- Long, structured document (142 pages, 473K characters) — naive word-count chunking cuts sentences mid-way and merges different sections into the same chunk
- Exact terms that embeddings miss — numbers (`$97.69B`), abbreviations (`NACS`, `FSD`, `GWh`), proper nouns (`Gigafactory Berlin-Brandenburg`, `Cybercab`)
- Multi-paragraph reasoning required — hard questions need context from multiple paragraphs
- Verifiable answers — every answer can be checked against the real filing

---

## 3. Evaluation Baseline

**Golden dataset:** `tesla_doc/golden_dataset.md` — 20 questions with verified answers.

| Section | Questions | Difficulty | What it tests |
|---|---|---|---|
| Business (Item 1) | 5 | Simple | Baseline sanity check |
| Risk Factors (Item 1A) | 7 | Medium | Retrieval precision |
| MD&A (Item 7) | 8 | Hard | Multi-paragraph reasoning |

RAGAS scores against this dataset are the primary measure of system quality.

---

## 4. Architecture

**Approach: Hierarchical + Hybrid RAG**

### Ingestion Pipeline (runs once)
```
tsla-10k-2024.txt
  → chunker.py: parent chunks (full sections) + child chunks (paragraphs ~100-150 words)
      each child stores: parent_id + section_name
  → ingest.py:
      → embed child chunks → ChromaDB
      → store parent chunks → parents.pkl (dict: parent_id → text)
      → build BM25 index on child chunks → bm25_index.pkl
```

### Query Pipeline (every question)
```
Question
  → Dense search: ChromaDB top-20 child chunks by cosine similarity  ┐ parallel
  → BM25 search: keyword match top-20 child chunks                   ┘
  → RRF merge → up to 40 candidates
  → Cross-encoder reranker → top-5
  → Parent lookup → swap child chunks for full parent sections
  → GPT-4o: question + 5 parent sections → answer
  → RAGAS: retrieval recall + faithfulness + answer correctness
  → Logger: console summary + logs/<timestamp>.json full trace
```

---

## 5. Key Design Decisions

- **Hierarchical chunking**: parent=full section (no word cap), child=paragraph by double newline (~100-150 words). Respects document structure, not word count.
- **Hybrid search**: embeddings miss exact terms; BM25 misses paraphrase. Together they cover both failure modes. Merged via Reciprocal Rank Fusion.
- **Local cross-encoder reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`. Free, inspectable — similarity measures vector shape, reranker measures answer relevance.
- **Retrieve small, pass large**: child chunks retrieved for precision, parent sections passed to LLM for context.
- **No UI**: terminal script exposes all scores and internals. Essential for learning.
- **Two-level logging**: console (human-readable) + JSON file (full trace for cross-run analysis).
- **Unit tests only for pure logic**: chunker hierarchy + RRF merge. Golden dataset run is the integration test.

---

## 6. File Structure

```
rag/
├── config.py
├── chunker.py
├── ingest.py
├── retriever.py
├── generator.py
├── evaluator.py
├── logger.py
├── query.py
├── logs/
├── tesla_doc/
│   ├── tsla-10k-2024.txt
│   └── golden_dataset.md
└── tests/
    ├── test_chunker.py
    └── test_retriever.py
```

---

## 7. Dependencies

| Library | Purpose |
|---|---|
| `openai` | Embeddings + GPT-4o |
| `chromadb` | Vector storage + dense search |
| `rank-bm25` | BM25 sparse keyword index |
| `sentence-transformers` | Local cross-encoder reranker |
| `ragas>=0.1.0,<0.2.0` | Evaluation metrics |
| `datasets` | Required by RAGAS |
| `pytest` | Unit tests |
