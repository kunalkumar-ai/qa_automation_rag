# Tesla 10-K Financial RAG — Learning Project

## Why We Are Building This

This is a learning project, not a production tool. The goal is to understand RAG deeply — retrieval, reasoning, and evaluation — by building a system where **naive RAG visibly fails**, then fixing it. Each fix teaches one layer of the pipeline.

**The learning loop:**
```
Build → Measure with RAGAS → See where it breaks → Fix one thing → Measure again
```

## Why Tesla 10-K

The Tesla 10-K filings (2022, 2023, 2024) were chosen specifically because they break naive RAG:
- Long, dense documents (~130-260 pages each) — naive chunking loses context
- Tables and numbers — standard text embeddings struggle with structured data
- Questions require multi-paragraph synthesis — top-3 retrieval often returns wrong chunks
- Multi-year questions require cross-document reasoning — single-document RAG fails entirely
- Answers are verifiable against the real filings — we can measure correctness

## What We Have

| File/Folder | What it is |
|---|---|
| `tesla_doc/sec_tesla_2022.pdf` | Original Tesla 2022 10-K PDF |
| `tesla_doc/sec_tesla_2023.pdf` | Original Tesla 2023 10-K PDF |
| `tesla_doc/sec_tesla_2024.pdf` | Original Tesla 2024 10-K PDF |
| `tesla_doc/tsla-10k-2022.txt` | Clean extracted text — re-extracted with layout-aware pdfplumber |
| `tesla_doc/tsla-10k-2023.txt` | Clean extracted text |
| `tesla_doc/tsla-10k-2024.txt` | Clean extracted text |
| `tesla_doc/golden_dataset.md` | 20 verified Q&As — our evaluation baseline |

## Golden Dataset

20 questions with verified answers drawn directly from the filing. Split across:
- 5 simple (Business section) — baseline sanity check
- 7 medium (Risk Factors) — tests retrieval precision
- 8 hard (MD&A) — tests multi-paragraph reasoning

Every answer in the golden dataset includes the exact source line so it can be verified.

## Architecture

**Approach: Hierarchical + Hybrid RAG with Query Routing**

Two phases:

**Ingestion (runs once, covers all documents):**
```
For each document in DOCUMENTS registry (company + year + path):
  → Hierarchical Chunker (parent sections + child paragraphs)
      — each chunk tagged with company + year metadata
      — chunk IDs prefixed: tesla_2024_parent_0_1, tesla_2024_child_0_1_0
      — hard ceiling: child chunks truncated at 6000 chars (safety against malformed PDFs)
  → Embed child chunks (OpenAI text-embedding-3-large)
  → Store in ChromaDB (vectors + parent_id + company + year metadata)
  → Build unified BM25 keyword index (all child chunks across all documents)
```

**Query (runs every question):**
```
Question
  → Router (GPT-4o-mini) — decides which companies + years to search
  → Dense search (ChromaDB, top 30, filtered by company+year)    ┐
  → Sparse search (BM25, top 30, filtered by company+year)       ┘ run in parallel
  → Merge with Reciprocal Rank Fusion → ~55 candidates
  → Reranker (cross-encoder, dynamic top-k):
      1 year  → pick best 5
      2 years → pick best 7
      3 years → pick best 9
  → Parent lookup (swap child chunks for full parent sections)
  → GPT-4o (question + labeled parent sections → answer)
      — each context block labeled: [Source: Tesla 2022 10-K — Section Name]
  → RAGAS evaluation (retrieval score + faithfulness + correctness)
  → Two-level logging (console + JSON trace)
```

## Key Decisions

**Hierarchical chunking over flat word-count chunking**
Flat 500-word chunks split sentences mid-way and merge unrelated content. Hierarchical chunking respects section boundaries — parent chunks are full sections, child chunks are individual paragraphs. The hierarchy is preserved in metadata via `parent_id`.

**Hard child chunk ceiling (6000 chars)**
The 2022 PDF had sections extracting as single 40K-76K char blocks due to missing paragraph breaks. Rather than truncating silently, the chunker splits oversized paragraphs at sentence boundaries. Ensures no chunk exceeds the embedding model's token limit regardless of PDF extraction quality.

**Layout-aware PDF re-extraction for 2022**
The original 2022 extraction produced 390 children. Re-extracting with `pdfplumber layout=True` recovered paragraph breaks and produced 603 children — better granularity for retrieval.

**Document registry in config.py**
All documents defined in one place as `DOCUMENTS = [{"company": ..., "year": ..., "path": ...}]`. Adding a new company or year requires one line change — ingest, router, and retriever all read from this list automatically.

**LLM-based query routing (router.py)**
Before any retrieval, GPT-4o-mini reads the question and returns which companies and years are relevant as JSON. Handles ambiguous queries gracefully:
- "What were Tesla risks in 2024?" → years: [2024]
- "Compare R&D spend 2022 vs 2024" → years: [2022, 2024]
- "How did gross margin trend over 3 years?" → years: [2022, 2023, 2024]
Rule-based routing (regex) was rejected — too brittle for paraphrase and implicit year references.

**Hybrid search (dense + BM25) over dense-only**
Two distinct failure modes require two search strategies:
- Embeddings fail on exact terms: numbers (`$97.69B`), abbreviations (`NACS`, `FSD`), proper nouns
- BM25 fails on paraphrase: "why did income drop?" won't match "net income decreased"
Hybrid search covers both. Merging via Reciprocal Rank Fusion combines ranked lists without tuning score thresholds.

**Dynamic top-k reranking for multi-year queries**
Fixed top-5 was too small for 3-year queries — one irrelevant chunk could displace an entire year's context. Solution: `top_k = 5 + (num_years - 1) * 2`. Single-year queries stay at 5. Three-year queries get 9 slots. Year-balanced forced selection was rejected — it forces irrelevant chunks as years grow. A larger candidate pool (30+30=~55) combined with dynamic slots lets relevance scoring decide naturally.

**Labeled context blocks in generator**
Each parent section passed to GPT-4o is labeled `[Source: Tesla 2022 10-K — Section]`. Without labels, GPT-4o cannot attribute numbers to years in cross-document answers.

**Reranker after hybrid search**
Cross-encoder reads question + each candidate together, scoring true relevance. We retrieve ~55 candidates, rerank all, keep dynamic top-k. Using local `BAAI/bge-reranker-v2-m3` — inspectable scores are more valuable for learning than marginal accuracy from a cloud API.

**Retrieve small, pass large (parent lookup)**
Child chunks retrieved for precision. Parent sections passed to GPT-4o for context. Precision from child, reasoning context from parent. Deduplication by parent_id prevents sending the same section twice.

**Unit tests only for pure logic — not for RAG quality**
`test_chunker.py` tests hierarchy correctness, child-parent links, section boundaries.
`test_retriever.py` tests RRF merge algorithm.
We do NOT mock the reranker, ChromaDB, or GPT-4o — those mocks give false confidence. The golden dataset run is the real integration test.

**Two-level logging: console + JSON**
Every query produces:
- Console: question → top chunks with scores + year tags → answer
- JSON: full trace — dense/BM25/reranker scores, parent_ids, route decision, answer, RAGAS scores, timing

**No UI — terminal script only**
A UI would hide the pipeline internals. Learning comes from seeing every score and decision.

## Known Limitations

**Low-relevance chunks at high slot counts**
For 3-year queries (9 slots), the bottom 2-3 slots sometimes admit low-relevance chunks (reranker score < 0.3) when the document doesn't have 9 highly relevant sections. GPT-4o ignores them correctly but it wastes context tokens. A reranker score threshold would fix this — deferred.

**2023 coverage gap (fixed)**
Originally TOP_K_DENSE/BM25=20 with fixed top-5 caused 2023 chunks to be excluded from 3-year queries. Fixed by increasing pool to 30+30 and dynamic top-k.

**Query decomposition not implemented**
For maximum accuracy on multi-year queries, query decomposition (one sub-query per year, merged results) would be more reliable. Deferred — adds latency, API cost, and answer synthesis complexity. Documented as the right long-term direction.

## Files

| File | Responsibility |
|---|---|
| `config.py` | All constants + DOCUMENTS registry (company, year, path per filing) |
| `chunker.py` | Hierarchical chunking — company+year tags on every chunk, 6000-char safety ceiling |
| `ingest.py` | Loops all documents in registry: chunk → embed → ChromaDB → unified BM25 index |
| `router.py` | LLM-based query routing — GPT-4o-mini returns which companies+years to search |
| `retriever.py` | Hybrid search filtered by route → RRF → dynamic rerank → parent lookup |
| `generator.py` | GPT-4o with labeled context blocks (year + section per chunk) → answer |
| `evaluator.py` | RAGAS scoring — retrieval recall, faithfulness, answer correctness |
| `logger.py` | Two-level logging — console + JSON trace per query |
| `query.py` | Entry point: `python3 query.py "your question"` |
| `logs/` | JSON log files, one per query, named by timestamp |
| `tests/test_chunker.py` | Unit tests: hierarchy correct, child-parent links valid |
| `tests/test_retriever.py` | Unit tests: RRF ranking correct, handles zero results |

## How to Run

```bash
# One-time ingestion (all 3 documents)
python3 ingest.py

# Ask a single question
python3 query.py "What were Tesla's main risks in 2024?"

# Cross-year question
python3 query.py "How did Tesla gross margin change from 2022 to 2024?"

# With ground truth for RAGAS scoring
python3 query.py "question" "expected answer"
```
