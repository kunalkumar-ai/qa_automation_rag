# Tesla 10-K Financial RAG

A conversational RAG system for multi-year financial due diligence. Ask questions across Tesla's 2022, 2023, and 2024 10-K filings — including follow-up questions — and get cited, source-grounded answers.

> Built a conversational hierarchical RAG system for multi-year financial due diligence, combining hybrid search, cross-encoder reranking, and LLM-based query routing with audit-ready cited answers; validated retrieval quality using RAGAS.

---

## What It Does

- Answers questions about Tesla's financials, risks, and operations across 3 years of 10-K filings
- Supports follow-up questions ("which of those were new in 2024?") via query rewriting
- Cites the exact source section for every fact in the answer
- Warns when retrieval confidence is low
- Routes queries intelligently — single-year questions only search the relevant year

---

## Pipeline

**Ingestion (run once):**
```
PDF → pdfplumber extraction → hierarchical chunking (parent sections + child paragraphs)
  → OpenAI embeddings → ChromaDB + BM25 index
  — every chunk tagged with company + year
```

**Query:**
```
Question
  → Router (GPT-4o-mini) — which years are relevant?
  → Dense search (ChromaDB) + Sparse search (BM25) — top 30 each
  → Reciprocal Rank Fusion → top 20 candidates
  → Cross-encoder reranking (BAAI/bge-reranker-v2-m3, local)
  → Confidence check — ⚠️ warning if top score < 0.4
  → Parent section lookup
  → GPT-4o — answer with inline citations
```

**Conversational (chat.py):**
```
Follow-up question
  → Rewriter (GPT-4o-mini) — resolves references using last 3 turns
  → Same query pipeline above
```

---

## Key Architecture Decisions

| Decision | Why |
|---|---|
| Hierarchical chunking | Flat chunks split sentences mid-paragraph. Parent sections give GPT-4o full context; child chunks give retrieval precision. |
| Hybrid search (dense + BM25) | Embeddings fail on exact numbers and abbreviations. BM25 fails on paraphrase. Both together cover both failure modes. |
| LLM-based routing | Regex routing breaks on paraphrase and implicit year references. GPT-4o-mini handles all cases reliably. |
| Dynamic top-k | Fixed top-5 excludes years in multi-year queries. `top_k = 5 + (num_years - 1) * 2` scales slots with years. |
| Sliding window memory | Last 3 turns covers 99% of real follow-up patterns without growing prompt cost. |
| Model cost routing | GPT-4o-mini for routing + rewriting ($0.15/M tokens). GPT-4o only for final answer ($2.50/M tokens). |
| Inline citations | Every factual sentence cites its source section. Finance teams can verify any number without reading the full filing. |
| Confidence signal | Top reranker score below 0.4 triggers a warning. Answer still generated — signal is informational, not a hard stop. |

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Add your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# One-time ingestion (all 3 documents)
python3 ingest.py
```

---

## Usage

```bash
# Single question
python3 query.py "What were Tesla's main risks in 2024?"

# Cross-year question
python3 query.py "How did Tesla gross margin change from 2022 to 2024?"

# Conversational mode — supports follow-up questions
python3 chat.py
```

**Example conversation:**
```
You: What were Tesla's main risks in 2024?
Assistant: Tesla's main risks included supply chain disruptions *(Tesla 2024 10-K — Risk Factors)*...

You: Which of those were new compared to 2023?
[Rewritten: Which Tesla 2024 risk factors were not present in the 2023 10-K?]
Assistant: ...
```

---

## Files

| File | Responsibility |
|---|---|
| `config.py` | All constants and document registry |
| `chunker.py` | Hierarchical chunking with company/year tags |
| `ingest.py` | Ingestion pipeline — chunk, embed, store |
| `router.py` | LLM-based query routing by company and year |
| `retriever.py` | Hybrid search → RRF → rerank → parent lookup |
| `generator.py` | GPT-4o answer generation with cited context |
| `rewriter.py` | Query rewriting for conversational follow-ups |
| `chat.py` | Interactive conversational loop |
| `query.py` | Single-question entry point |
| `evaluator.py` | RAGAS evaluation — retrieval recall, faithfulness, correctness |
| `logger.py` | Per-query JSON logging |

---

## Evaluation

20-question golden dataset (`tesla_doc/golden_dataset.md`) covering simple, medium, and hard questions across all 3 years. Evaluated with RAGAS — retrieval recall, faithfulness, and answer correctness.

---

## Tech Stack

- LLM: GPT-4o (answer generation) + GPT-4o-mini (routing, rewriting)
- Embeddings: text-embedding-3-large (OpenAI)
- Vector database: ChromaDB (local)
- Keyword search: BM25 (rank-bm25)
- Reranker: BAAI/bge-reranker-v2-m3 (local, sentence-transformers)
- Evaluation: RAGAS
- Language: Python 3.10+
