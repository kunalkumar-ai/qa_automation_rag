# RAG Customer Q&A System

A Retrieval-Augmented Generation (RAG) tool for ElectroTech, an electronics manufacturer. Staff paste a customer email question into a Gradio web UI and receive a GPT-4o-generated suggested answer drawn from 14 department document collections. The answer is copied into the email reply — no email integration.

**Status:** Fully implemented and tested. 14 tests passing, 28 documents ingested into ChromaDB.

## How to Run

**First time only — generate sample documents:**
```bash
python3 generate_docs.py
```

**First time only — ingest documents into ChromaDB:**
```bash
python3 ingest.py
```
Re-run this whenever documents in `docs/` are updated.

**Run the web app:**
```bash
python3 app.py
```
Opens at `http://127.0.0.1:7860`

**Run tests:**
```bash
python3 -m pytest -v
```
Expected: 14 tests passing across 4 test files.

## Architecture

Metadata-Filtered RAG (Option B):

```
Customer question (pasted into UI)
    → Department Classifier (GPT-4o, temperature=0) → e.g. "legal"
    → ChromaDB retrieval filtered by department metadata → top 3 chunks
    → Answer Generator (GPT-4o, temperature=0.3) → professional reply
    → Staff copies answer into email
```

If the classifier cannot identify the department, no answer is generated — the UI tells staff to handle the email manually.

## Key Decisions

- **Raw prose documents, not Q&A pairs** — GPT-4o synthesises answers at query time, so any customer question is answerable without pre-writing Q&A pairs.
- **One ChromaDB collection** — all chunks stored together, filtered by `department` metadata at query time. Simpler than separate collections per department.
- **Idempotent ingestion** — `ingest.py` deletes and recreates the collection on every run, so re-ingesting after document updates is always clean.
- **Unknown department = no answer** — if GPT-4o cannot confidently classify the question, the UI instructs staff to handle it manually rather than risk a wrong answer.
- **Chunk size: 500 words, overlap: 50 words** — overlap prevents losing context at chunk boundaries.
- **temperature=0 for classifier** — deterministic department classification, no randomness.
- **temperature=0.3 for generator** — slightly creative for natural-sounding answers while staying factual.
- **DEPARTMENTS derived from DEPARTMENT_DESCRIPTIONS keys** — single source of truth in `config.py`, no duplication.

## File Responsibilities

| File | Responsibility |
|---|---|
| `config.py` | Single source of truth for all constants (models, paths, departments, descriptions) |
| `generate_docs.py` | One-time script to generate 28 sample department documents via GPT-4o |
| `ingest.py` | One-time script: chunk → embed → store in ChromaDB with department metadata |
| `classifier.py` | GPT-4o call to identify which department owns the customer question |
| `retriever.py` | ChromaDB query filtered by department metadata, returns top-3 chunks |
| `generator.py` | GPT-4o call to produce a professional answer from question + chunks |
| `app.py` | Gradio UI wiring classifier → retriever → generator, with pipeline logging |

## Tests

| Test File | What it covers |
|---|---|
| `tests/test_ingest.py` | `chunk_text` — short docs, long docs, overlap, empty string |
| `tests/test_classifier.py` | `classify_department` — known dept, unknown, invalid GPT response, whitespace |
| `tests/test_retriever.py` | `retrieve_chunks` — matching chunks, no results, department filter passed correctly |
| `tests/test_generator.py` | `generate_answer` — with chunks, no chunks (no API call), all chunks in context |

All external API calls (OpenAI, ChromaDB) are mocked in tests — no real API calls made during testing.

## Documents

28 sample documents across 14 departments (2 per department). Stored in `docs/<department>/`.

To use real company documents: replace the `.txt` files in `docs/` and re-run `python3 ingest.py`.

Departments: `legal`, `quality_control`, `supply_chain`, `environment_sustainability`, `it`, `energy_water`, `after_sales`, `product_safety`, `finance`, `hr`, `rd`, `marketing`, `logistics`, `customer_support`

## Future Improvements (not in scope)

- Swap ChromaDB for Pinecone when moving to production
- Add confidence score display in the UI
- Gmail API integration to auto-fetch questions and draft replies
- Document upload via UI instead of filesystem management
- Multi-language support

## Design Docs

- Spec: `docs/superpowers/specs/2026-05-19-rag-design.md`
- Implementation plan: `docs/superpowers/plans/2026-05-19-rag-system.md`
