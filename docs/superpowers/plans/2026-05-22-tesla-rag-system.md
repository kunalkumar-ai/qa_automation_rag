# Tesla 10-K Financial RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Hierarchical + Hybrid RAG system over Tesla's 2024 10-K that teaches retrieval, reasoning, and evaluation by making each component's contribution measurable.

**Architecture:** Child paragraphs are retrieved via hybrid search (ChromaDB dense + BM25 sparse), merged with RRF, reranked by a local cross-encoder, then swapped for full parent sections before passing to GPT-4o. RAGAS evaluates every answer against a 20-question golden dataset.

**Tech Stack:** Python, OpenAI (text-embedding-3-small + gpt-4o), ChromaDB, rank-bm25, sentence-transformers (cross-encoder/ms-marco-MiniLM-L-6-v2), RAGAS 0.1.x, pytest

---

## Task 1: Clean Up Old Files and Install Dependencies

**Files:**
- Delete: `classifier.py`, `app.py`, `generate_docs.py`
- Delete: `tests/test_classifier.py`, `tests/test_generator.py`, `tests/test_ingest.py`, `tests/test_retriever.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Remove old source files no longer needed**

```bash
rm classifier.py app.py generate_docs.py
rm tests/test_classifier.py tests/test_generator.py tests/test_ingest.py tests/test_retriever.py
```

Expected: no output, files gone.

- [ ] **Step 2: Update requirements.txt**

Replace the full contents of `requirements.txt` with:

```
openai
chromadb
rank-bm25
sentence-transformers
ragas>=0.1.0,<0.2.0
datasets
pdfplumber
pytest
python-dotenv
```

- [ ] **Step 3: Install new dependencies**

```bash
pip install rank-bm25 sentence-transformers "ragas>=0.1.0,<0.2.0" datasets -q
```

Expected output ends with: `Successfully installed ...`

- [ ] **Step 4: Verify key imports work**

```bash
python3 -c "from rank_bm25 import BM25Okapi; from sentence_transformers import CrossEncoder; from ragas import evaluate; print('all ok')"
```

Expected: `all ok`

- [ ] **Step 5: Delete old ChromaDB data to start fresh**

```bash
rm -rf chroma_db
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: remove old system files, update dependencies for hybrid RAG"
```

---

## Task 2: config.py — All Constants

**Files:**
- Modify: `config.py`

- [ ] **Step 1: Replace config.py with new constants**

```python
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
RRF_K = 60          # standard RRF constant — higher = less penalty for lower ranks
GENERATION_TEMPERATURE = 0.3
```

- [ ] **Step 2: Verify it imports cleanly**

```bash
python3 -c "import config; print(config.EMBEDDING_MODEL)"
```

Expected: `text-embedding-3-small`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "feat: add config.py for hybrid RAG system constants"
```

---

## Task 3: logger.py — Console + JSON Logging

**Files:**
- Modify: `logger.py`

- [ ] **Step 1: Replace logger.py with two-level logger**

```python
import json
import os
from datetime import datetime
from config import LOGS_DIR


def log_query(data: dict) -> str:
    """Log query to console (human-readable) and JSON file (full trace).

    Returns the path to the JSON log file.
    """
    q = data.get("question", "")
    answer = data.get("answer", "")
    chunks = data.get("top_child_chunks", [])
    ragas = data.get("ragas", {})

    print("\n" + "=" * 60)
    print(f"QUESTION: {q}")
    print("=" * 60)

    print(f"\nTOP {len(chunks)} RETRIEVED CHUNKS:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n  [{i}] dense={chunk.get('dense_score', 0):.4f} | "
              f"bm25={chunk.get('bm25_score', 0):.4f} | "
              f"reranker={chunk.get('reranker_score', 0):.4f}")
        print(f"      section: {chunk.get('section_name', 'unknown')}")
        print(f"      text: {chunk.get('text', '')[:120]}...")

    print(f"\nANSWER:\n{answer}")

    if ragas:
        print(f"\nRAGAS SCORES:")
        print(f"  faithfulness:      {ragas.get('faithfulness', 'n/a')}")
        print(f"  answer_correctness:{ragas.get('answer_correctness', 'n/a')}")
        print(f"  context_recall:    {ragas.get('context_recall', 'n/a')}")

    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGS_DIR, f"{timestamp}.json")
    with open(log_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\nFull trace → {log_path}")
    print("=" * 60 + "\n")
    return log_path
```

- [ ] **Step 2: Verify it imports and runs with dummy data**

```bash
python3 -c "
from logger import log_query
log_query({
    'question': 'test question',
    'top_child_chunks': [{'dense_score': 0.9, 'bm25_score': 5.2, 'reranker_score': 0.8, 'section_name': 'ITEM 1', 'text': 'sample text here'}],
    'answer': 'test answer',
    'ragas': {'faithfulness': 0.9, 'answer_correctness': 0.85, 'context_recall': 0.7}
})
"
```

Expected: formatted console output + `logs/<timestamp>.json` created.

- [ ] **Step 3: Commit**

```bash
git add logger.py logs/
git commit -m "feat: add two-level logger (console + JSON trace)"
```

---

## Task 4: chunker.py — Hierarchical Chunking (TDD)

**Files:**
- Create: `chunker.py`
- Create: `tests/test_chunker.py`

- [ ] **Step 1: Write the failing tests first**

Create `tests/test_chunker.py`:

```python
import pytest
from chunker import build_chunks, parse_sections

SAMPLE_TEXT = """ITEM 1. BUSINESS
Overview
We design and manufacture electric vehicles and energy storage systems.

We sell them directly to customers through our website and stores.

ITEM 1A. RISK FACTORS
We may experience delays in launching products.

We face significant competition from other manufacturers."""


def test_parse_sections_finds_two_items():
    sections = parse_sections(SAMPLE_TEXT)
    assert len(sections) == 2


def test_parse_sections_names_start_with_item():
    sections = parse_sections(SAMPLE_TEXT)
    assert sections[0]["name"].startswith("ITEM 1.")
    assert sections[1]["name"].startswith("ITEM 1A.")


def test_each_section_has_text():
    sections = parse_sections(SAMPLE_TEXT)
    for section in sections:
        assert len(section["text"].strip()) > 0


def test_build_chunks_produces_parents_and_children():
    chunks = build_chunks(SAMPLE_TEXT)
    types = {c.chunk_type for c in chunks}
    assert "parent" in types
    assert "child" in types


def test_child_chunk_parent_id_matches_a_parent():
    chunks = build_chunks(SAMPLE_TEXT)
    parent_ids = {c.chunk_id for c in chunks if c.chunk_type == "parent"}
    children = [c for c in chunks if c.chunk_type == "child"]
    for child in children:
        assert child.parent_id in parent_ids, f"child {child.chunk_id} has orphan parent_id {child.parent_id}"


def test_no_empty_chunks():
    chunks = build_chunks(SAMPLE_TEXT)
    for chunk in chunks:
        assert chunk.text.strip() != "", f"chunk {chunk.chunk_id} is empty"


def test_section_name_propagated_to_children():
    chunks = build_chunks(SAMPLE_TEXT)
    children = [c for c in chunks if c.chunk_type == "child"]
    for child in children:
        assert child.section_name != "", f"child {child.chunk_id} has no section_name"


def test_very_short_paragraphs_are_skipped():
    text = """ITEM 2. PROPERTIES
We own the following properties.

Note

These properties support manufacturing operations across all regions."""
    chunks = build_chunks(text)
    children = [c for c in chunks if c.chunk_type == "child"]
    child_texts = [c.text for c in children]
    assert not any(t.strip() == "Note" for t in child_texts)


def test_chunk_ids_are_unique():
    chunks = build_chunks(SAMPLE_TEXT)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "duplicate chunk IDs found"
```

- [ ] **Step 2: Run tests and confirm they all fail**

```bash
python3 -m pytest tests/test_chunker.py -v
```

Expected: all 9 tests FAIL with `ModuleNotFoundError: No module named 'chunker'`

- [ ] **Step 3: Implement chunker.py**

Create `chunker.py`:

```python
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    text: str
    parent_id: str
    section_name: str
    chunk_type: str  # "parent" or "child"


def parse_sections(text: str) -> list[dict]:
    """Split document into sections based on ITEM headings."""
    pattern = re.compile(r'(ITEM\s+\d+[A-C]?\.\s+[A-Z][A-Z\s,&\(\)]+)', re.MULTILINE)
    matches = list(pattern.finditer(text))
    sections = []
    for i, match in enumerate(matches):
        name = match.group(1).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        sections.append({"name": name, "text": section_text})
    return sections


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs by double newline, skip short ones."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if len(p.strip().split()) >= 10]


def build_chunks(text: str) -> list[Chunk]:
    """Build hierarchical chunks: one parent per section, children are paragraphs."""
    sections = parse_sections(text)
    all_chunks: list[Chunk] = []

    for i, section in enumerate(sections):
        parent_id = f"parent_{i}"

        parent = Chunk(
            chunk_id=parent_id,
            text=section["text"],
            parent_id=parent_id,
            section_name=section["name"],
            chunk_type="parent",
        )
        all_chunks.append(parent)

        paragraphs = _split_paragraphs(section["text"])
        for j, para in enumerate(paragraphs):
            child = Chunk(
                chunk_id=f"child_{i}_{j}",
                text=para,
                parent_id=parent_id,
                section_name=section["name"],
                chunk_type="child",
            )
            all_chunks.append(child)

    return all_chunks
```

- [ ] **Step 4: Run tests and confirm all pass**

```bash
python3 -m pytest tests/test_chunker.py -v
```

Expected: 9 tests PASS.

- [ ] **Step 5: Smoke test on real document**

```bash
python3 -c "
from chunker import build_chunks
with open('tesla_doc/tsla-10k-2024.txt') as f:
    text = f.read()
chunks = build_chunks(text)
parents = [c for c in chunks if c.chunk_type == 'parent']
children = [c for c in chunks if c.chunk_type == 'child']
print(f'Sections (parents): {len(parents)}')
print(f'Paragraphs (children): {len(children)}')
print(f'First section: {parents[0].section_name}')
print(f'Sample child: {children[0].text[:100]}')
"
```

Expected: several parent sections (10–20), hundreds of child chunks, no errors.

- [ ] **Step 6: Commit**

```bash
git add chunker.py tests/test_chunker.py
git commit -m "feat: hierarchical chunker with parent sections and child paragraphs (TDD)"
```

---

## Task 5: ingest.py — Build ChromaDB + BM25 Index

**Files:**
- Modify: `ingest.py`

- [ ] **Step 1: Replace ingest.py**

```python
import os
import pickle
import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi
from chunker import build_chunks
from config import (
    OPENAI_API_KEY, EMBEDDING_MODEL, CHROMA_PATH,
    BM25_INDEX_PATH, PARENTS_PATH, TESLA_DOC_PATH,
)

client = OpenAI(api_key=OPENAI_API_KEY)


def _embed_batch(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def ingest(doc_path: str = TESLA_DOC_PATH) -> None:
    print(f"Reading {doc_path}...")
    with open(doc_path) as f:
        text = f.read()

    all_chunks = build_chunks(text)
    children = [c for c in all_chunks if c.chunk_type == "child"]
    parents = {c.chunk_id: c.text for c in all_chunks if c.chunk_type == "parent"}

    print(f"Built {len(parents)} parent sections, {len(children)} child paragraphs")

    # ── ChromaDB: store child chunks with embeddings ──────────────────────
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        chroma_client.delete_collection("tesla")
    except Exception:
        pass
    collection = chroma_client.create_collection("tesla")

    batch_size = 100
    for i in range(0, len(children), batch_size):
        batch = children[i : i + batch_size]
        embeddings = _embed_batch([c.text for c in batch])
        collection.add(
            ids=[c.chunk_id for c in batch],
            embeddings=embeddings,
            documents=[c.text for c in batch],
            metadatas=[
                {"parent_id": c.parent_id, "section_name": c.section_name}
                for c in batch
            ],
        )
        print(f"  Embedded {min(i + batch_size, len(children))}/{len(children)} child chunks")

    # ── Parent store: save as pickle for parent lookup ────────────────────
    with open(PARENTS_PATH, "wb") as f:
        pickle.dump(parents, f)
    print(f"Parent store saved to {PARENTS_PATH}")

    # ── BM25 index on child chunks ────────────────────────────────────────
    tokenized = [c.text.lower().split() for c in children]
    bm25 = BM25Okapi(tokenized)
    bm25_data = {
        "bm25": bm25,
        "chunk_ids": [c.chunk_id for c in children],
        "chunk_texts": {c.chunk_id: c.text for c in children},
        "chunk_metas": {c.chunk_id: {"parent_id": c.parent_id, "section_name": c.section_name} for c in children},
    }
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_data, f)
    print(f"BM25 index saved to {BM25_INDEX_PATH}")
    print("Ingestion complete.")


if __name__ == "__main__":
    ingest()
```

- [ ] **Step 2: Run ingestion**

```bash
python3 ingest.py
```

Expected output (will take 2-5 minutes due to embedding API calls):
```
Reading tesla_doc/tsla-10k-2024.txt...
Built N parent sections, M child paragraphs
  Embedded 100/M child chunks
  ...
Parent store saved to parents.pkl
BM25 index saved to bm25_index.pkl
Ingestion complete.
```

- [ ] **Step 3: Verify ChromaDB and index files exist**

```bash
python3 -c "
import chromadb, pickle
c = chromadb.PersistentClient('chroma_db')
col = c.get_collection('tesla')
print(f'ChromaDB child chunks: {col.count()}')
with open('bm25_index.pkl','rb') as f:
    d = pickle.load(f)
print(f'BM25 index chunks: {len(d[\"chunk_ids\"])}')
with open('parents.pkl','rb') as f:
    p = pickle.load(f)
print(f'Parent sections: {len(p)}')
"
```

Expected: ChromaDB count matches BM25 chunk count, parents > 0.

- [ ] **Step 4: Commit**

```bash
git add ingest.py
git commit -m "feat: ingest Tesla 10-K into ChromaDB and BM25 index"
```

---

## Task 6: retriever.py — Hybrid Search + RRF + Reranker + Parent Lookup (TDD)

**Files:**
- Modify: `retriever.py`
- Create: `tests/test_retriever.py`

- [ ] **Step 1: Write failing tests for RRF merge (pure logic only)**

Create `tests/test_retriever.py`:

```python
import pytest
from retriever import rrf_merge


def test_rrf_item_in_both_lists_ranks_first():
    """A chunk appearing in both dense and BM25 results should rank highest."""
    dense = ["a", "b", "c"]
    bm25 = ["b", "d", "e"]
    result = rrf_merge(dense, bm25)
    assert result[0] == "b"


def test_rrf_includes_all_unique_ids():
    dense = ["a", "b"]
    bm25 = ["c", "d"]
    result = rrf_merge(dense, bm25)
    assert set(result) == {"a", "b", "c", "d"}


def test_rrf_higher_rank_beats_lower_rank():
    """First item in dense list should score higher than last item."""
    dense = ["first", "second", "third"]
    bm25 = []
    result = rrf_merge(dense, bm25)
    assert result[0] == "first"
    assert result[-1] == "third"


def test_rrf_handles_empty_bm25():
    dense = ["a", "b", "c"]
    bm25 = []
    result = rrf_merge(dense, bm25)
    assert result == ["a", "b", "c"]


def test_rrf_handles_empty_dense():
    dense = []
    bm25 = ["x", "y", "z"]
    result = rrf_merge(dense, bm25)
    assert result == ["x", "y", "z"]


def test_rrf_handles_both_empty():
    result = rrf_merge([], [])
    assert result == []


def test_rrf_no_duplicates_in_output():
    dense = ["a", "b", "c"]
    bm25 = ["a", "b", "d"]
    result = rrf_merge(dense, bm25)
    assert len(result) == len(set(result))
```

- [ ] **Step 2: Run tests and confirm they fail**

```bash
python3 -m pytest tests/test_retriever.py -v
```

Expected: all 7 tests FAIL with `ImportError`.

- [ ] **Step 3: Implement retriever.py**

```python
import pickle
import chromadb
from openai import OpenAI
from sentence_transformers import CrossEncoder
from config import (
    OPENAI_API_KEY, EMBEDDING_MODEL, CHROMA_PATH,
    BM25_INDEX_PATH, PARENTS_PATH, RERANKER_MODEL,
    TOP_K_DENSE, TOP_K_BM25, TOP_K_RERANK, RRF_K,
)

client = OpenAI(api_key=OPENAI_API_KEY)
reranker = CrossEncoder(RERANKER_MODEL)


def rrf_merge(dense_ids: list[str], bm25_ids: list[str], k: int = RRF_K) -> list[str]:
    """Reciprocal Rank Fusion — merge two ranked lists by position score."""
    scores: dict[str, float] = {}
    for rank, chunk_id in enumerate(dense_ids):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, chunk_id in enumerate(bm25_ids):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


def retrieve(question: str) -> dict:
    """Full retrieval: hybrid search → RRF → rerank → parent lookup."""

    # ── Dense search ──────────────────────────────────────────────────────
    q_vec = client.embeddings.create(model=EMBEDDING_MODEL, input=[question]).data[0].embedding
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection("tesla")

    dense_result = collection.query(
        query_embeddings=[q_vec],
        n_results=TOP_K_DENSE,
        include=["documents", "metadatas", "distances"],
    )
    dense_ids: list[str] = dense_result["ids"][0]
    dense_score_map = {
        cid: float(dist)
        for cid, dist in zip(dense_ids, dense_result["distances"][0])
    }
    dense_text_map = {
        cid: doc
        for cid, doc in zip(dense_ids, dense_result["documents"][0])
    }
    dense_meta_map = {
        cid: meta
        for cid, meta in zip(dense_ids, dense_result["metadatas"][0])
    }

    # ── BM25 search ───────────────────────────────────────────────────────
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)

    bm25_scores = bm25_data["bm25"].get_scores(question.lower().split())
    ranked = sorted(
        zip(bm25_data["chunk_ids"], bm25_scores),
        key=lambda x: x[1],
        reverse=True,
    )
    bm25_ids = [cid for cid, _ in ranked[:TOP_K_BM25]]
    bm25_score_map = {cid: float(score) for cid, score in ranked[:TOP_K_BM25]}

    # Fetch BM25 candidates not already in dense results
    missing_ids = [cid for cid in bm25_ids if cid not in dense_text_map]
    if missing_ids:
        fetched = collection.get(ids=missing_ids, include=["documents", "metadatas"])
        for cid, doc, meta in zip(fetched["ids"], fetched["documents"], fetched["metadatas"]):
            dense_text_map[cid] = doc
            dense_meta_map[cid] = meta

    # ── RRF merge ─────────────────────────────────────────────────────────
    merged_ids = rrf_merge(dense_ids, bm25_ids)

    # ── Rerank ────────────────────────────────────────────────────────────
    valid_ids = [cid for cid in merged_ids if cid in dense_text_map]
    pairs = [[question, dense_text_map[cid]] for cid in valid_ids]
    rerank_scores = reranker.predict(pairs)
    scored = sorted(zip(valid_ids, rerank_scores), key=lambda x: x[1], reverse=True)
    top_ids = [cid for cid, _ in scored[:TOP_K_RERANK]]
    rerank_score_map = {cid: float(score) for cid, score in scored}

    # ── Parent lookup ─────────────────────────────────────────────────────
    with open(PARENTS_PATH, "rb") as f:
        parents: dict[str, str] = pickle.load(f)

    parent_texts: list[str] = []
    seen_parents: set[str] = set()
    for cid in top_ids:
        pid = dense_meta_map[cid]["parent_id"]
        if pid not in seen_parents and pid in parents:
            parent_texts.append(parents[pid])
            seen_parents.add(pid)

    return {
        "top_child_chunks": [
            {
                "chunk_id": cid,
                "text": dense_text_map.get(cid, ""),
                "section_name": dense_meta_map.get(cid, {}).get("section_name", ""),
                "dense_score": dense_score_map.get(cid, 0.0),
                "bm25_score": bm25_score_map.get(cid, 0.0),
                "reranker_score": rerank_score_map.get(cid, 0.0),
            }
            for cid in top_ids
        ],
        "parent_texts": parent_texts,
        "all_candidates": merged_ids,
        "dense_ids": dense_ids,
        "bm25_ids": bm25_ids,
    }
```

- [ ] **Step 4: Run tests — all 7 should pass**

```bash
python3 -m pytest tests/test_retriever.py -v
```

Expected: 7 tests PASS. (These test only `rrf_merge` — pure logic, no API calls.)

- [ ] **Step 5: Smoke test retrieve on real question**

```bash
python3 -c "
from retriever import retrieve
result = retrieve('What are Tesla two business segments?')
print(f'Dense hits: {len(result[\"dense_ids\"])}')
print(f'BM25 hits: {len(result[\"bm25_ids\"])}')
print(f'Top chunks: {len(result[\"top_child_chunks\"])}')
print(f'Parent sections: {len(result[\"parent_texts\"])}')
print(f'First chunk section: {result[\"top_child_chunks\"][0][\"section_name\"]}')
print(f'Reranker score: {result[\"top_child_chunks\"][0][\"reranker_score\"]:.4f}')
"
```

Expected: 20 dense, 20 BM25, 5 top chunks, 1-5 parent sections, no errors.

- [ ] **Step 6: Commit**

```bash
git add retriever.py tests/test_retriever.py
git commit -m "feat: hybrid retriever — dense + BM25 + RRF + cross-encoder reranker + parent lookup (TDD)"
```

---

## Task 7: generator.py — GPT-4o Answer Generation

**Files:**
- Modify: `generator.py`

- [ ] **Step 1: Replace generator.py**

```python
from openai import OpenAI
from config import OPENAI_API_KEY, GENERATION_MODEL, GENERATION_TEMPERATURE

client = OpenAI(api_key=OPENAI_API_KEY)

_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. "
    "Answer the question using ONLY the provided context from Tesla's 2024 10-K filing. "
    "If the context does not contain enough information to answer, say so clearly. "
    "Be precise with numbers and facts."
)


def generate_answer(question: str, context_chunks: list[str]) -> str:
    if not context_chunks:
        return "No relevant context found to answer this question."

    context = "\n\n---\n\n".join(context_chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=GENERATION_TEMPERATURE,
    )
    return response.choices[0].message.content.strip()
```

- [ ] **Step 2: Smoke test with dummy context**

```bash
python3 -c "
from generator import generate_answer
answer = generate_answer(
    'What are Tesla two segments?',
    ['Tesla operates as two segments: automotive and energy generation and storage.']
)
print(answer)
"
```

Expected: coherent answer referencing the two segments. No errors.

- [ ] **Step 3: Commit**

```bash
git add generator.py
git commit -m "feat: GPT-4o answer generator with system prompt for 10-K context"
```

---

## Task 8: evaluator.py — Golden Dataset Parser + RAGAS Scoring

**Files:**
- Create: `evaluator.py`

- [ ] **Step 1: Implement evaluator.py**

```python
import re
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness, context_recall
from config import GOLDEN_DATASET_PATH
from retriever import retrieve
from generator import generate_answer


def parse_golden_dataset(path: str = GOLDEN_DATASET_PATH) -> list[dict]:
    """Parse golden_dataset.md into list of {question, answer} dicts."""
    with open(path) as f:
        content = f.read()

    qa_pairs = []
    # Match Q/A pairs: **Q<n>.** ... \n**A:** ...
    blocks = re.split(r'---+', content)
    for block in blocks:
        q_match = re.search(r'\*\*Q\d+\.\*\*\s+(.+?)(?=\n\*\*A:\*\*)', block, re.DOTALL)
        a_match = re.search(r'\*\*A:\*\*\s+(.+?)(?=\n\*\*Source:|$)', block, re.DOTALL)
        if q_match and a_match:
            question = q_match.group(1).strip()
            answer = a_match.group(1).strip()
            qa_pairs.append({"question": question, "ground_truth": answer})

    return qa_pairs


def evaluate_single(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> dict:
    """Run RAGAS on one Q&A. Returns dict of metric scores."""
    dataset = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [ground_truth],
    })
    result = evaluate(dataset, metrics=[faithfulness, answer_correctness, context_recall])
    return {
        "faithfulness": float(result["faithfulness"]),
        "answer_correctness": float(result["answer_correctness"]),
        "context_recall": float(result["context_recall"]),
    }


def run_full_evaluation() -> None:
    """Run all 20 golden dataset questions and print aggregate RAGAS scores."""
    pairs = parse_golden_dataset()
    print(f"\nRunning evaluation on {len(pairs)} questions...\n")

    all_scores = {"faithfulness": [], "answer_correctness": [], "context_recall": []}

    for i, pair in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {pair['question'][:60]}...")
        retrieval = retrieve(pair["question"])
        answer = generate_answer(pair["question"], retrieval["parent_texts"])
        scores = evaluate_single(
            question=pair["question"],
            answer=answer,
            contexts=retrieval["parent_texts"],
            ground_truth=pair["ground_truth"],
        )
        for metric, score in scores.items():
            all_scores[metric].append(score)
        print(f"         faithfulness={scores['faithfulness']:.3f} | "
              f"correctness={scores['answer_correctness']:.3f} | "
              f"recall={scores['context_recall']:.3f}")

    print("\n" + "=" * 50)
    print("AGGREGATE RAGAS SCORES (mean across 20 questions):")
    for metric, scores in all_scores.items():
        print(f"  {metric:<22}: {sum(scores)/len(scores):.3f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_full_evaluation()
```

- [ ] **Step 2: Test golden dataset parser**

```bash
python3 -c "
from evaluator import parse_golden_dataset
pairs = parse_golden_dataset()
print(f'Parsed {len(pairs)} Q&A pairs')
print(f'First Q: {pairs[0][\"question\"]}')
print(f'First A: {pairs[0][\"ground_truth\"][:80]}')
"
```

Expected: 20 pairs, first question and answer print correctly.

- [ ] **Step 3: Commit**

```bash
git add evaluator.py
git commit -m "feat: RAGAS evaluator with golden dataset parser and full evaluation run"
```

---

## Task 9: query.py — Entry Point, Wire Everything

**Files:**
- Modify: `query.py`

- [ ] **Step 1: Replace query.py**

```python
import sys
from retriever import retrieve
from generator import generate_answer
from evaluator import evaluate_single
from logger import log_query


def query(question: str, ground_truth: str | None = None) -> str:
    """Run one question through the full pipeline. Returns the answer."""
    retrieval = retrieve(question)
    answer = generate_answer(question, retrieval["parent_texts"])

    ragas_scores: dict = {}
    if ground_truth:
        ragas_scores = evaluate_single(
            question=question,
            answer=answer,
            contexts=retrieval["parent_texts"],
            ground_truth=ground_truth,
        )

    log_query({
        "question": question,
        "top_child_chunks": retrieval["top_child_chunks"],
        "parent_texts": retrieval["parent_texts"],
        "answer": answer,
        "ragas": ragas_scores,
        "dense_candidates": len(retrieval["dense_ids"]),
        "bm25_candidates": len(retrieval["bm25_ids"]),
        "total_candidates": len(retrieval["all_candidates"]),
    })

    return answer


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 query.py 'your question here'")
        print("       python3 query.py 'question' 'expected answer'")
        sys.exit(1)

    q = sys.argv[1]
    gt = sys.argv[2] if len(sys.argv) > 2 else None
    query(q, gt)
```

- [ ] **Step 2: Run a simple end-to-end test**

```bash
python3 query.py "What are Tesla's two business segments?"
```

Expected: console output showing top-5 chunks with dense/BM25/reranker scores, then the answer mentioning "automotive" and "energy generation and storage", then log file path. No errors.

- [ ] **Step 3: Run with ground truth to see RAGAS scores**

```bash
python3 query.py "What are Tesla's two business segments?" "Automotive, and energy generation and storage."
```

Expected: same output plus RAGAS scores printed (faithfulness, answer_correctness, context_recall).

- [ ] **Step 4: Commit**

```bash
git add query.py
git commit -m "feat: query.py entry point wiring retriever + generator + evaluator + logger"
```

---

## Task 10: Run Full Golden Dataset Evaluation

**Files:** None — verification only.

- [ ] **Step 1: Run full evaluation**

```bash
python3 evaluator.py
```

Expected: 20 questions evaluated one by one with per-question scores, then aggregate means printed. Will take ~10-15 minutes (API calls per question).

- [ ] **Step 2: Read the aggregate scores**

Note down the three mean scores. These are your **baseline RAGAS scores** — the starting point of the learning loop. Write them into CLAUDE.md under a new section.

- [ ] **Step 3: Add baseline scores to CLAUDE.md**

Open `CLAUDE.md` and add under a new `## Baseline RAGAS Scores` section:

```markdown
## Baseline RAGAS Scores

First run — [date] — Hierarchical + Hybrid RAG:
- Faithfulness:        [score]
- Answer correctness:  [score]
- Context recall:      [score]

Use these as the benchmark. Any future architecture change should be measured against these.
```

- [ ] **Step 4: Run all unit tests to confirm nothing is broken**

```bash
python3 -m pytest tests/ -v
```

Expected: all tests pass (test_chunker.py × 9, test_retriever.py × 7 = 16 tests).

- [ ] **Step 5: Final commit**

```bash
git add CLAUDE.md
git commit -m "docs: record baseline RAGAS scores from first full evaluation run"
```

---

## Self-Review Notes

**Spec coverage check:**
- ✅ Hierarchical chunking → Task 4 (chunker.py)
- ✅ BM25 + dense hybrid → Task 6 (retriever.py)
- ✅ RRF merge → Task 6 (rrf_merge function + tests)
- ✅ Cross-encoder reranker → Task 6 (retriever.py)
- ✅ Parent lookup → Task 6 (retriever.py)
- ✅ GPT-4o generation → Task 7 (generator.py)
- ✅ RAGAS evaluation → Task 8 (evaluator.py)
- ✅ Console + JSON logging → Task 3 (logger.py)
- ✅ Unit tests for chunker → Task 4 (9 tests)
- ✅ Unit tests for RRF → Task 6 (7 tests)
- ✅ Full evaluation run → Task 10
- ✅ Clean up old files → Task 1

**Type consistency:**
- `rrf_merge(dense_ids, bm25_ids)` defined in Task 6, imported in tests/test_retriever.py ✅
- `retrieve()` returns `dict` with keys `top_child_chunks`, `parent_texts`, `dense_ids`, `bm25_ids`, `all_candidates` — used consistently in query.py and evaluator.py ✅
- `log_query(data: dict)` called in query.py with matching keys ✅
- `evaluate_single(question, answer, contexts, ground_truth)` defined in evaluator.py, called in query.py with same signature ✅
- `build_chunks()` returns `list[Chunk]` — used in ingest.py with `.chunk_type`, `.chunk_id`, `.text`, `.parent_id`, `.section_name` ✅
