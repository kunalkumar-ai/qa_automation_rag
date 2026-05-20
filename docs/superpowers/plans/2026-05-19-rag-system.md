# RAG Customer Q&A System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Gradio web app where a staff member pastes a customer email question and receives a GPT-4o-generated answer sourced from 14 department document collections stored in ChromaDB.

**Architecture:** Metadata-Filtered RAG — each document chunk is tagged with its department name. GPT-4o first classifies which department owns the question, then ChromaDB retrieves only that department's chunks, and GPT-4o generates a professional answer from them. If no department is identified, the UI instructs the user to handle manually — no answer is generated.

**Tech Stack:** Python 3.10+, OpenAI (GPT-4o + text-embedding-3-small), ChromaDB (local), Gradio, pytest, python-dotenv

---

## File Map

| File | Responsibility |
|---|---|
| `config.py` | All constants and settings (model names, paths, department list) |
| `ingest.py` | One-time script: chunk docs → embed → store in ChromaDB |
| `classifier.py` | GPT-4o call to identify department from customer question |
| `retriever.py` | ChromaDB query filtered by department metadata |
| `generator.py` | GPT-4o call to produce answer from question + retrieved chunks |
| `app.py` | Gradio UI wiring classifier → retriever → generator |
| `generate_docs.py` | One-time script to create all 28 sample department documents |
| `tests/test_ingest.py` | Tests for chunk_text function |
| `tests/test_classifier.py` | Tests for classify_department function |
| `tests/test_retriever.py` | Tests for retrieve_chunks function |
| `tests/test_generator.py` | Tests for generate_answer function |
| `pyproject.toml` | Pytest configuration (pythonpath = root) |
| `requirements.txt` | Python dependencies |
| `.env` | OPENAI_API_KEY (never committed) |

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `.env`
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `config.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `requirements.txt`**

```
openai>=1.30.0
chromadb>=0.5.0
gradio>=4.0.0
python-dotenv>=1.0.0
pytest>=8.0.0
```

- [ ] **Step 2: Create `.env`**

```
OPENAI_API_KEY=your_openai_api_key_here
```

- [ ] **Step 3: Create `.gitignore`**

```
.env
chroma_db/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 4: Create `pyproject.toml`**

This tells pytest to treat the project root as importable so test files can do `from classifier import ...` without any path hacks.

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
```

- [ ] **Step 5: Create `config.py`**

```python
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
CHROMA_PATH = "chroma_db"
DOCS_PATH = "docs"
TOP_K_RESULTS = 3
CHUNK_SIZE = 500   # words per chunk
CHUNK_OVERLAP = 50  # words shared between adjacent chunks

DEPARTMENTS = [
    "legal",
    "quality_control",
    "supply_chain",
    "environment_sustainability",
    "it",
    "energy_water",
    "after_sales",
    "product_safety",
    "finance",
    "hr",
    "rd",
    "marketing",
    "logistics",
    "customer_support",
]

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
```

- [ ] **Step 6: Create `tests/__init__.py`**

Empty file — makes `tests/` a Python package.

```python
```

- [ ] **Step 7: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: All packages install without errors.

- [ ] **Step 8: Commit**

```bash
git init
git add requirements.txt .gitignore pyproject.toml config.py tests/__init__.py
git commit -m "chore: project setup with config and dependencies"
```

---

## Task 2: Create Sample Documents

**Files:**
- Create: `generate_docs.py`
- Create: `docs/legal/warranty_policy.txt` (and 27 more — generated by script)

Sample documents contain **raw information prose** — not Q&A pairs. GPT-4o reads them at query time to formulate answers. Each document is ~400 words of formal business language with realistic specifics (product names, standard numbers, dates).

- [ ] **Step 1: Create `generate_docs.py`**

Run this script once to create all 28 sample documents. It uses GPT-4o to write each document so the content is realistic and varied.

```python
import os
from openai import OpenAI
from config import OPENAI_API_KEY, CHAT_MODEL, DOCS_PATH

client = OpenAI(api_key=OPENAI_API_KEY)

DOCUMENTS = {
    "legal": [
        ("warranty_policy.txt", "ElectroTech 24-month product warranty: coverage scope, claim process, exclusions, contact details"),
        ("terms_of_service.txt", "ElectroTech terms of service: purchase conditions, liability limits, dispute resolution, governing law"),
    ],
    "quality_control": [
        ("iso_certification.txt", "ElectroTech ISO 9001:2015 certification: scope, audit schedule, quality management process, certificate validity"),
        ("defect_return_process.txt", "ElectroTech defective product return process: eligibility, RMA number, packaging instructions, refund timelines"),
    ],
    "supply_chain": [
        ("delivery_timelines.txt", "ElectroTech standard delivery timelines by region: domestic 3-5 days, EU 7-10 days, rest of world 14-21 days"),
        ("shipping_regions.txt", "ElectroTech shipping coverage: supported countries, restricted regions, courier partners, tracking information"),
    ],
    "environment_sustainability": [
        ("rohs_compliance.txt", "ElectroTech RoHS 3 compliance (EU 2015/863): restricted substances, declaration of conformity, product scope"),
        ("recycling_program.txt", "ElectroTech WEEE recycling program: drop-off points, take-back scheme, packaging recycling, annual targets"),
    ],
    "it": [
        ("firmware_support.txt", "ElectroTech firmware update policy: supported models, update frequency, how to update, end-of-life schedule"),
        ("software_compatibility.txt", "ElectroTech software compatibility: supported OS versions (Windows 10/11, macOS 13+, Ubuntu 22.04), driver downloads"),
    ],
    "energy_water": [
        ("energy_ratings.txt", "ElectroTech EU energy label ratings for product range: ET-X200 (A+), ET-X300 (A++), ET-S100 (B), test conditions"),
        ("power_consumption.txt", "ElectroTech product power consumption specs: standby wattage, active wattage, annual kWh estimates, ErP compliance"),
    ],
    "after_sales": [
        ("repair_process.txt", "ElectroTech authorised repair process: booking, turnaround times (in-warranty 5 days, out-of-warranty 10 days), courier collection"),
        ("spare_parts.txt", "ElectroTech spare parts availability: supported models, ordering process, lead times, pricing structure"),
    ],
    "product_safety": [
        ("ce_marking.txt", "ElectroTech CE marking: applicable directives (LVD, EMC, RED), notified body, declaration of conformity process"),
        ("recall_procedures.txt", "ElectroTech product recall process: how customers are notified, return instructions, replacement timelines, RAPEX reporting"),
    ],
    "finance": [
        ("payment_terms.txt", "ElectroTech B2B payment terms: net-30 standard, early payment discount 2/10, accepted methods, invoice dispute process"),
        ("invoice_policy.txt", "ElectroTech invoice policy: VAT treatment, credit note process, invoice correction requests, retention period"),
    ],
    "hr": [
        ("careers_faq.txt", "ElectroTech careers FAQ: how to apply, interview process, graduate programmes, relocation support, benefits overview"),
        ("internship_program.txt", "ElectroTech internship programme: duration (6 months), departments available, application window (January-March), stipend"),
    ],
    "rd": [
        ("product_roadmap_faq.txt", "ElectroTech product roadmap FAQ: upcoming product lines, feature request process, beta programme, NDA-covered information policy"),
        ("innovation_programs.txt", "ElectroTech innovation partnerships: university collaboration programme, startup accelerator, joint development agreements"),
    ],
    "marketing": [
        ("product_catalog.txt", "ElectroTech 2025 product catalog: ET-X200, ET-X300, ET-S100, ET-P50 — key specs, target segments, pricing tiers"),
        ("partner_guidelines.txt", "ElectroTech authorised reseller brand guidelines: logo usage, approved imagery, co-marketing rules, prohibited claims"),
    ],
    "logistics": [
        ("import_export_info.txt", "ElectroTech import/export information: HS codes, country of origin, EORI registration, export control classification"),
        ("customs_documentation.txt", "ElectroTech customs documentation: commercial invoice requirements, packing list, certificate of origin, incoterms used (DDP/DAP)"),
    ],
    "customer_support": [
        ("escalation_process.txt", "ElectroTech customer support escalation: tier 1 (email 48h SLA), tier 2 (phone, 24h SLA), tier 3 (management, 4h SLA)"),
        ("sla_response_times.txt", "ElectroTech service level agreement: response time commitments by channel and severity, measurement method, exclusions"),
    ],
}


def generate_document(topic: str) -> str:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You write internal company documents for ElectroTech, an electronics manufacturer. "
                    "Write in formal business prose (not bullet points). "
                    "Include realistic specifics: product model names (ET-X200, ET-X300, ET-S100, ET-P50), "
                    "regulation numbers, dates (use year 2025), policy revision numbers, and contact email addresses. "
                    "Length: approximately 400 words. Do not use bullet points or headers — write in paragraphs."
                ),
            },
            {"role": "user", "content": f"Write a company document about: {topic}"},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def main():
    for dept, docs in DOCUMENTS.items():
        dept_path = os.path.join(DOCS_PATH, dept)
        os.makedirs(dept_path, exist_ok=True)
        for filename, topic in docs:
            filepath = os.path.join(dept_path, filename)
            print(f"Generating {dept}/{filename}...")
            content = generate_document(topic)
            with open(filepath, "w") as f:
                f.write(content)
    print("All documents generated.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the document generation script**

```bash
python generate_docs.py
```

Expected output:
```
Generating legal/warranty_policy.txt...
Generating legal/terms_of_service.txt...
...
Generating customer_support/sla_response_times.txt...
All documents generated.
```

- [ ] **Step 3: Verify documents were created**

```bash
find docs -name "*.txt" | wc -l
```

Expected: `28`

- [ ] **Step 4: Spot-check one document**

```bash
cat docs/legal/warranty_policy.txt
```

Expected: ~400 words of formal prose about ElectroTech warranty terms. No bullet points.

- [ ] **Step 5: Commit**

```bash
git add generate_docs.py docs/
git commit -m "feat: add sample document generation script and 28 department documents"
```

---

## Task 3: Document Ingestion

**Files:**
- Create: `tests/test_ingest.py`
- Create: `ingest.py`

The ingestion pipeline reads `.txt` files from `docs/`, splits them into word-based chunks, embeds each chunk via OpenAI, and stores chunks + vectors + department metadata in ChromaDB.

- [ ] **Step 1: Write failing tests for `chunk_text`**

`chunk_text` is a pure function — no mocking needed.

```python
# tests/test_ingest.py
from ingest import chunk_text


def test_chunk_text_short_document_returns_single_chunk():
    text = " ".join(["word"] * 100)
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_long_document_returns_multiple_chunks():
    text = " ".join([str(i) for i in range(1000)])
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 3


def test_chunk_text_overlap_shares_words_between_adjacent_chunks():
    text = " ".join([str(i) for i in range(600)])
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    last_words_of_first = set(chunks[0].split()[-50:])
    first_words_of_second = set(chunks[1].split()[:50])
    assert last_words_of_first == first_words_of_second


def test_chunk_text_empty_string_returns_empty_list():
    chunks = chunk_text("", chunk_size=500, overlap=50)
    assert chunks == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ingest.py -v
```

Expected: `ImportError` — `ingest` module does not exist yet.

- [ ] **Step 3: Create `ingest.py`**

```python
import os
import chromadb
from openai import OpenAI
from config import OPENAI_API_KEY, EMBEDDING_MODEL, CHROMA_PATH, DOCS_PATH, CHUNK_SIZE, CHUNK_OVERLAP

client = OpenAI(api_key=OPENAI_API_KEY)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
        if end == len(words):
            break
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def ingest_docs(docs_path: str = DOCS_PATH, chroma_path: str = CHROMA_PATH) -> None:
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    try:
        chroma_client.delete_collection("documents")
    except Exception:
        pass
    collection = chroma_client.create_collection("documents")

    for dept in os.listdir(docs_path):
        dept_path = os.path.join(docs_path, dept)
        if not os.path.isdir(dept_path) or dept.startswith(".") or dept == "superpowers":
            continue
        for filename in os.listdir(dept_path):
            if not filename.endswith(".txt"):
                continue
            with open(os.path.join(dept_path, filename)) as f:
                text = f.read()
            chunks = chunk_text(text)
            embeddings = embed_texts(chunks)
            ids = [f"{dept}__{filename}__{i}" for i in range(len(chunks))]
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=[{"department": dept, "source": filename} for _ in chunks],
            )
            print(f"Ingested {len(chunks)} chunk(s) from {dept}/{filename}")

    print("Ingestion complete.")


if __name__ == "__main__":
    ingest_docs()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ingest.py -v
```

Expected:
```
PASSED tests/test_ingest.py::test_chunk_text_short_document_returns_single_chunk
PASSED tests/test_ingest.py::test_chunk_text_long_document_returns_multiple_chunks
PASSED tests/test_ingest.py::test_chunk_text_overlap_shares_words_between_adjacent_chunks
PASSED tests/test_ingest.py::test_chunk_text_empty_string_returns_empty_list
4 passed
```

- [ ] **Step 5: Run the ingestion script against real docs**

```bash
python ingest.py
```

Expected: Lines like `Ingested 1 chunk(s) from legal/warranty_policy.txt` for all 28 files, ending with `Ingestion complete.`

- [ ] **Step 6: Verify ChromaDB was populated**

```bash
python -c "
import chromadb
c = chromadb.PersistentClient(path='chroma_db')
col = c.get_collection('documents')
print('Total chunks stored:', col.count())
"
```

Expected: `Total chunks stored: 28` (or more if any docs were long enough to produce multiple chunks)

- [ ] **Step 7: Commit**

```bash
git add ingest.py tests/test_ingest.py
git commit -m "feat: add document ingestion pipeline with chunking and ChromaDB storage"
```

---

## Task 4: Department Classifier

**Files:**
- Create: `tests/test_classifier.py`
- Create: `classifier.py`

`classify_department` sends the customer question to GPT-4o with a list of departments and their descriptions. It returns a department string (e.g. `"legal"`) or `"unknown"` if GPT-4o cannot identify a confident match.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_classifier.py
from unittest.mock import patch, MagicMock
from classifier import classify_department


def _mock_response(content: str) -> MagicMock:
    mock = MagicMock()
    mock.choices[0].message.content = content
    return mock


@patch("classifier.client")
def test_classify_returns_known_department(mock_client):
    mock_client.chat.completions.create.return_value = _mock_response("legal")
    result = classify_department("What is your warranty period?")
    assert result == "legal"


@patch("classifier.client")
def test_classify_returns_unknown_for_unrecognised_question(mock_client):
    mock_client.chat.completions.create.return_value = _mock_response("unknown")
    result = classify_department("What is the meaning of life?")
    assert result == "unknown"


@patch("classifier.client")
def test_classify_returns_unknown_for_invalid_gpt_response(mock_client):
    mock_client.chat.completions.create.return_value = _mock_response("made_up_department")
    result = classify_department("Some random question")
    assert result == "unknown"


@patch("classifier.client")
def test_classify_strips_whitespace_and_lowercases(mock_client):
    mock_client.chat.completions.create.return_value = _mock_response("  Quality_Control  ")
    result = classify_department("Is the product ISO certified?")
    assert result == "quality_control"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_classifier.py -v
```

Expected: `ImportError` — `classifier` module does not exist yet.

- [ ] **Step 3: Create `classifier.py`**

```python
from openai import OpenAI
from config import OPENAI_API_KEY, CHAT_MODEL, DEPARTMENTS, DEPARTMENT_DESCRIPTIONS

client = OpenAI(api_key=OPENAI_API_KEY)


def classify_department(question: str) -> str:
    """Returns a department name from DEPARTMENTS, or 'unknown' if not confident."""
    dept_list = "\n".join(
        f"- {dept}: {desc}" for dept, desc in DEPARTMENT_DESCRIPTIONS.items()
    )
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a department classifier for ElectroTech, an electronics manufacturer. "
                    "Given a customer question, identify which single department it belongs to. "
                    "Reply with ONLY the department name from the list below, exactly as written. "
                    "If the question does not clearly belong to any one department, reply with 'unknown'.\n\n"
                    f"Departments:\n{dept_list}"
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    result = response.choices[0].message.content.strip().lower()
    return result if result in DEPARTMENTS else "unknown"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_classifier.py -v
```

Expected:
```
PASSED tests/test_classifier.py::test_classify_returns_known_department
PASSED tests/test_classifier.py::test_classify_returns_unknown_for_unrecognised_question
PASSED tests/test_classifier.py::test_classify_returns_unknown_for_invalid_gpt_response
PASSED tests/test_classifier.py::test_classify_strips_whitespace_and_lowercases
4 passed
```

- [ ] **Step 5: Commit**

```bash
git add classifier.py tests/test_classifier.py
git commit -m "feat: add department classifier using GPT-4o"
```

---

## Task 5: Retriever

**Files:**
- Create: `tests/test_retriever.py`
- Create: `retriever.py`

`retrieve_chunks` embeds the question, queries ChromaDB with a department filter, and returns the top-3 most relevant text chunks. If ChromaDB returns no results, it returns an empty list.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_retriever.py
from unittest.mock import patch, MagicMock
from retriever import retrieve_chunks


def _mock_embedding_response(vector: list[float]) -> MagicMock:
    mock = MagicMock()
    mock.data[0].embedding = vector
    return mock


def _mock_chroma_results(docs: list[str]) -> dict:
    return {"documents": [docs]}


@patch("retriever.chromadb.PersistentClient")
@patch("retriever.client")
def test_retrieve_returns_matching_chunks(mock_openai, mock_chroma_cls):
    mock_openai.embeddings.create.return_value = _mock_embedding_response([0.1, 0.2, 0.3])
    mock_collection = MagicMock()
    mock_collection.query.return_value = _mock_chroma_results(["Warranty lasts 24 months.", "Claim via email."])
    mock_chroma_cls.return_value.get_collection.return_value = mock_collection

    chunks = retrieve_chunks("What is the warranty?", "legal")

    assert len(chunks) == 2
    assert chunks[0] == "Warranty lasts 24 months."


@patch("retriever.chromadb.PersistentClient")
@patch("retriever.client")
def test_retrieve_returns_empty_list_when_no_results(mock_openai, mock_chroma_cls):
    mock_openai.embeddings.create.return_value = _mock_embedding_response([0.1])
    mock_collection = MagicMock()
    mock_collection.query.return_value = {"documents": []}
    mock_chroma_cls.return_value.get_collection.return_value = mock_collection

    chunks = retrieve_chunks("Something totally irrelevant", "legal")

    assert chunks == []


@patch("retriever.chromadb.PersistentClient")
@patch("retriever.client")
def test_retrieve_passes_department_filter_to_chroma(mock_openai, mock_chroma_cls):
    mock_openai.embeddings.create.return_value = _mock_embedding_response([0.1])
    mock_collection = MagicMock()
    mock_collection.query.return_value = _mock_chroma_results(["Some chunk."])
    mock_chroma_cls.return_value.get_collection.return_value = mock_collection

    retrieve_chunks("Any question", "quality_control")

    call_kwargs = mock_collection.query.call_args.kwargs
    assert call_kwargs["where"] == {"department": "quality_control"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_retriever.py -v
```

Expected: `ImportError` — `retriever` module does not exist yet.

- [ ] **Step 3: Create `retriever.py`**

```python
import chromadb
from openai import OpenAI
from config import OPENAI_API_KEY, EMBEDDING_MODEL, CHROMA_PATH, TOP_K_RESULTS

client = OpenAI(api_key=OPENAI_API_KEY)


def retrieve_chunks(question: str, department: str, chroma_path: str = CHROMA_PATH) -> list[str]:
    """Returns top-K document chunks from ChromaDB filtered by department."""
    embedding_response = client.embeddings.create(model=EMBEDDING_MODEL, input=[question])
    query_vector = embedding_response.data[0].embedding

    chroma_client = chromadb.PersistentClient(path=chroma_path)
    collection = chroma_client.get_collection("documents")
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=TOP_K_RESULTS,
        where={"department": department},
    )
    return results["documents"][0] if results["documents"] else []
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_retriever.py -v
```

Expected:
```
PASSED tests/test_retriever.py::test_retrieve_returns_matching_chunks
PASSED tests/test_retriever.py::test_retrieve_returns_empty_list_when_no_results
PASSED tests/test_retriever.py::test_retrieve_passes_department_filter_to_chroma
3 passed
```

- [ ] **Step 5: Commit**

```bash
git add retriever.py tests/test_retriever.py
git commit -m "feat: add ChromaDB retriever with department metadata filtering"
```

---

## Task 6: Answer Generator

**Files:**
- Create: `tests/test_generator.py`
- Create: `generator.py`

`generate_answer` sends the question and retrieved chunks to GPT-4o. If no chunks were retrieved, it returns a polite "no information" message without calling the API.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_generator.py
from unittest.mock import patch, MagicMock
from generator import generate_answer, NO_INFO_RESPONSE


def _mock_chat_response(content: str) -> MagicMock:
    mock = MagicMock()
    mock.choices[0].message.content = content
    return mock


@patch("generator.client")
def test_generate_returns_answer_when_chunks_provided(mock_client):
    mock_client.chat.completions.create.return_value = _mock_chat_response(
        "The warranty period is 24 months from the date of purchase."
    )
    result = generate_answer("What is the warranty?", ["Warranty covers 24 months."])
    assert "24 months" in result


@patch("generator.client")
def test_generate_returns_no_info_response_when_no_chunks(mock_client):
    result = generate_answer("What is the warranty?", [])
    assert result == NO_INFO_RESPONSE
    mock_client.chat.completions.create.assert_not_called()


@patch("generator.client")
def test_generate_includes_all_chunks_in_context(mock_client):
    mock_client.chat.completions.create.return_value = _mock_chat_response("Some answer.")
    chunks = ["Chunk one content.", "Chunk two content.", "Chunk three content."]
    generate_answer("Some question?", chunks)

    call_args = mock_client.chat.completions.create.call_args
    system_message = call_args.kwargs["messages"][0]["content"]
    assert "Chunk one content." in system_message
    assert "Chunk two content." in system_message
    assert "Chunk three content." in system_message
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_generator.py -v
```

Expected: `ImportError` — `generator` module does not exist yet.

- [ ] **Step 3: Create `generator.py`**

```python
from openai import OpenAI
from config import OPENAI_API_KEY, CHAT_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

NO_INFO_RESPONSE = (
    "I don't have enough information to answer this question. "
    "Please contact our support team directly."
)


def generate_answer(question: str, chunks: list[str]) -> str:
    """Generates a professional customer support answer from retrieved chunks."""
    if not chunks:
        return NO_INFO_RESPONSE

    context = "\n\n---\n\n".join(chunks)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional customer support agent for ElectroTech, an electronics manufacturer. "
                    "Answer the customer's question using only the information provided in the context below. "
                    "Be clear, concise, and professional. Do not invent information not found in the context. "
                    "If the context does not contain enough information to fully answer, say so politely.\n\n"
                    f"Context:\n{context}"
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_generator.py -v
```

Expected:
```
PASSED tests/test_generator.py::test_generate_returns_answer_when_chunks_provided
PASSED tests/test_generator.py::test_generate_returns_no_info_response_when_no_chunks
PASSED tests/test_generator.py::test_generate_includes_all_chunks_in_context
3 passed
```

- [ ] **Step 5: Run the full test suite to confirm nothing is broken**

```bash
pytest -v
```

Expected: All 11 tests pass.

- [ ] **Step 6: Commit**

```bash
git add generator.py tests/test_generator.py
git commit -m "feat: add GPT-4o answer generator with no-chunks guard"
```

---

## Task 7: Gradio UI

**Files:**
- Create: `app.py`

The UI has a single text input for the customer question and two outputs: detected department and suggested answer. If the classifier returns `"unknown"`, no answer is generated — the UI tells the staff to handle the email manually.

- [ ] **Step 1: Create `app.py`**

```python
import gradio as gr
from classifier import classify_department
from retriever import retrieve_chunks
from generator import generate_answer


def handle_question(question: str) -> tuple[str, str]:
    question = question.strip()
    if not question:
        return "", "Please enter a customer question."

    try:
        department = classify_department(question)

        if department == "unknown":
            return (
                "Unknown",
                "This question could not be matched to a department. Please handle this manually.",
            )

        chunks = retrieve_chunks(question, department)
        answer = generate_answer(question, chunks)
        dept_display = department.replace("_", " ").title()
        return dept_display, answer
    except Exception as e:
        print(f"Pipeline error: {e}")
        return "", "An error occurred while processing your question. Please try again."


with gr.Blocks(title="ElectroTech Customer Q&A") as app:
    gr.Markdown("## ElectroTech Customer Q&A Assistant")
    gr.Markdown(
        "Paste a customer email question below. "
        "The system will find the relevant department and suggest a reply."
    )

    question_input = gr.Textbox(
        label="Customer Question",
        placeholder="Paste the customer's question here...",
        lines=4,
    )
    submit_btn = gr.Button("Get Answer", variant="primary")

    with gr.Row():
        dept_output = gr.Textbox(label="Detected Department", interactive=False)
        answer_output = gr.Textbox(label="Suggested Answer", lines=8, interactive=False)

    submit_btn.click(
        fn=handle_question,
        inputs=[question_input],
        outputs=[dept_output, answer_output],
    )

if __name__ == "__main__":
    app.launch()
```

- [ ] **Step 2: Launch the app**

```bash
python app.py
```

Expected output:
```
Running on local URL:  http://127.0.0.1:7860
```

Open `http://127.0.0.1:7860` in your browser.

- [ ] **Step 3: Manual test — known department question**

Paste this into the UI:
```
I bought an ET-X200 three months ago and it stopped working. Is it still covered under warranty?
```

Expected:
- Detected Department: `Legal`
- Suggested Answer: A professional paragraph explaining the 24-month warranty and how to submit a claim.

- [ ] **Step 4: Manual test — unknown department question**

Paste this into the UI:
```
What is the best recipe for chocolate cake?
```

Expected:
- Detected Department: `Unknown`
- Suggested Answer: `This question could not be matched to a department. Please handle this manually.`

- [ ] **Step 5: Manual test — quality control question**

Paste this into the UI:
```
Does your ET-X300 have ISO 9001 certification?
```

Expected:
- Detected Department: `Quality Control`
- Suggested Answer: A professional response referencing ISO 9001:2015 certification.

- [ ] **Step 6: Stop the server and commit**

Press `Ctrl+C` in the terminal to stop the server.

```bash
git add app.py
git commit -m "feat: add Gradio UI wiring classifier, retriever, and generator"
```

---

## Task 8: End-to-End Smoke Test

**Files:** No new files — verify the full pipeline works correctly.

- [ ] **Step 1: Run the full test suite one final time**

```bash
pytest -v
```

Expected: All 11 tests pass with no warnings.

- [ ] **Step 2: Verify ChromaDB has data (in case chroma_db was deleted)**

```bash
python -c "
import chromadb
c = chromadb.PersistentClient(path='chroma_db')
col = c.get_collection('documents')
print('Chunks in ChromaDB:', col.count())
"
```

If this fails with a collection-not-found error, re-run `python ingest.py` first.

- [ ] **Step 3: Run a quick pipeline smoke test from the terminal**

```bash
python -c "
from classifier import classify_department
from retriever import retrieve_chunks
from generator import generate_answer

q = 'How do I return a defective ET-X200?'
dept = classify_department(q)
print('Department:', dept)
chunks = retrieve_chunks(q, dept)
print('Chunks retrieved:', len(chunks))
answer = generate_answer(q, chunks)
print('Answer:', answer[:200])
"
```

Expected:
```
Department: quality_control
Chunks retrieved: 3
Answer: Thank you for reaching out. To return a defective ET-X200...
```

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "chore: verified end-to-end RAG pipeline working"
```

---

## How to Update Documents Later (Real Company Docs)

When you are ready to plug in real company documents:

1. Replace the contents of each folder in `docs/` with your real `.txt` files (or convert PDFs/Word docs to `.txt` first)
2. Re-run `python ingest.py` — it clears and re-populates ChromaDB automatically
3. No code changes needed

---

## Dependency Summary

```
openai>=1.30.0       # GPT-4o and text-embedding-3-small
chromadb>=0.5.0      # local vector database
gradio>=4.0.0        # web UI
python-dotenv>=1.0.0 # load OPENAI_API_KEY from .env
pytest>=8.0.0        # test runner
```
