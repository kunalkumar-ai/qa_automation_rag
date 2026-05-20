# RAG System Design — Multi-Department Customer Q&A
**Date:** 2026-05-19
**Status:** Approved for implementation

---

## 1. Problem Statement

An electronics manufacturer receives customer questions via email across 14 departments (Legal, Quality Control, Supply Chain, Environment & Sustainability, IT, Energy & Water, After-Sales, Product Safety, Finance, HR, R&D, Marketing, Logistics, Customer Support). Currently, a staff member manually reads each email, searches for the answer, and replies. This is slow and does not scale.

The goal is a RAG-powered tool where the staff member pastes the customer's question, instantly gets a well-formed suggested answer drawn from the company's documents, and copies it into their email reply.

---

## 2. Architecture

```
[Gradio UI] — staff pastes customer question
      │
      ▼
[Department Classifier] — GPT-4o identifies which department owns this question
      │  e.g., "legal"
      ▼
[ChromaDB Retriever] — vector search filtered by department metadata
      │  returns top 3-5 relevant document chunks
      ▼
[Answer Generator] — GPT-4o synthesizes a professional reply from retrieved chunks
      │
      ▼
[Gradio UI] — staff copies the answer into their email
```

**One-time ingestion pipeline (run once per document update):**
```
Raw .txt documents (per department folder)
      │
      ▼
Chunking (500 words, 50-word overlap)
      │
      ▼
OpenAI Embeddings (text-embedding-3-small)
      │
      ▼
ChromaDB storage with metadata: {"department": "<dept_name>"}
```

---

## 3. System Components

### 3.1 Document Ingestion (`ingest.py`)
- Walks the `docs/` folder; each subfolder name is the department name
- Splits each document into chunks of ~500 words with 50-word overlap (overlap prevents losing context at chunk boundaries)
- Calls OpenAI `text-embedding-3-small` to embed each chunk
- Stores chunks in a single ChromaDB collection with `department` metadata tag
- Idempotent: clears and re-ingests on each run so documents can be updated freely

### 3.2 Department Classifier (`classifier.py`)
- Receives the raw customer question
- Sends it to GPT-4o with a system prompt listing all 14 departments and their responsibilities
- Returns a single department name (lowercase, matching folder name)
- Returns `"unknown"` if GPT-4o cannot identify a confident match — no answer is generated in this case

### 3.3 Retriever (`retriever.py`)
- Takes question + department name
- Embeds the question using `text-embedding-3-small`
- Queries ChromaDB with `where={"department": dept_name}` filter
- Returns top 3 most semantically similar chunks

### 3.4 Answer Generator (`generator.py`)
- Takes question + retrieved chunks
- Builds a prompt: system role (professional customer support agent for an electronics manufacturer) + retrieved context + customer question
- Calls GPT-4o to generate a clear, professional reply
- Returns the answer text only (no extra formatting)

### 3.5 Gradio UI (`app.py`)
- Single text input: "Paste customer question here"
- On submit: runs classifier → retriever → generator pipeline
- Displays: detected department + suggested answer
- Copy-friendly output text box

---

## 4. Project File Structure

```
rag/
├── docs/                          # Sample documents (one folder per department)
│   ├── legal/
│   │   ├── warranty_policy.txt
│   │   ├── terms_of_service.txt
│   │   └── privacy_policy.txt
│   ├── quality_control/
│   │   ├── iso_certification.txt
│   │   └── defect_return_process.txt
│   ├── supply_chain/
│   │   ├── delivery_timelines.txt
│   │   └── shipping_regions.txt
│   ├── environment_sustainability/
│   │   ├── rohs_compliance.txt
│   │   └── recycling_program.txt
│   ├── it/
│   │   ├── firmware_support.txt
│   │   └── software_compatibility.txt
│   ├── energy_water/
│   │   ├── energy_ratings.txt
│   │   └── power_consumption.txt
│   ├── after_sales/
│   │   ├── repair_process.txt
│   │   └── spare_parts.txt
│   ├── product_safety/
│   │   ├── ce_marking.txt
│   │   └── recall_procedures.txt
│   ├── finance/
│   │   ├── payment_terms.txt
│   │   └── invoice_policy.txt
│   ├── hr/
│   │   ├── careers_faq.txt
│   │   └── internship_program.txt
│   ├── rd/
│   │   ├── product_roadmap_faq.txt
│   │   └── innovation_programs.txt
│   ├── marketing/
│   │   ├── product_catalog.txt
│   │   └── partner_guidelines.txt
│   ├── logistics/
│   │   ├── import_export_info.txt
│   │   └── customs_documentation.txt
│   └── customer_support/
│       ├── escalation_process.txt
│       └── sla_response_times.txt
│
├── ingest.py                      # One-time ingestion script
├── classifier.py                  # Department classifier
├── retriever.py                   # ChromaDB retrieval with metadata filter
├── generator.py                   # GPT-4o answer generation
├── app.py                         # Gradio UI entry point
├── config.py                      # API keys, model names, constants
├── requirements.txt               # Python dependencies
├── .env                           # OPENAI_API_KEY (not committed to git)
└── chroma_db/                     # Auto-created by ChromaDB on first ingest
```

---

## 5. Data — Sample Documents

Documents contain **raw information prose only** — not Q&A pairs. GPT-4o synthesizes answers from the raw content at query time. This means:
- Any customer question is answerable, not just pre-written ones
- Updating a policy means updating one document, not rewriting Q&A pairs
- GPT-4o can combine context from multiple chunks to answer complex questions

Each document is ~300-500 words of formal business prose with realistic specifics:
- Product names: "ElectroTech ET-X200 series"
- Policy revision numbers and dates
- Specific regulatory standards (RoHS 3, WEEE Directive 2012/19/EU, ISO 9001:2015)
- Contact details and process steps

28 total documents: 2 per department × 14 departments.

---

## 6. Tech Stack

| Component | Technology | Reason |
|---|---|---|
| LLM (generation + classification) | GPT-4o (OpenAI) | High quality, well-documented |
| Embeddings | text-embedding-3-small (OpenAI) | Fast, cheap, high quality |
| Vector Database | ChromaDB (local) | Zero setup, persists to disk, great for learning |
| UI | Gradio | Python-native, browser UI, no frontend knowledge needed |
| Language | Python 3.10+ | Ecosystem matches all tools above |

**Key dependencies:**
```
openai
chromadb
gradio
python-dotenv
```

---

## 7. Data Flow (End to End)

**Ingestion (once):**
1. `ingest.py` walks `docs/` and reads all `.txt` files
2. Each file is chunked into ~500 word segments with 50-word overlap
3. Each chunk is embedded via OpenAI API
4. Chunk text + vector + `{"department": folder_name}` stored in ChromaDB

**Query (every customer question):**
1. Staff pastes question into Gradio UI
2. `classifier.py` sends question to GPT-4o → returns `"legal"` (or other dept)
3. `retriever.py` embeds question, queries ChromaDB filtered to `department="legal"`, returns top 3 chunks
4. `generator.py` sends question + 3 chunks to GPT-4o → returns professional answer
5. Gradio displays the department detected + the answer

---

## 8. Error Handling

- **Unknown department (low classifier confidence):** The UI displays "This question could not be matched to a department. Please handle this manually." No answer is generated. The staff member treats it as a normal manual email. Confidence is considered low when GPT-4o cannot identify a single clear department from the 14 options.
- **No relevant chunks found:** Generator is instructed to respond "I don't have enough information to answer this question. Please contact our support team directly."
- **OpenAI API failure:** Gradio shows a user-friendly error message; raw exception is logged to console
- **Empty question submitted:** Gradio validates input is non-empty before triggering pipeline

---

## 9. Future Extensibility (not in scope now)

- Swap ChromaDB for Pinecone when moving to production
- Add explicit confidence score output from classifier; currently confidence is binary (known/unknown) — future version could show a score (e.g., 0.85) so staff can judge borderline cases themselves
- Plug in real company documents by replacing contents of `docs/` folder and re-running `ingest.py`
- Add Gmail API integration to auto-fetch questions and draft replies

---

## 10. What is Out of Scope

- Email sending or receiving
- User authentication
- Multi-language support
- Document upload via UI (docs are managed via filesystem for now)
- Analytics or logging of past questions
