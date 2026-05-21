A Retrieval-Augmented Generation (RAG) tool that helps customer support staff answer email questions faster. Paste a customer
  question, get a suggested reply sourced from company documents, copy it into your email.
  
  ---
  
  ## How It Works
  
  Customer Email Question
          ↓
  Department Classifier (GPT-4o)
          ↓ e.g. "Legal"
  ChromaDB Vector Search (filtered by department)
          ↓ top 3 relevant document chunks
  Answer Generator (GPT-4o)
          ↓
  Suggested Reply (copy into email)

  If the question cannot be matched to any department, the system tells you to handle it manually — it never generates a guess.
  
  ---
  
  ## Prerequisites
  
  - Python 3.10+
  - An OpenAI API key

  ---
  
  ## Setup

  **1. Clone the repo and install dependencies:**
  ```bash
  pip install -r requirements.txt
  
  2. Add your OpenAI API key:
  # Edit .env and replace the placeholder
  OPENAI_API_KEY=your_key_here

  3. Generate sample documents (first time only):
  python3 generate_docs.py

  4. Ingest documents into ChromaDB (first time only):
  python3 ingest.py
  
  5. Run the app:
  python3 app.py
  
  Open http://127.0.0.1:7860 in your browser.
  
  ---
  Usage

  1. Copy a customer email question
  2. Paste it into the text box
  3. Click Get Answer
  4. Review the suggested reply and the detected department
  5. Copy the answer into your email
  
  ---
  Running Tests

  python3 -m pytest -v
  
  Expected: 14 tests passing. All tests mock external API calls — no OpenAI charges during testing.
  
  ---
  Project Structure
  
  rag/
  ├── docs/                    # Department documents (2 per department)
  │   ├── legal/
  │   ├── quality_control/
  │   ├── supply_chain/
  │   └── ...                  # 14 departments total
  ├── tests/
  │   ├── test_ingest.py
  │   ├── test_classifier.py
  │   ├── test_retriever.py
  │   └── test_generator.py
  ├── config.py                # All constants (models, paths, departments)
  ├── generate_docs.py         # One-time: generate sample documents
  ├── ingest.py                # One-time: chunk → embed → store in ChromaDB
  ├── classifier.py            # Identify which department owns the question
  ├── retriever.py             # Fetch relevant document chunks from ChromaDB
  ├── generator.py             # Generate professional answer from chunks
  ├── app.py                   # Gradio web UI
  └── requirements.txt

  ---
  Departments Covered
  
  ┌──────────────────────────────┬──────────────────────────────────────────┐
  │          Department          │            Example Questions             │
  ├──────────────────────────────┼──────────────────────────────────────────┤
  │ Legal                        │ Warranty, terms of service, privacy      │
  ├──────────────────────────────┼──────────────────────────────────────────┤
  │ Quality Control              │ ISO certification, defect returns        │
  ├──────────────────────────────┼──────────────────────────────────────────┤
  │ Supply Chain                 │ Delivery timelines, shipping regions     │
  ├──────────────────────────────┼──────────────────────────────────────────┤
  │ Environment & Sustainability │ RoHS, WEEE, recycling                    │
  ├──────────────────────────────┼──────────────────────────────────────────┤
  │ IT                           │ Firmware updates, software compatibility │
  ├──────────────────────────────┼──────────────────────────────────────────┤
  │ Energy & Water               │ Energy ratings, power consumption        │
  ├──────────────────────────────┼──────────────────────────────────────────┤
  │ After Sales                  │ Repairs, spare parts                     │
  ├──────────────────────────────┼──────────────────────────────────────────┤
  │ Product Safety               │ CE marking, recalls                      │
  ├──────────────────────────────┼──────────────────────────────────────────┤
  │ Finance                      │ Payment terms, invoices                  │
  ├──────────────────────────────┼──────────────────────────────────────────┤
  │ HR                           │ Careers, internships                     │
  ├──────────────────────────────┼──────────────────────────────────────────┤
  │ R&D                          │ Product roadmap, innovation              │
  ├──────────────────────────────┼──────────────────────────────────────────┤
  │ Marketing                    │ Product catalog, partner guidelines      │
  ├──────────────────────────────┼──────────────────────────────────────────┤
  │ Logistics                    │ Import/export, customs                   │
  ├──────────────────────────────┼──────────────────────────────────────────┤
  │ Customer Support             │ Escalation, SLA response times           │
  └──────────────────────────────┴──────────────────────────────────────────┘

  ---
  Using Real Company Documents
  
  1. Replace the .txt files in docs/<department>/ with your real documents
  2. Re-run ingestion:
  python3 ingest.py
  
  No code changes needed.

  ---
  Tech Stack
  
  - LLM: GPT-4o (OpenAI)
  - Embeddings: text-embedding-3-small (OpenAI)
  - Vector Database: ChromaDB (local)
  - UI: Gradio
  - Language: Python 3.10+
