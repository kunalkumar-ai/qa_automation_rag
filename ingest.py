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
    """Send a batch of texts to OpenAI and return their embedding vectors."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def ingest(doc_path: str = TESLA_DOC_PATH) -> None:
    """Chunk the document, embed children into ChromaDB, save parents and BM25 index."""
    print(f"Reading {doc_path}...")
    with open(doc_path) as f:
        text = f.read()

    all_chunks = build_chunks(text)
    children = [c for c in all_chunks if c.chunk_type == "child"]
    parents = {c.chunk_id: c.text for c in all_chunks if c.chunk_type == "parent"}
    print(f"Built {len(parents)} parent sections, {len(children)} child paragraphs")

    # ChromaDB — store child chunks with their embedding vectors
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

    # Parent store — save as dict so retriever can fetch by ID without vector search
    with open(PARENTS_PATH, "wb") as f:
        pickle.dump(parents, f)
    print(f"Parent store saved to {PARENTS_PATH}")

    # BM25 index — keyword search on same child chunks
    tokenized = [c.text.lower().split() for c in children]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "chunk_ids": [c.chunk_id for c in children],
            "chunk_texts": {c.chunk_id: c.text for c in children},
            "chunk_metas": {
                c.chunk_id: {"parent_id": c.parent_id, "section_name": c.section_name}
                for c in children
            },
        }, f)
    print(f"BM25 index saved to {BM25_INDEX_PATH}")
    print("Ingestion complete.")


if __name__ == "__main__":
    ingest()
