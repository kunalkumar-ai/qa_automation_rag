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
        if end == len(words):
            break
        start += chunk_size - overlap
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
