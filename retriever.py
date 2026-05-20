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
