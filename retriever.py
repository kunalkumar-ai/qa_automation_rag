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


def _dense_search(question: str) -> tuple[list[str], dict, dict, dict]:
    """Embed the question and find the top-K most similar child chunks in ChromaDB.

    Returns: (ordered chunk IDs, texts by ID, metadatas by ID, scores by ID)
    """
    q_vec = client.embeddings.create(model=EMBEDDING_MODEL, input=[question]).data[0].embedding

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection("tesla")
    result = collection.query(
        query_embeddings=[q_vec],
        n_results=TOP_K_DENSE,
        include=["documents", "metadatas", "distances"],
    )

    ids = result["ids"][0]
    texts = {cid: doc for cid, doc in zip(ids, result["documents"][0])}
    metas = {cid: meta for cid, meta in zip(ids, result["metadatas"][0])}
    scores = {cid: float(dist) for cid, dist in zip(ids, result["distances"][0])}
    return ids, texts, metas, scores


def _bm25_search(question: str) -> tuple[list[str], dict, dict, dict]:
    """Score all child chunks by keyword match and return the top-K results.

    Returns: (ordered chunk IDs, texts by ID, metadatas by ID, scores by ID)
    """
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)

    tokenized_query = question.lower().split()
    raw_scores = bm25_data["bm25"].get_scores(tokenized_query)

    ranked = sorted(
        zip(bm25_data["chunk_ids"], raw_scores),
        key=lambda x: x[1],
        reverse=True,
    )[:TOP_K_BM25]

    ids = [cid for cid, _ in ranked]
    texts = {cid: bm25_data["chunk_texts"][cid] for cid, _ in ranked}
    metas = {cid: bm25_data["chunk_metas"][cid] for cid, _ in ranked}
    scores = {cid: float(score) for cid, score in ranked}
    return ids, texts, metas, scores


def rrf_merge(dense_ids: list[str], bm25_ids: list[str], k: int = RRF_K) -> list[str]:
    """Merge two ranked lists using Reciprocal Rank Fusion.

    Each chunk scores 1/(k + rank) per list it appears in.
    Chunks appearing in both lists accumulate scores from both.
    Returns a single list sorted by combined score, highest first.
    """
    scores: dict[str, float] = {}
    for rank, chunk_id in enumerate(dense_ids):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, chunk_id in enumerate(bm25_ids):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


def retrieve(question: str) -> dict:
    """Full retrieval pipeline: dual search → RRF merge → rerank → parent lookup.

    Returns a dict with top child chunks (with scores), parent section texts,
    and candidate counts for logging.
    """
    # Step 1: run both searches
    dense_ids, dense_texts, dense_metas, dense_scores = _dense_search(question)
    bm25_ids, bm25_texts, bm25_metas, bm25_scores = _bm25_search(question)

    # Step 2: merge all texts and metas into one lookup (dense takes priority on overlap)
    all_texts = {**bm25_texts, **dense_texts}
    all_metas = {**bm25_metas, **dense_metas}

    # Step 3: RRF merge → single ranked list of up to 40 candidates
    merged_ids = rrf_merge(dense_ids, bm25_ids)
    valid_ids = [cid for cid in merged_ids if cid in all_texts]

    # Step 4: rerank — cross-encoder reads question + each chunk together
    pairs = [[question, all_texts[cid]] for cid in valid_ids]
    rerank_scores_list = reranker.predict(pairs)
    reranked = sorted(zip(valid_ids, rerank_scores_list), key=lambda x: x[1], reverse=True)
    top_ids = [cid for cid, _ in reranked[:TOP_K_RERANK]]
    rerank_score_map = {cid: float(score) for cid, score in reranked}

    # Step 5: parent lookup — swap each child for its full parent section
    with open(PARENTS_PATH, "rb") as f:
        parents: dict[str, str] = pickle.load(f)

    parent_texts: list[str] = []
    seen: set[str] = set()
    for cid in top_ids:
        pid = all_metas[cid]["parent_id"]
        if pid not in seen and pid in parents:
            parent_texts.append(parents[pid])
            seen.add(pid)

    return {
        "top_child_chunks": [
            {
                "chunk_id": cid,
                "text": all_texts.get(cid, ""),
                "section_name": all_metas.get(cid, {}).get("section_name", ""),
                "dense_score": dense_scores.get(cid, 0.0),
                "bm25_score": bm25_scores.get(cid, 0.0),
                "reranker_score": rerank_score_map.get(cid, 0.0),
            }
            for cid in top_ids
        ],
        "parent_texts": parent_texts,
        "all_candidates": merged_ids,
        "dense_ids": dense_ids,
        "bm25_ids": bm25_ids,
    }
