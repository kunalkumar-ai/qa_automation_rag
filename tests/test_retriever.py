from unittest.mock import patch, MagicMock
from retriever import retrieve_chunks


def _mock_embedding_response(vector: list[float]) -> MagicMock:
    mock = MagicMock()
    mock.data[0].embedding = vector
    return mock


@patch("retriever.chromadb.PersistentClient")
@patch("retriever.client")
def test_retrieve_returns_matching_chunks(mock_openai, mock_chroma_cls):
    mock_openai.embeddings.create.return_value = _mock_embedding_response([0.1, 0.2, 0.3])
    mock_collection = MagicMock()
    mock_collection.query.return_value = {"documents": [["Warranty lasts 24 months.", "Claim via email."]]}
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

    chunks = retrieve_chunks("Something irrelevant", "legal")

    assert chunks == []


@patch("retriever.chromadb.PersistentClient")
@patch("retriever.client")
def test_retrieve_passes_department_filter_to_chroma(mock_openai, mock_chroma_cls):
    mock_openai.embeddings.create.return_value = _mock_embedding_response([0.1])
    mock_collection = MagicMock()
    mock_collection.query.return_value = {"documents": [["Some chunk."]]}
    mock_chroma_cls.return_value.get_collection.return_value = mock_collection

    retrieve_chunks("Any question", "quality_control")

    call_kwargs = mock_collection.query.call_args.kwargs
    assert call_kwargs["where"] == {"department": "quality_control"}
