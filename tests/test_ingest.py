from ingest import chunk_text


def test_short_document_returns_single_chunk():
    text = " ".join(["word"] * 100)
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_document_returns_multiple_chunks():
    text = " ".join([str(i) for i in range(1000)])
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 3


def test_overlap_shares_words_between_adjacent_chunks():
    text = " ".join([str(i) for i in range(600)])
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    last_words_of_first = set(chunks[0].split()[-50:])
    first_words_of_second = set(chunks[1].split()[:50])
    assert last_words_of_first == first_words_of_second


def test_empty_string_returns_empty_list():
    chunks = chunk_text("", chunk_size=500, overlap=50)
    assert chunks == []
