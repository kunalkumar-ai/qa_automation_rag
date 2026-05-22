from retriever import rrf_merge


def test_item_in_both_lists_ranks_first():
    """A chunk found by both dense and BM25 should outscore chunks found by only one."""
    dense = ["a", "b", "c"]
    bm25  = ["b", "d", "e"]
    result = rrf_merge(dense, bm25)
    assert result[0] == "b"


def test_includes_all_unique_ids():
    dense = ["a", "b"]
    bm25  = ["c", "d"]
    result = rrf_merge(dense, bm25)
    assert set(result) == {"a", "b", "c", "d"}


def test_higher_rank_beats_lower_rank():
    """First item in a list should score higher than the last item."""
    result = rrf_merge(["first", "second", "third"], [])
    assert result[0] == "first"
    assert result[-1] == "third"


def test_handles_empty_bm25():
    result = rrf_merge(["a", "b", "c"], [])
    assert result == ["a", "b", "c"]


def test_handles_empty_dense():
    result = rrf_merge([], ["x", "y", "z"])
    assert result == ["x", "y", "z"]


def test_handles_both_empty():
    assert rrf_merge([], []) == []


def test_no_duplicates_in_output():
    dense = ["a", "b", "c"]
    bm25  = ["a", "b", "d"]
    result = rrf_merge(dense, bm25)
    assert len(result) == len(set(result))
