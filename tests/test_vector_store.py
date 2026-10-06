import json

import numpy as np
import pytest

from src.rag.chunker import Chunk
from src.rag.vector_store import (
    CHUNKS_FILE,
    EMBEDDINGS_FILE,
    METADATA_FILE,
    VectorStore,
    VectorStoreError,
    VectorStoreNotFoundError,
)

MODEL = "nomic-embed-text"  # only a label here; no model is called


def make_chunks():
    return [
        Chunk(chunk_id="lecture_04-page12-chunk01", source="lecture_04.pdf", page=12, text="Overfitting"),
        Chunk(chunk_id="lecture_05-page3-chunk01", source="lecture_05.pdf", page=3, text="Regularisation"),
        Chunk(chunk_id="notes-chunk01", source="notes.md", page=None, text="Neural networks"),
    ]


# Hand-made 3-D vectors pointing roughly along x, y, and z.
VECTORS = [[1.0, 0.1, 0.0], [0.0, 1.0, 0.1], [0.1, 0.0, 1.0]]


def make_store():
    store = VectorStore(embedding_model=MODEL)
    store.add(make_chunks(), VECTORS)
    return store


def test_search_returns_nearest_chunks_by_cosine_similarity():
    """Verify results are ordered by cosine similarity, with the exact-direction chunk first."""
    results = make_store().search([0.9, 0.2, 0.0], top_k=3)

    ids = [chunk.chunk_id for chunk, _ in results]
    scores = [score for _, score in results]
    assert ids[0] == "lecture_04-page12-chunk01"
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in scores)


def test_search_score_is_cosine_similarity_independent_of_vector_length():
    """Verify scores equal cosine similarity, so scaling a vector does not change its score."""
    store = VectorStore(embedding_model=MODEL)
    store.add(make_chunks()[:2], [[3.0, 4.0, 0.0], [0.0, 0.0, 10.0]])

    (best, best_score), (other, other_score) = store.search([6.0, 8.0, 0.0], top_k=2)

    assert best.chunk_id == "lecture_04-page12-chunk01"
    assert best_score == pytest.approx(1.0)
    assert other_score == pytest.approx(0.0)


def test_top_k_limits_results():
    """Verify only top_k results are returned."""
    assert len(make_store().search([1.0, 0.0, 0.0], top_k=2)) == 2


def test_top_k_larger_than_store_returns_all_chunks():
    """Verify asking for more results than chunks returns every chunk once."""
    results = make_store().search([1.0, 0.0, 0.0], top_k=10)

    assert len(results) == 3
    assert len({chunk.chunk_id for chunk, _ in results}) == 3


def test_invalid_top_k_is_rejected():
    """Verify a non-positive top_k raises ValueError."""
    with pytest.raises(ValueError):
        make_store().search([1.0, 0.0, 0.0], top_k=0)


def test_empty_store_search_returns_no_results():
    """Verify searching an empty store returns an empty list instead of failing."""
    store = VectorStore(embedding_model=MODEL)

    assert store.is_empty
    assert len(store) == 0
    assert store.search([1.0, 0.0, 0.0], top_k=4) == []


def test_search_preserves_chunk_metadata():
    """Verify returned chunks carry their original source, page, and text."""
    results = make_store().search([0.1, 0.0, 1.0], top_k=1)

    chunk, _ = results[0]
    assert chunk == Chunk(chunk_id="notes-chunk01", source="notes.md", page=None, text="Neural networks")


def test_query_with_wrong_dimension_or_zero_vector_is_rejected():
    """Verify mismatched-dimension and zero query vectors raise a clear error."""
    store = make_store()

    with pytest.raises(VectorStoreError, match="dimension 3"):
        store.search([1.0, 0.0], top_k=1)
    with pytest.raises(VectorStoreError, match="non-zero"):
        store.search([0.0, 0.0, 0.0], top_k=1)


def test_add_rejects_invalid_input():
    """Verify mismatched counts, dimensions, zero vectors, and duplicate IDs are rejected."""
    chunks = make_chunks()
    store = make_store()

    with pytest.raises(VectorStoreError, match="must match"):
        VectorStore(MODEL).add(chunks, VECTORS[:2])
    with pytest.raises(VectorStoreError, match="dimension"):
        store.add([Chunk("new-chunk01", "new.md", None, "x")], [[1.0, 0.0]])
    with pytest.raises(VectorStoreError, match="all-zero"):
        VectorStore(MODEL).add(chunks[:1], [[0.0, 0.0, 0.0]])
    with pytest.raises(VectorStoreError, match="unique"):
        store.add(chunks[:1], VECTORS[:1])
    assert len(store) == 3


def test_save_and_load_round_trip(tmp_path):
    """Verify a saved store writes all three files and loads back with identical search results."""
    store = make_store()
    store.save(tmp_path / "store")

    for name in (EMBEDDINGS_FILE, CHUNKS_FILE, METADATA_FILE):
        assert (tmp_path / "store" / name).is_file()

    loaded = VectorStore.load(tmp_path / "store")
    query = [0.2, 0.9, 0.1]
    assert loaded.embedding_model == MODEL
    assert loaded.embedding_dimension == 3
    assert loaded.chunks == store.chunks
    assert [(c.chunk_id, pytest.approx(s)) for c, s in loaded.search(query, 3)] == [
        (c.chunk_id, s) for c, s in store.search(query, 3)
    ]


def test_persisted_files_have_expected_layout(tmp_path):
    """Verify embeddings are in chunk order, chunks.json holds only chunk records, metadata holds index info."""
    store = VectorStore(embedding_model=MODEL, document_embedding_task="search_document")
    store.add(make_chunks(), VECTORS)
    store.save(tmp_path)

    embeddings = np.load(tmp_path / EMBEDDINGS_FILE)
    chunks = json.loads((tmp_path / CHUNKS_FILE).read_text(encoding="utf-8"))
    metadata = json.loads((tmp_path / METADATA_FILE).read_text(encoding="utf-8"))

    assert embeddings.dtype == np.float32
    np.testing.assert_allclose(embeddings, np.asarray(VECTORS, dtype=np.float32))

    assert isinstance(chunks, list)
    assert [c["chunk_id"] for c in chunks] == [c.chunk_id for c in make_chunks()]
    assert all(set(c) == {"chunk_id", "source", "page", "text"} for c in chunks)
    assert chunks[0] == {
        "chunk_id": "lecture_04-page12-chunk01",
        "source": "lecture_04.pdf",
        "page": 12,
        "text": "Overfitting",
    }

    assert not any("document_embedding_task" in c or "embedding_model" in c for c in chunks)

    assert metadata == {
        "format_version": 2,
        "embedding_model": MODEL,
        "embedding_dimension": 3,
        "document_embedding_task": "search_document",
        "chunk_count": 3,
    }


def test_empty_store_round_trip(tmp_path):
    """Verify an empty store can be saved and loaded, and stays empty."""
    VectorStore(embedding_model=MODEL).save(tmp_path)

    loaded = VectorStore.load(tmp_path)

    assert loaded.is_empty
    assert loaded.embedding_model == MODEL


def test_load_missing_store_raises_not_found(tmp_path):
    """Verify loading where no store exists raises VectorStoreNotFoundError."""
    with pytest.raises(VectorStoreNotFoundError):
        VectorStore.load(tmp_path / "never_built")
    with pytest.raises(VectorStoreNotFoundError):
        VectorStore.load(tmp_path)


def test_load_incomplete_store_raises_clear_error(tmp_path):
    """Verify a store with a missing file is reported as incomplete, naming the file."""
    make_store().save(tmp_path)
    (tmp_path / METADATA_FILE).unlink()

    with pytest.raises(VectorStoreError, match="incomplete.*metadata.json") as excinfo:
        VectorStore.load(tmp_path)

    assert not isinstance(excinfo.value, VectorStoreNotFoundError)


def test_load_rejects_row_count_mismatch(tmp_path):
    """Verify loading fails when embeddings.npy has a different row count than chunks.json."""
    make_store().save(tmp_path)
    np.save(tmp_path / EMBEDDINGS_FILE, np.asarray(VECTORS[:2], dtype=np.float32))

    with pytest.raises(VectorStoreError, match="2 rows but chunks.json has 3 chunks"):
        VectorStore.load(tmp_path)


def test_load_rejects_width_mismatch_with_metadata(tmp_path):
    """Verify loading fails when the embedding width differs from metadata embedding_dimension."""
    make_store().save(tmp_path)
    metadata = json.loads((tmp_path / METADATA_FILE).read_text(encoding="utf-8"))
    metadata["embedding_dimension"] = 768
    (tmp_path / METADATA_FILE).write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(VectorStoreError, match="width 3.*embedding_dimension 768"):
        VectorStore.load(tmp_path)


@pytest.mark.parametrize("missing_key", ["embedding_model", "embedding_dimension"])
def test_load_rejects_metadata_missing_required_key(tmp_path, missing_key):
    """Verify metadata.json must contain embedding_model and embedding_dimension."""
    make_store().save(tmp_path)
    metadata = json.loads((tmp_path / METADATA_FILE).read_text(encoding="utf-8"))
    del metadata[missing_key]
    (tmp_path / METADATA_FILE).write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(VectorStoreError, match=missing_key):
        VectorStore.load(tmp_path)


@pytest.mark.parametrize(
    "file_name, content",
    [
        (METADATA_FILE, "{not valid json"),
        (CHUNKS_FILE, "[{\"chunk_id\": \"a\"}"),
        (CHUNKS_FILE, "{\"not\": \"a list\"}"),
        (CHUNKS_FILE, json.dumps([{"chunk_id": "a", "source": "s", "page": 1}] * 3)),
        (EMBEDDINGS_FILE, "this is not a numpy file"),
    ],
)
def test_load_rejects_corrupt_files(tmp_path, file_name, content):
    """Verify unreadable or malformed files fail with VectorStoreError rather than a raw exception."""
    make_store().save(tmp_path)
    (tmp_path / file_name).write_text(content, encoding="utf-8")

    with pytest.raises(VectorStoreError):
        VectorStore.load(tmp_path)


def test_load_rejects_chunk_records_with_extra_index_metadata(tmp_path):
    """Verify chunks.json records may contain only chunk fields, not index-level metadata."""
    make_store().save(tmp_path)
    chunks = json.loads((tmp_path / CHUNKS_FILE).read_text(encoding="utf-8"))
    chunks[0]["embedding_model"] = MODEL
    (tmp_path / CHUNKS_FILE).write_text(json.dumps(chunks), encoding="utf-8")

    with pytest.raises(VectorStoreError, match="record 0"):
        VectorStore.load(tmp_path)


# ---------------------------------------------------------------- document embedding task marker


def save_task_store(path, task="search_document"):
    store = VectorStore(embedding_model=MODEL, document_embedding_task=task)
    store.add(make_chunks(), VECTORS)
    store.save(path)


def rewrite_metadata(path, change):
    metadata = json.loads((path / METADATA_FILE).read_text(encoding="utf-8"))
    change(metadata)
    (path / METADATA_FILE).write_text(json.dumps(metadata), encoding="utf-8")


def test_new_store_defaults_to_raw_task_and_is_recorded():
    """Verify the API stays backward compatible: omitting the task means raw text (None), recorded."""
    store = VectorStore(embedding_model=MODEL)

    assert store.document_embedding_task is None
    assert store.document_embedding_task_recorded is True


@pytest.mark.parametrize("task", ["search_document", None])
def test_save_load_preserves_document_embedding_task(tmp_path, task):
    """Verify the task marker (including an explicit null for raw text) survives a round trip."""
    save_task_store(tmp_path, task=task)

    metadata = json.loads((tmp_path / METADATA_FILE).read_text(encoding="utf-8"))
    loaded = VectorStore.load(tmp_path)

    assert "document_embedding_task" in metadata
    assert metadata["document_embedding_task"] == task
    assert loaded.document_embedding_task == task
    assert loaded.document_embedding_task_recorded is True


def test_task_marker_stores_logical_name_not_prefix(tmp_path):
    """Verify metadata holds 'search_document', never the literal model prefix 'search_document: '."""
    save_task_store(tmp_path)

    raw = (tmp_path / METADATA_FILE).read_text(encoding="utf-8")

    assert '"document_embedding_task": "search_document"' in raw
    assert "search_document: " not in raw


def test_compatible_index_reports_no_issue(tmp_path):
    """Verify an index built with the expected model and task is compatible."""
    save_task_store(tmp_path)

    assert VectorStore.load(tmp_path).compatibility_issue(MODEL, "search_document") is None


def test_legacy_index_without_task_marker_loads_but_is_never_compatible(tmp_path):
    """Verify an older index lacking document_embedding_task still loads, but is flagged, not treated as equivalent."""
    save_task_store(tmp_path)
    rewrite_metadata(tmp_path, lambda m: m.pop("document_embedding_task"))

    loaded = VectorStore.load(tmp_path)

    assert len(loaded) == 3
    assert loaded.document_embedding_task is None
    assert loaded.document_embedding_task_recorded is False
    for expected_task in ("search_document", None):
        issue = loaded.compatibility_issue(MODEL, expected_task)
        assert issue is not None and "does not record a document_embedding_task" in issue


def test_incompatible_task_marker_is_reported(tmp_path):
    """Verify an index built with a different task (or raw text) is reported as incompatible."""
    save_task_store(tmp_path, task=None)

    issue = VectorStore.load(tmp_path).compatibility_issue(MODEL, "search_document")

    assert issue is not None
    assert "document embedding task None" in issue and "'search_document' is expected" in issue


def test_embedding_model_mismatch_is_reported_first(tmp_path):
    """Verify an index built with another embedding model is reported by model name."""
    save_task_store(tmp_path)

    issue = VectorStore.load(tmp_path).compatibility_issue("mxbai-embed-large", "search_document")

    assert issue is not None and "'nomic-embed-text'" in issue and "'mxbai-embed-large'" in issue


@pytest.mark.parametrize("bad_value", ["", 5, ["search_document"]])
def test_load_rejects_malformed_task_marker(tmp_path, bad_value):
    """Verify a document_embedding_task that is not a non-empty string or null fails clearly."""
    save_task_store(tmp_path)
    rewrite_metadata(tmp_path, lambda m: m.update(document_embedding_task=bad_value))

    with pytest.raises(VectorStoreError, match="document_embedding_task"):
        VectorStore.load(tmp_path)


def test_load_rejects_non_finite_embeddings(tmp_path):
    """Verify a matrix containing NaN values is rejected on load."""
    make_store().save(tmp_path)
    corrupted = np.asarray(VECTORS, dtype=np.float32)
    corrupted[1, 0] = np.nan
    np.save(tmp_path / EMBEDDINGS_FILE, corrupted)

    with pytest.raises(VectorStoreError, match="non-finite"):
        VectorStore.load(tmp_path)
