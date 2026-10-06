import json
from unittest.mock import patch

import ollama
import pytest

import src.services.rag_service as rag_service_module
from src.models.model_client import OllamaConnectionError, OllamaModelClient
from src.rag.chunker import Chunk
from src.rag.embeddings import SEARCH_DOCUMENT, SEARCH_QUERY, OllamaEmbedder
from src.rag.ingest import build_index
from src.rag.vector_store import METADATA_FILE, VectorStore
from src.services.rag_service import RAGService, RetrievalResult, RetrievalStatus, ScoredChunk

MODEL = "fake-embed"
TOPICS = ["overfitting", "evaluation", "network", "football"]


class KeywordEmbedder:
    """Deterministic embedder: one dimension per topic keyword plus a small constant dimension."""

    def __init__(self, model_name=MODEL, dimension=None):
        self.model_name = model_name
        self.dimension = dimension
        self.calls = []

    def embed(self, texts, task=None):
        self.calls.append((list(texts), task))
        vectors = [[float(text.lower().count(t)) for t in TOPICS] + [0.05] for text in texts]
        if self.dimension is not None:
            vectors = [v[: self.dimension] for v in vectors]
        return vectors


CHUNKS = [
    Chunk("lecture_04-page2-chunk01", "lecture_04.pdf", 2, "Overfitting happens when a model memorises noise."),
    Chunk("lecture_04-page3-chunk01", "lecture_04.pdf", 3, "Regularisation reduces overfitting. Overfitting again."),
    Chunk("lecture_05-page1-chunk01", "lecture_05.pdf", 1, "Model evaluation uses a held-out test set."),
    Chunk("notes-chunk01", "notes.md", None, "A neural network has layers."),
]


def make_store(model=MODEL, task=SEARCH_DOCUMENT):
    store = VectorStore(embedding_model=model, document_embedding_task=task)
    store.add(CHUNKS, KeywordEmbedder().embed([c.text for c in CHUNKS]))
    return store


def make_service(store=None, embedder=None, **kwargs):
    return RAGService(
        embedder=embedder or KeywordEmbedder(),
        store=store if store is not None else make_store(),
        top_k=kwargs.pop("top_k", 4),
        min_score=kwargs.pop("min_score", 0.5),
        **kwargs,
    )


# ---------------------------------------------------------------- successful retrieval


def test_related_query_returns_chunks_with_metadata():
    """Verify a related question returns OK with the matching chunks, scores, and source metadata."""
    result = make_service().retrieve("What is overfitting?")

    assert result.status is RetrievalStatus.OK and result.ok
    assert result.detail is None
    ids = [item.chunk.chunk_id for item in result.chunks]
    assert set(ids) == {"lecture_04-page2-chunk01", "lecture_04-page3-chunk01"}
    first = result.chunks[0]
    assert isinstance(first, ScoredChunk)
    assert first.chunk.source == "lecture_04.pdf" and first.chunk.page in (2, 3)
    assert first.chunk.text.startswith(("Overfitting", "Regularisation"))
    assert [item.score for item in result.chunks] == sorted((item.score for item in result.chunks), reverse=True)


def test_query_is_embedded_with_search_query_task():
    """Verify the question is embedded once, stripped, with the SEARCH_QUERY retrieval role."""
    embedder = KeywordEmbedder()

    make_service(embedder=embedder).retrieve("  What is overfitting?  ")

    assert embedder.calls == [(["What is overfitting?"], SEARCH_QUERY)]


def test_top_k_is_respected():
    """Verify no more than top_k chunks are returned even when more pass the threshold."""
    result = make_service(top_k=1).retrieve("overfitting")

    assert result.ok
    assert len(result.chunks) == 1


def test_relevance_threshold_filters_weak_chunks():
    """Verify only chunks with score >= min_score are returned."""
    low = make_service(min_score=0.0, top_k=4).retrieve("overfitting and evaluation")
    high = make_service(min_score=0.6, top_k=4).retrieve("overfitting and evaluation")

    assert len(low.chunks) == 4  # every chunk passes a zero threshold
    assert 0 < len(high.chunks) < 4
    assert all(item.score >= 0.6 for item in high.chunks)


def test_threshold_boundary_is_inclusive():
    """Verify a chunk whose score equals min_score exactly is kept."""
    store = VectorStore(embedding_model=MODEL, document_embedding_task=SEARCH_DOCUMENT)
    store.add([CHUNKS[0]], [[1.0, 0.0, 0.0, 0.0, 0.0]])

    class FixedEmbedder(KeywordEmbedder):
        def embed(self, texts, task=None):
            return [[0.6, 0.8, 0.0, 0.0, 0.0]]  # cosine with [1, 0, ...] is exactly 0.6

    result = make_service(store=store, embedder=FixedEmbedder(), min_score=0.6).retrieve("q")

    assert result.ok and result.chunks[0].score == pytest.approx(0.6)


def test_defaults_come_from_config(monkeypatch):
    """Verify top_k, min_score, and store path default to RAG_TOP_K, RAG_MIN_SCORE, VECTOR_STORE_PATH."""
    monkeypatch.setattr(rag_service_module.config, "rag_top_k", 3)
    monkeypatch.setattr(rag_service_module.config, "rag_min_score", 0.42)
    monkeypatch.setattr(rag_service_module.config, "vector_store_path", "some/index")

    service = RAGService(embedder=KeywordEmbedder())

    assert (service.top_k, service.min_score, service.store_path) == (3, 0.42, "some/index")


def test_default_embedder_uses_configured_embedding_model(monkeypatch):
    """Verify the service builds an OllamaEmbedder for the configured embedding model and host."""
    monkeypatch.setattr(rag_service_module.config, "embedding_model", "nomic-embed-text")
    monkeypatch.setattr(rag_service_module.config, "ollama_base_url", "http://ollama:1")

    with patch("ollama.Client") as client_class:
        embedder = RAGService(store=make_store())._get_embedder()

    assert isinstance(embedder, OllamaEmbedder)
    assert embedder.model_name == "nomic-embed-text"
    client_class.assert_called_once_with(host="http://ollama:1")


@pytest.mark.parametrize("kwargs", [{"top_k": 0}, {"min_score": 1.5}, {"min_score": float("nan")}])
def test_invalid_settings_are_rejected(kwargs):
    """Verify a non-positive top_k or a min_score outside [-1, 1] raises ValueError."""
    with pytest.raises(ValueError):
        RAGService(embedder=KeywordEmbedder(), store=make_store(), **{"top_k": 4, "min_score": 0.5, **kwargs})


def test_empty_query_is_rejected():
    """Verify an empty question is a programming error (AIService validates input first)."""
    with pytest.raises(ValueError):
        make_service().retrieve("   ")


# ---------------------------------------------------------------- non-OK statuses


def test_unrelated_query_returns_no_relevant_context():
    """Verify a question unrelated to every chunk returns NO_RELEVANT_CONTEXT with no chunks."""
    result = make_service().retrieve("Who won the football match?")

    assert result.status is RetrievalStatus.NO_RELEVANT_CONTEXT
    assert result.chunks == []
    assert "minimum score 0.5" in result.detail and "best score" in result.detail


def test_no_relevant_context_detail_does_not_round_best_score_up_to_threshold():
    """Verify a best score just below min_score is reported below it (e.g. 0.499824, not 0.500)."""
    store = VectorStore(embedding_model=MODEL, document_embedding_task=SEARCH_DOCUMENT)
    store.add([CHUNKS[0]], [[1.0, 0.0, 0.0, 0.0, 0.0]])
    below = 0.499824

    class FixedEmbedder(KeywordEmbedder):
        def embed(self, texts, task=None):
            return [[below, (1 - below**2) ** 0.5, 0.0, 0.0, 0.0]]  # cosine with [1, 0, ...] == below

    result = make_service(store=store, embedder=FixedEmbedder(), min_score=0.5).retrieve("q")

    assert result.status is RetrievalStatus.NO_RELEVANT_CONTEXT
    assert "best score 0.499824" in result.detail


def test_missing_index_returns_empty_index_without_embedding(tmp_path):
    """Verify a missing index returns EMPTY_INDEX and the question is never embedded."""
    embedder = KeywordEmbedder()
    service = RAGService(embedder=embedder, store_path=str(tmp_path / "missing"), top_k=4, min_score=0.5)

    result = service.retrieve("What is overfitting?")

    assert result.status is RetrievalStatus.EMPTY_INDEX
    assert result.chunks == []
    assert embedder.calls == []


def test_index_without_chunks_returns_empty_index(tmp_path):
    """Verify a saved index that contains no chunks is also reported as EMPTY_INDEX."""
    VectorStore(embedding_model=MODEL, document_embedding_task=SEARCH_DOCUMENT).save(tmp_path)

    result = RAGService(embedder=KeywordEmbedder(), store_path=str(tmp_path), top_k=4, min_score=0.5).retrieve("q")

    assert result.status is RetrievalStatus.EMPTY_INDEX


def test_embedding_model_mismatch_returns_index_mismatch():
    """Verify an index built with a different embedding model returns INDEX_MISMATCH without embedding."""
    embedder = KeywordEmbedder(model_name="nomic-embed-text")

    result = make_service(store=make_store(model="mxbai-embed-large"), embedder=embedder).retrieve("overfitting")

    assert result.status is RetrievalStatus.INDEX_MISMATCH
    assert "mxbai-embed-large" in result.detail and "nomic-embed-text" in result.detail
    assert embedder.calls == []


def test_document_task_mismatch_returns_index_mismatch():
    """Verify an index built from raw (un-prefixed) documents is not searched with task-aware queries."""
    embedder = KeywordEmbedder()

    result = make_service(store=make_store(task=None), embedder=embedder).retrieve("overfitting")

    assert result.status is RetrievalStatus.INDEX_MISMATCH
    assert "document embedding task" in result.detail
    assert embedder.calls == []


def test_legacy_index_without_task_marker_returns_index_mismatch(tmp_path):
    """Verify an index whose metadata predates document_embedding_task is reported as INDEX_MISMATCH."""
    make_store().save(tmp_path)
    metadata = json.loads((tmp_path / METADATA_FILE).read_text(encoding="utf-8"))
    del metadata["document_embedding_task"]
    (tmp_path / METADATA_FILE).write_text(json.dumps(metadata), encoding="utf-8")

    result = RAGService(embedder=KeywordEmbedder(), store_path=str(tmp_path), top_k=4, min_score=0.5).retrieve("overfitting")

    assert result.status is RetrievalStatus.INDEX_MISMATCH
    assert "does not record a document_embedding_task" in result.detail


def test_query_embedding_dimension_mismatch_returns_index_mismatch():
    """Verify a query vector of the wrong size gives a clear INDEX_MISMATCH instead of an exception."""
    result = make_service(embedder=KeywordEmbedder(dimension=3)).retrieve("overfitting")

    assert result.status is RetrievalStatus.INDEX_MISMATCH
    assert "dimension 3" in result.detail and "dimension 5" in result.detail


def test_corrupt_index_returns_index_error(tmp_path):
    """Verify corrupt index files produce a non-OK INDEX_ERROR result rather than an exception."""
    make_store().save(tmp_path)
    (tmp_path / METADATA_FILE).write_text("{not json", encoding="utf-8")

    result = RAGService(embedder=KeywordEmbedder(), store_path=str(tmp_path), top_k=4, min_score=0.5).retrieve("overfitting")

    assert result.status is RetrievalStatus.INDEX_ERROR
    assert result.chunks == [] and not result.ok
    assert METADATA_FILE in result.detail


def test_embedding_failure_propagates_for_application_layer():
    """Verify an embedding failure is not swallowed, so AIService can show a friendly message."""
    class FailingEmbedder(KeywordEmbedder):
        def embed(self, texts, task=None):
            raise OllamaConnectionError("Failed to connect to Ollama service at http://x.")

    with pytest.raises(OllamaConnectionError):
        make_service(embedder=FailingEmbedder()).retrieve("overfitting")


# ---------------------------------------------------------------- lazy loading and integration


def test_index_is_loaded_lazily_and_only_once(tmp_path):
    """Verify nothing is read at construction, the index loads on first use, and is then reused."""
    make_store().save(tmp_path)

    with patch.object(VectorStore, "load", wraps=VectorStore.load) as load:
        service = RAGService(embedder=KeywordEmbedder(), store_path=str(tmp_path), top_k=4, min_score=0.5)
        assert load.call_count == 0

        service.retrieve("overfitting")
        service.retrieve("evaluation")

    assert load.call_count == 1


def test_missing_index_result_is_kept_until_restart(tmp_path):
    """Verify an index built after startup is not picked up without a restart (no hot reload)."""
    service = RAGService(embedder=KeywordEmbedder(), store_path=str(tmp_path), top_k=4, min_score=0.5)
    assert service.retrieve("overfitting").status is RetrievalStatus.EMPTY_INDEX

    make_store().save(tmp_path)

    assert service.retrieve("overfitting").status is RetrievalStatus.EMPTY_INDEX
    fresh = RAGService(embedder=KeywordEmbedder(), store_path=str(tmp_path), top_k=4, min_score=0.5)
    assert fresh.retrieve("overfitting").ok


def test_end_to_end_with_ingested_index_and_nomic_prefixes(tmp_path):
    """Verify an index built by ingest is retrieved with 'search_query: ' queries via the real
    OllamaEmbedder (client patched), and no chat/generation call is made."""
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "lecture.md").write_text("Overfitting happens when a model memorises noise.", encoding="utf-8")
    (docs / "other.md").write_text("Model evaluation uses a held-out test set.", encoding="utf-8")

    def fake_embed(model, input):
        # Strip the Nomic role prefix before scoring so both roles share one vector space.
        texts = [t.split(": ", 1)[1] for t in input]
        return ollama.EmbedResponse(embeddings=KeywordEmbedder().embed(texts))

    with patch("ollama.Client") as client_class:
        client = client_class.return_value
        client.embed.side_effect = fake_embed
        embedder = OllamaEmbedder(model_name="nomic-embed-text")
        build_index(docs, tmp_path / "store", embedder, size=1000, overlap=200)
        client.embed.reset_mock()

        result = RAGService(embedder=embedder, store_path=str(tmp_path / "store"), top_k=4, min_score=0.5).retrieve(
            "What is overfitting?"
        )

    assert result.ok
    assert [item.chunk.source for item in result.chunks] == ["lecture.md"]
    assert client.embed.call_args.kwargs["input"] == ["search_query: What is overfitting?"]
    client.chat.assert_not_called()


def test_no_generation_model_is_used():
    """Verify retrieval never touches the generation client, for any status."""
    with patch.object(OllamaModelClient, "generate") as generate, patch.object(OllamaModelClient, "__init__") as init:
        make_service().retrieve("overfitting")
        make_service().retrieve("football")
        make_service(store=make_store(task=None)).retrieve("overfitting")

    generate.assert_not_called()
    init.assert_not_called()
    assert "OllamaModelClient" not in vars(rag_service_module)


def test_retrieval_result_defaults():
    """Verify RetrievalResult uses independent empty chunk lists and reports ok only for OK."""
    first, second = RetrievalResult(RetrievalStatus.EMPTY_INDEX), RetrievalResult(RetrievalStatus.EMPTY_INDEX)
    first.chunks.append(ScoredChunk(CHUNKS[0], 1.0))

    assert second.chunks == []
    assert not first.ok and RetrievalResult(RetrievalStatus.OK).ok
    assert RetrievalStatus.NO_RELEVANT_CONTEXT == "no_relevant_context"
