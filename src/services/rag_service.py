import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.config import config
from src.rag.chunker import Chunk
from src.rag.embeddings import SEARCH_DOCUMENT, SEARCH_QUERY, Embedder, OllamaEmbedder
from src.rag.vector_store import VectorStore, VectorStoreError, VectorStoreNotFoundError


class RetrievalStatus(str, Enum):
    """Outcome of a retrieval request. Only OK carries usable context."""

    OK = "ok"
    EMPTY_INDEX = "empty_index"  # no index has been built, or it contains no chunks
    NO_RELEVANT_CONTEXT = "no_relevant_context"  # nothing reached RAG_MIN_SCORE
    INDEX_MISMATCH = "index_mismatch"  # built with another model/task, legacy, or wrong dimension
    INDEX_ERROR = "index_error"  # index files exist but are corrupt or inconsistent


@dataclass
class ScoredChunk:
    """A retrieved chunk with its cosine similarity to the query."""

    chunk: Chunk
    score: float


@dataclass
class RetrievalResult:
    """Chunks retrieved for a query (only when status is OK) and a reason otherwise."""

    status: RetrievalStatus
    chunks: list[ScoredChunk] = field(default_factory=list)
    detail: Optional[str] = None  # technical explanation for non-OK statuses

    @property
    def ok(self) -> bool:
        return self.status is RetrievalStatus.OK


class RAGService:
    """
    Retrieval orchestration: embeds a validated question, searches the local
    index, and returns the relevant chunks with their metadata.

    The index is loaded lazily on the first request and kept for the lifetime of
    the service, so a rebuilt index is picked up only after the application is
    restarted. This service never calls the generation model.

    Embedding failures (e.g. Ollama unreachable) propagate as ModelClientError so
    the application service can translate them into user-facing messages.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        store: Optional[VectorStore] = None,
        store_path: Optional[str] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        expected_document_task: Optional[str] = SEARCH_DOCUMENT,
    ):
        self.store_path = store_path or config.vector_store_path
        self.top_k = config.rag_top_k if top_k is None else top_k
        self.min_score = config.rag_min_score if min_score is None else min_score
        self.expected_document_task = expected_document_task

        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}.")
        if not math.isfinite(self.min_score) or not -1.0 <= self.min_score <= 1.0:
            raise ValueError(f"min_score must be a cosine similarity in [-1, 1], got {self.min_score}.")

        self._embedder = embedder
        self._store = store
        # Cached outcome of the one-time load when it did not produce a usable store.
        self._load_failure: Optional[RetrievalResult] = None
        self._load_attempted = store is not None

    def _get_embedder(self) -> Embedder:
        """Returns the active embedder, creating the configured Ollama embedder if none was injected."""
        if self._embedder is None:
            self._embedder = OllamaEmbedder(
                base_url=config.ollama_base_url, model_name=config.embedding_model
            )
        return self._embedder

    def _get_store(self) -> Optional[VectorStore]:
        """Loads the index once; on failure records a non-OK result and returns None."""
        if not self._load_attempted:
            self._load_attempted = True
            try:
                self._store = VectorStore.load(self.store_path)
            except VectorStoreNotFoundError as err:
                self._load_failure = RetrievalResult(RetrievalStatus.EMPTY_INDEX, detail=str(err))
            except VectorStoreError as err:
                self._load_failure = RetrievalResult(RetrievalStatus.INDEX_ERROR, detail=str(err))
        return self._store

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Returns the top-k chunks whose similarity to `query` is at least min_score.

        Statuses: OK (chunks returned), EMPTY_INDEX, INDEX_ERROR, INDEX_MISMATCH,
        or NO_RELEVANT_CONTEXT. Raises ValueError for an empty query and
        ModelClientError if the query cannot be embedded.
        """
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        store = self._get_store()
        if store is None:
            return self._load_failure

        embedder = self._get_embedder()
        issue = store.compatibility_issue(embedder.model_name, self.expected_document_task)
        if issue:
            return RetrievalResult(RetrievalStatus.INDEX_MISMATCH, detail=issue)
        if store.is_empty:
            return RetrievalResult(RetrievalStatus.EMPTY_INDEX, detail="The index contains no chunks.")

        query_vector = embedder.embed([query.strip()], task=SEARCH_QUERY)[0]
        if len(query_vector) != store.embedding_dimension:
            return RetrievalResult(
                RetrievalStatus.INDEX_MISMATCH,
                detail=(
                    f"The query embedding has dimension {len(query_vector)}, but the index "
                    f"was built with dimension {store.embedding_dimension}."
                ),
            )

        try:
            candidates = store.search(query_vector, top_k=self.top_k)
        except VectorStoreError as err:
            return RetrievalResult(RetrievalStatus.INDEX_MISMATCH, detail=str(err))

        relevant = [ScoredChunk(chunk, score) for chunk, score in candidates if score >= self.min_score]
        if not relevant:
            # Enough precision that a score just below the threshold never displays as reaching it.
            best = f"{candidates[0][1]:.6f}" if candidates else "n/a"
            return RetrievalResult(
                RetrievalStatus.NO_RELEVANT_CONTEXT,
                detail=f"No chunk reached the minimum score {self.min_score} (best score {best}).",
            )
        return RetrievalResult(RetrievalStatus.OK, chunks=relevant)
