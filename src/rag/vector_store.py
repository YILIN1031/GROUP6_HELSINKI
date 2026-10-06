import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src.rag.chunker import Chunk

EMBEDDINGS_FILE = "embeddings.npy"
CHUNKS_FILE = "chunks.json"
METADATA_FILE = "metadata.json"
# Version 2 added document_embedding_task to metadata.json.
FORMAT_VERSION = 2

_CHUNK_FIELDS = {"chunk_id": str, "source": str, "page": (int, type(None)), "text": str}


class VectorStoreError(Exception):
    """Raised when a persisted vector store is corrupt or inconsistent, or input is invalid."""


class VectorStoreNotFoundError(VectorStoreError):
    """Raised when no persisted vector store exists at the given location."""


class VectorStore:
    """
    A small local vector store: chunk embeddings in a NumPy matrix plus chunk
    records, searched by cosine similarity.

    Persisted layout inside the store directory:
      - embeddings.npy : float32 matrix, one row per chunk, in chunk order
      - chunks.json    : list of chunk records (chunk_id, source, page, text)
      - metadata.json  : index-level metadata (embedding_model, embedding_dimension,
                         document_embedding_task, ...)

    `document_embedding_task` is the logical retrieval task the chunks were
    embedded with (e.g. "search_document"), or None for raw text. The store only
    records and compares it; how a task affects embedding lives in embeddings.py.
    """

    def __init__(
        self,
        embedding_model: str,
        embedding_dimension: Optional[int] = None,
        document_embedding_task: Optional[str] = None,
    ):
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.document_embedding_task = document_embedding_task
        # False only for a loaded index whose metadata predates the task marker.
        self._document_embedding_task_recorded = True
        self._chunks: list[Chunk] = []
        self._embeddings = np.zeros((0, embedding_dimension or 0), dtype=np.float32)
        self._normalized = self._embeddings

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def document_embedding_task_recorded(self) -> bool:
        """False if this index was loaded from metadata without a document_embedding_task field."""
        return self._document_embedding_task_recorded

    def compatibility_issue(
        self, embedding_model: str, document_embedding_task: Optional[str]
    ) -> Optional[str]:
        """
        Returns a human-readable reason why this index cannot be searched with
        queries embedded by `embedding_model` for an index built with
        `document_embedding_task`, or None if it is compatible.

        An index without a recorded task marker is never treated as compatible.
        """
        if self.embedding_model != embedding_model:
            return (
                f"the index was built with embedding model '{self.embedding_model}', "
                f"but '{embedding_model}' is configured"
            )
        if not self._document_embedding_task_recorded:
            return (
                "the index metadata does not record a document_embedding_task "
                "(it was built before task-aware ingestion)"
            )
        if self.document_embedding_task != document_embedding_task:
            return (
                f"the index was built with document embedding task "
                f"{self.document_embedding_task!r}, but {document_embedding_task!r} is expected"
            )
        return None

    @property
    def is_empty(self) -> bool:
        return not self._chunks

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Adds chunks with their embedding vectors (one vector per chunk, same order)."""
        if len(chunks) != len(vectors):
            raise VectorStoreError(
                f"Got {len(chunks)} chunks but {len(vectors)} vectors; they must match one-to-one."
            )
        if not chunks:
            return

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise VectorStoreError("Vectors must all have the same length.")
        if self.embedding_dimension is None:
            self.embedding_dimension = matrix.shape[1]
            self._embeddings = np.zeros((0, self.embedding_dimension), dtype=np.float32)
        if matrix.shape[1] != self.embedding_dimension:
            raise VectorStoreError(
                f"Vectors have dimension {matrix.shape[1]}, "
                f"but this store uses dimension {self.embedding_dimension}."
            )
        _check_usable_rows(matrix, "Vectors")

        existing_ids = {chunk.chunk_id for chunk in self._chunks}
        new_ids = [chunk.chunk_id for chunk in chunks]
        if len(set(new_ids)) != len(new_ids) or existing_ids.intersection(new_ids):
            raise VectorStoreError("Chunk IDs must be unique within the store.")

        self._chunks.extend(chunks)
        self._embeddings = np.vstack([self._embeddings, matrix])
        self._normalized = _normalize_rows(self._embeddings)

    def search(self, query_vector: Sequence[float], top_k: int) -> list[tuple[Chunk, float]]:
        """
        Returns up to `top_k` (chunk, cosine similarity) pairs, most similar first.
        An empty store returns an empty list.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}.")
        if self.is_empty:
            return []

        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self.embedding_dimension:
            raise VectorStoreError(
                f"Query vector has shape {query.shape}, "
                f"but this store uses dimension {self.embedding_dimension}."
            )
        norm = np.linalg.norm(query)
        if not np.isfinite(norm) or norm == 0:
            raise VectorStoreError("Query vector must be finite and non-zero.")

        scores = self._normalized @ (query / norm)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [(self._chunks[i], float(scores[i])) for i in order]

    def save(self, directory: str | Path) -> None:
        """Writes embeddings.npy, chunks.json, and metadata.json into `directory`."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        np.save(path / EMBEDDINGS_FILE, self._embeddings, allow_pickle=False)
        with open(path / CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump([asdict(chunk) for chunk in self._chunks], f, ensure_ascii=False, indent=2)
        metadata = {
            "format_version": FORMAT_VERSION,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "document_embedding_task": self.document_embedding_task,
            "chunk_count": len(self._chunks),
        }
        with open(path / METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        """
        Loads a store saved by `save()` and validates that its files are consistent.

        Raises VectorStoreNotFoundError if no store exists in `directory`, and
        VectorStoreError if the store is incomplete, corrupt, or inconsistent.
        """
        path = Path(directory)
        files = [path / EMBEDDINGS_FILE, path / CHUNKS_FILE, path / METADATA_FILE]
        present = [f.is_file() for f in files]
        if not any(present):
            raise VectorStoreNotFoundError(f"No vector store found at {path}.")
        if not all(present):
            missing = ", ".join(f.name for f, ok in zip(files, present) if not ok)
            raise VectorStoreError(f"Vector store at {path} is incomplete (missing {missing}).")

        metadata = _read_json(path / METADATA_FILE)
        records = _read_json(path / CHUNKS_FILE)
        try:
            embeddings = np.load(path / EMBEDDINGS_FILE, allow_pickle=False)
        except Exception as err:
            raise VectorStoreError(f"Could not read {EMBEDDINGS_FILE}: {err}") from err

        embedding_model, dimension, task, task_recorded = _validate_metadata(metadata)
        chunks = _parse_chunks(records)

        if embeddings.ndim != 2 or not np.issubdtype(embeddings.dtype, np.floating):
            raise VectorStoreError(f"{EMBEDDINGS_FILE} must contain a 2-D floating-point matrix.")
        if embeddings.shape[0] != len(chunks):
            raise VectorStoreError(
                f"{EMBEDDINGS_FILE} has {embeddings.shape[0]} rows but {CHUNKS_FILE} "
                f"has {len(chunks)} chunks."
            )
        if metadata.get("chunk_count", len(chunks)) != len(chunks):
            raise VectorStoreError(
                f"{METADATA_FILE} records {metadata['chunk_count']} chunks but "
                f"{CHUNKS_FILE} has {len(chunks)}."
            )
        if chunks:
            if dimension is None or embeddings.shape[1] != dimension:
                raise VectorStoreError(
                    f"{EMBEDDINGS_FILE} has width {embeddings.shape[1]} but {METADATA_FILE} "
                    f"records embedding_dimension {dimension}."
                )
            _check_usable_rows(embeddings, EMBEDDINGS_FILE)

        store = cls(
            embedding_model=embedding_model,
            embedding_dimension=dimension,
            document_embedding_task=task,
        )
        store._document_embedding_task_recorded = task_recorded
        if chunks:
            store._chunks = chunks
            store._embeddings = embeddings.astype(np.float32, copy=False)
            store._normalized = _normalize_rows(store._embeddings)
        return store


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms if len(matrix) else matrix


def _check_usable_rows(matrix: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(matrix)):
        raise VectorStoreError(f"{label} contain non-finite values.")
    if np.any(np.linalg.norm(matrix, axis=1) == 0):
        raise VectorStoreError(f"{label} contain an all-zero vector, which has no direction.")


def _read_json(path: Path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as err:
        raise VectorStoreError(f"Could not read {path.name}: {err}") from err


def _validate_metadata(metadata) -> tuple[str, Optional[int], Optional[str], bool]:
    """Returns (embedding_model, embedding_dimension, document_embedding_task, task_recorded)."""
    if not isinstance(metadata, dict):
        raise VectorStoreError(f"{METADATA_FILE} must contain a JSON object.")
    for key in ("embedding_model", "embedding_dimension"):
        if key not in metadata:
            raise VectorStoreError(f"{METADATA_FILE} is missing required key '{key}'.")

    embedding_model = metadata["embedding_model"]
    dimension = metadata["embedding_dimension"]
    if not isinstance(embedding_model, str) or not embedding_model:
        raise VectorStoreError(f"{METADATA_FILE}: embedding_model must be a non-empty string.")
    if dimension is not None and (
        not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0
    ):
        raise VectorStoreError(f"{METADATA_FILE}: embedding_dimension must be a positive integer.")

    # Optional for loading (older indexes lack it), but its absence is remembered.
    task_recorded = "document_embedding_task" in metadata
    task = metadata.get("document_embedding_task")
    if task is not None and (not isinstance(task, str) or not task):
        raise VectorStoreError(
            f"{METADATA_FILE}: document_embedding_task must be a non-empty string or null."
        )
    return embedding_model, dimension, task, task_recorded


def _parse_chunks(records) -> list[Chunk]:
    if not isinstance(records, list):
        raise VectorStoreError(f"{CHUNKS_FILE} must contain a JSON list of chunk records.")

    chunks = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != set(_CHUNK_FIELDS):
            raise VectorStoreError(
                f"{CHUNKS_FILE} record {index} must have exactly the fields "
                f"{sorted(_CHUNK_FIELDS)}."
            )
        for field, expected in _CHUNK_FIELDS.items():
            value = record[field]
            if not isinstance(value, expected) or isinstance(value, bool):
                raise VectorStoreError(f"{CHUNKS_FILE} record {index} has an invalid '{field}'.")
        chunks.append(Chunk(**record))

    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise VectorStoreError(f"{CHUNKS_FILE} contains duplicate chunk IDs.")
    return chunks
