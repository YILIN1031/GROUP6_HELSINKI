"""
Builds the local course-material index.

Usage (from the repository root):

    python -m src.rag.ingest

Reads documents from DOCUMENTS_PATH, chunks and embeds them with EMBEDDING_MODEL,
and writes the index to VECTOR_STORE_PATH. A running application keeps the index
it loaded at startup, so restart the application after rebuilding the index.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.config import Config, config
from src.models.model_client import ModelClientError, ModelNotFoundError, OllamaConnectionError
from src.rag.chunker import chunk_documents
from src.rag.document_loader import SkippedFile, load_documents
from src.rag.embeddings import SEARCH_DOCUMENT, Embedder, OllamaEmbedder
from src.rag.vector_store import VectorStore, VectorStoreError, VectorStoreNotFoundError

RESTART_REMINDER = "Restart the application (python -m app.main) to query the rebuilt index."


class IngestError(Exception):
    """A user-facing ingestion failure that should be reported without a traceback."""


class NoUsableDocumentsError(IngestError):
    """Raised when the documents directory contains nothing that can be indexed."""

    def __init__(self, message: str, skipped: list[SkippedFile]):
        super().__init__(message)
        self.skipped = skipped


@dataclass
class IngestReport:
    """Summary of a successful index build."""

    documents_path: str
    store_path: str
    embedding_model: str
    embedding_dimension: Optional[int]
    document_embedding_task: Optional[str]
    document_count: int
    chunk_count: int
    skipped: list[SkippedFile] = field(default_factory=list)


def build_index(
    documents_path: str | Path,
    store_path: str | Path,
    embedder: Embedder,
    size: int,
    overlap: int,
) -> IngestReport:
    """
    Runs the ingestion pipeline: load -> chunk -> embed -> build VectorStore -> save.

    The index is written only after every step has succeeded, so a failure never
    leaves a partially built index, and an existing index is left untouched.

    Raises:
        IngestError: the documents directory does not exist.
        NoUsableDocumentsError: no document produced any indexable text.
        ValueError: invalid chunk size/overlap.
        ModelClientError: embedding failed (e.g. Ollama unreachable, model missing).
    """
    try:
        documents, skipped = load_documents(documents_path)
    except FileNotFoundError as err:
        raise IngestError(str(err)) from err

    chunks = chunk_documents(documents, size=size, overlap=overlap)
    if not chunks:
        raise NoUsableDocumentsError(
            f"No usable course documents found in {documents_path}. "
            "Add .pdf, .txt, or .md files with extractable text.",
            skipped,
        )

    # Chunks are stored raw; the embedder applies any model-specific document prefix.
    # The same task is recorded in the index so queries can be checked against it.
    document_task = SEARCH_DOCUMENT
    vectors = embedder.embed([chunk.text for chunk in chunks], task=document_task)

    store = VectorStore(embedding_model=embedder.model_name, document_embedding_task=document_task)
    store.add(chunks, vectors)
    store.save(store_path)

    return IngestReport(
        documents_path=str(documents_path),
        store_path=str(store_path),
        embedding_model=store.embedding_model,
        embedding_dimension=store.embedding_dimension,
        document_embedding_task=store.document_embedding_task,
        document_count=len(documents),
        chunk_count=len(chunks),
        skipped=skipped,
    )


def _print_skipped(skipped: list[SkippedFile], stream) -> None:
    if skipped:
        print(f"Skipped {len(skipped)} file(s):", file=stream)
        for item in skipped:
            print(f"  - {item.source}: {item.reason}", file=stream)


def _existing_index_note(store_path: str) -> Optional[str]:
    try:
        VectorStore.load(store_path)
    except VectorStoreNotFoundError:
        return None
    except VectorStoreError:
        return f"Note: the existing index at {store_path} was left unchanged (it appears to be invalid)."
    return (
        f"Note: the existing index at {store_path} was left unchanged and still reflects "
        "previously ingested documents."
    )


def _make_console_output_safe() -> None:
    """
    Keeps printing from crashing when the console encoding cannot represent a
    character, e.g. Swedish å/ä/ö in file names on a GBK Windows console.
    Such characters are shown as '?' instead of raising UnicodeEncodeError.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def main(embedder: Optional[Embedder] = None, settings: Optional[Config] = None) -> int:
    """Command-line entry point. Returns a process exit code (0 = success)."""
    _make_console_output_safe()
    settings = settings or config
    embedder = embedder or OllamaEmbedder(
        base_url=settings.ollama_base_url, model_name=settings.embedding_model
    )

    print(
        f"Building index from {settings.documents_path} using embedding model '{embedder.model_name}'...",
        flush=True,  # keep this line ahead of any stderr error output when piped
    )
    try:
        report = build_index(
            settings.documents_path,
            settings.vector_store_path,
            embedder,
            size=settings.rag_chunk_size,
            overlap=settings.rag_chunk_overlap,
        )
    except NoUsableDocumentsError as err:
        print(f"Error: {err}", file=sys.stderr)
        _print_skipped(err.skipped, sys.stderr)
        note = _existing_index_note(settings.vector_store_path)
        if note:
            print(note, file=sys.stderr)
        print("No index was written.", file=sys.stderr)
        return 1
    except IngestError as err:
        print(f"Error: {err}", file=sys.stderr)
        print("Create the directory and add course documents, or set DOCUMENTS_PATH in .env.", file=sys.stderr)
        return 1
    except ValueError as err:
        print(f"Error: invalid chunking settings: {err}", file=sys.stderr)
        print("Check RAG_CHUNK_SIZE and RAG_CHUNK_OVERLAP in .env.", file=sys.stderr)
        return 1
    except OllamaConnectionError as err:
        print(f"Error: {err} Make sure Ollama is installed and running.", file=sys.stderr)
        print("No index was written.", file=sys.stderr)
        return 1
    except ModelNotFoundError as err:
        print(f"Error: {err} Install it with: ollama pull {embedder.model_name}", file=sys.stderr)
        print("No index was written.", file=sys.stderr)
        return 1
    except ModelClientError as err:
        print(f"Error: embedding failed: {err}", file=sys.stderr)
        print("No index was written.", file=sys.stderr)
        return 1
    except (VectorStoreError, OSError) as err:
        print(f"Error: could not write the index to {settings.vector_store_path}: {err}", file=sys.stderr)
        return 1

    print(
        f"Indexed {report.chunk_count} chunk(s) from {report.document_count} document(s) "
        f"(embedding dimension {report.embedding_dimension}, "
        f"document embedding task {report.document_embedding_task})."
    )
    _print_skipped(report.skipped, sys.stdout)
    print(f"Index written to {report.store_path}.")
    print(RESTART_REMINDER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
