import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import ollama
import pytest

from src.config import Config
from src.models.model_client import ModelNotFoundError, OllamaConnectionError
from src.rag.embeddings import SEARCH_DOCUMENT, OllamaEmbedder
from src.rag.ingest import (
    RESTART_REMINDER,
    IngestError,
    NoUsableDocumentsError,
    build_index,
    main,
)
from src.rag.vector_store import CHUNKS_FILE, EMBEDDINGS_FILE, METADATA_FILE, VectorStore
from tests.test_document_loader import make_pdf

REPO_ROOT = Path(__file__).resolve().parent.parent
LONG_TEXT = " ".join(f"token{i:03d}" for i in range(300))  # ~2,700 characters -> several chunks


class FakeEmbedder:
    """Deterministic embedder: counts of a few letters, plus 1 so no vector is all-zero."""

    model_name = "fake-embed"

    def __init__(self, error=None):
        self.calls = []
        self.tasks = []
        self.error = error

    def embed(self, texts, task=None):
        self.calls.append(list(texts))
        self.tasks.append(task)
        if self.error:
            raise self.error
        return [[text.count(c) + 1.0 for c in "aeiou"] for text in texts]


def write_course_documents(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "notes.txt").write_text(LONG_TEXT, encoding="utf-8")
    (directory / "revision.md").write_text("# Overfitting\n\nA model memorises noise.", encoding="utf-8")
    make_pdf(directory / "lecture_04.pdf", ["Regularisation reduces overfitting.", "Validation data."])


def make_settings(tmp_path, **overrides):
    values = {
        "documents_path": str(tmp_path / "documents"),
        "vector_store_path": str(tmp_path / "vector_store"),
        "rag_chunk_size": 1000,
        "rag_chunk_overlap": 200,
    }
    values.update(overrides)
    return Config(**values)


def store_files_exist(store_path):
    return [(Path(store_path) / name).exists() for name in (EMBEDDINGS_FILE, CHUNKS_FILE, METADATA_FILE)]


# ---------------------------------------------------------------- build_index


def test_build_index_runs_full_pipeline_and_saves_loadable_store(tmp_path):
    """Verify load -> chunk -> embed -> store -> save produces a consistent three-file index."""
    write_course_documents(tmp_path / "documents")
    embedder = FakeEmbedder()

    report = build_index(tmp_path / "documents", tmp_path / "store", embedder, size=1000, overlap=200)

    assert store_files_exist(tmp_path / "store") == [True, True, True]
    store = VectorStore.load(tmp_path / "store")
    assert report.document_count == 3
    assert report.chunk_count == len(store) > 3  # the long text file yields several chunks
    assert report.embedding_model == store.embedding_model == "fake-embed"
    assert report.embedding_dimension == store.embedding_dimension == 5
    assert report.skipped == []
    assert {c.source for c in store.chunks} == {"lecture_04.pdf", "notes.txt", "revision.md"}
    assert {c.page for c in store.chunks if c.source == "lecture_04.pdf"} == {1, 2}
    # All raw chunk texts were embedded in a single call, in store order, as documents.
    assert embedder.calls == [[c.text for c in store.chunks]]
    assert embedder.tasks == [SEARCH_DOCUMENT]


def test_build_index_records_document_embedding_task_in_metadata(tmp_path):
    """Verify the index records the document task it was built with, as the logical name only."""
    write_course_documents(tmp_path / "documents")

    report = build_index(tmp_path / "documents", tmp_path / "store", FakeEmbedder(), size=1000, overlap=200)

    metadata = json.loads((tmp_path / "store" / METADATA_FILE).read_text(encoding="utf-8"))
    chunks = json.loads((tmp_path / "store" / CHUNKS_FILE).read_text(encoding="utf-8"))
    loaded = VectorStore.load(tmp_path / "store")

    assert metadata["document_embedding_task"] == "search_document" == SEARCH_DOCUMENT
    assert report.document_embedding_task == "search_document"
    assert loaded.document_embedding_task == "search_document"
    assert loaded.compatibility_issue("fake-embed", SEARCH_DOCUMENT) is None
    assert all(set(c) == {"chunk_id", "source", "page", "text"} for c in chunks)


def test_build_index_sends_document_prefix_once_and_stores_raw_text(tmp_path):
    """Verify with the real OllamaEmbedder (client patched) that Nomic chunks are embedded as
    'search_document: <text>' exactly once, while chunks.json keeps the unprefixed text."""
    write_course_documents(tmp_path / "documents")

    def fake_embed(model, input):
        return ollama.EmbedResponse(embeddings=[[float(len(text)), 1.0] for text in input])

    with patch("ollama.Client") as client_class:
        client_class.return_value.embed.side_effect = fake_embed
        embedder = OllamaEmbedder(model_name="nomic-embed-text", batch_size=2)
        report = build_index(tmp_path / "documents", tmp_path / "store", embedder, size=1000, overlap=200)

    sent = [text for call in client_class.return_value.embed.call_args_list for text in call.kwargs["input"]]
    stored = json.loads((tmp_path / "store" / CHUNKS_FILE).read_text(encoding="utf-8"))

    assert len(sent) == report.chunk_count == len(stored)
    assert sent == [f"search_document: {record['text']}" for record in stored]
    assert all(text.count("search_document: ") == 1 for text in sent)
    assert not any(record["text"].startswith("search_document:") for record in stored)

    metadata = json.loads((tmp_path / "store" / METADATA_FILE).read_text(encoding="utf-8"))
    assert metadata["embedding_model"] == "nomic-embed-text"
    assert metadata["document_embedding_task"] == "search_document"


def test_build_index_reports_skipped_files_and_indexes_the_rest(tmp_path):
    """Verify unusable files are reported while usable documents are still indexed."""
    docs = tmp_path / "documents"
    write_course_documents(docs)
    (docs / "slides.pptx").write_bytes(b"binary")
    (docs / "empty.txt").write_text("", encoding="utf-8")
    (docs / "broken.pdf").write_bytes(b"%PDF-1.4 not really")

    report = build_index(docs, tmp_path / "store", FakeEmbedder(), size=1000, overlap=200)

    assert sorted(s.source for s in report.skipped) == ["broken.pdf", "empty.txt", "slides.pptx"]
    sources = {c.source for c in VectorStore.load(tmp_path / "store").chunks}
    assert sources == {"lecture_04.pdf", "notes.txt", "revision.md"}


def test_build_index_uses_chunk_settings(tmp_path):
    """Verify the chunk size passed in controls how many chunks are produced."""
    write_course_documents(tmp_path / "documents")

    small = build_index(tmp_path / "documents", tmp_path / "a", FakeEmbedder(), size=300, overlap=50)
    large = build_index(tmp_path / "documents", tmp_path / "b", FakeEmbedder(), size=1000, overlap=200)

    assert small.chunk_count > large.chunk_count


def test_build_index_missing_directory_raises_ingest_error(tmp_path):
    """Verify a missing documents directory raises IngestError naming the path, writing nothing."""
    with pytest.raises(IngestError, match="nowhere"):
        build_index(tmp_path / "nowhere", tmp_path / "store", FakeEmbedder(), size=1000, overlap=200)

    assert not (tmp_path / "store").exists()


def test_build_index_empty_directory_writes_no_index(tmp_path):
    """Verify an empty documents directory raises NoUsableDocumentsError and creates no index."""
    (tmp_path / "documents").mkdir()
    embedder = FakeEmbedder()

    with pytest.raises(NoUsableDocumentsError):
        build_index(tmp_path / "documents", tmp_path / "store", embedder, size=1000, overlap=200)

    assert not (tmp_path / "store").exists()
    assert embedder.calls == []


def test_build_index_only_unusable_documents_writes_no_index(tmp_path):
    """Verify a directory with only unusable files raises with the skipped details and writes nothing."""
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "empty.md").write_text("  ", encoding="utf-8")
    make_pdf(docs / "scanned.pdf", [None])

    with pytest.raises(NoUsableDocumentsError) as excinfo:
        build_index(docs, tmp_path / "store", FakeEmbedder(), size=1000, overlap=200)

    assert sorted(s.source for s in excinfo.value.skipped) == ["empty.md", "scanned.pdf"]
    assert not (tmp_path / "store").exists()


def test_build_index_rebuild_replaces_previous_index(tmp_path):
    """Verify re-running ingestion overwrites the index to match the current documents."""
    docs = tmp_path / "documents"
    write_course_documents(docs)
    build_index(docs, tmp_path / "store", FakeEmbedder(), size=1000, overlap=200)

    (docs / "notes.txt").unlink()
    report = build_index(docs, tmp_path / "store", FakeEmbedder(), size=1000, overlap=200)

    store = VectorStore.load(tmp_path / "store")
    assert len(store) == report.chunk_count
    assert "notes.txt" not in {c.source for c in store.chunks}


def test_build_index_embedding_failure_leaves_existing_index_untouched(tmp_path):
    """Verify an embedding error propagates and the previously built index is not modified."""
    docs = tmp_path / "documents"
    write_course_documents(docs)
    build_index(docs, tmp_path / "store", FakeEmbedder(), size=1000, overlap=200)
    before = (tmp_path / "store" / CHUNKS_FILE).read_bytes()

    (docs / "extra.txt").write_text("New material.", encoding="utf-8")
    with pytest.raises(OllamaConnectionError):
        build_index(docs, tmp_path / "store", FakeEmbedder(error=OllamaConnectionError("down")), 1000, 200)

    assert (tmp_path / "store" / CHUNKS_FILE).read_bytes() == before


def test_build_index_invalid_chunk_settings_raise_value_error(tmp_path):
    """Verify invalid chunking parameters are rejected before anything is embedded or saved."""
    write_course_documents(tmp_path / "documents")
    embedder = FakeEmbedder()

    with pytest.raises(ValueError):
        build_index(tmp_path / "documents", tmp_path / "store", embedder, size=100, overlap=100)

    assert embedder.calls == []
    assert not (tmp_path / "store").exists()


# ---------------------------------------------------------------- CLI main()


def test_main_success_prints_summary_and_restart_reminder(tmp_path, capsys):
    """Verify a successful CLI run returns 0, reports counts and skipped files, and reminds to restart."""
    settings = make_settings(tmp_path)
    write_course_documents(Path(settings.documents_path))
    (Path(settings.documents_path) / "slides.pptx").write_bytes(b"binary")

    exit_code = main(embedder=FakeEmbedder(), settings=settings)

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "from 3 document(s)" in out
    assert "document embedding task search_document" in out
    assert "Skipped 1 file(s):" in out
    assert "slides.pptx: unsupported file type '.pptx'" in out
    assert f"Index written to {settings.vector_store_path}." in out
    assert out.rstrip().endswith(RESTART_REMINDER)
    assert store_files_exist(settings.vector_store_path) == [True, True, True]


def test_main_missing_directory_reports_clear_error(tmp_path, capsys):
    """Verify a missing documents directory gives exit code 1 and a readable message, not a traceback."""
    settings = make_settings(tmp_path)

    exit_code = main(embedder=FakeEmbedder(), settings=settings)

    err = capsys.readouterr().err
    assert exit_code == 1
    assert "Error: Documents directory not found" in err
    assert "DOCUMENTS_PATH" in err
    assert "Traceback" not in err
    assert not Path(settings.vector_store_path).exists()


def test_main_empty_directory_reports_no_usable_documents(tmp_path, capsys):
    """Verify an empty directory gives exit code 1, says nothing was indexed, and writes no index."""
    settings = make_settings(tmp_path)
    Path(settings.documents_path).mkdir()

    exit_code = main(embedder=FakeEmbedder(), settings=settings)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "No usable course documents found" in captured.err
    assert "No index was written." in captured.err
    assert RESTART_REMINDER not in captured.out
    assert not Path(settings.vector_store_path).exists()


def test_main_no_usable_documents_lists_skipped_and_warns_about_existing_index(tmp_path, capsys):
    """Verify skipped files are listed and an older index is flagged as unchanged, not silently reused."""
    settings = make_settings(tmp_path)
    docs = Path(settings.documents_path)
    write_course_documents(docs)
    assert main(embedder=FakeEmbedder(), settings=settings) == 0
    capsys.readouterr()

    for path in docs.iterdir():
        path.unlink()
    (docs / "empty.txt").write_text("", encoding="utf-8")

    exit_code = main(embedder=FakeEmbedder(), settings=settings)

    err = capsys.readouterr().err
    assert exit_code == 1
    assert "empty.txt: file is empty" in err
    assert "left unchanged and still reflects previously ingested documents" in err


@pytest.mark.parametrize(
    "error, expected",
    [
        (OllamaConnectionError("Failed to connect to Ollama service at http://x."), "Make sure Ollama is installed and running."),
        (ModelNotFoundError("Model 'fake-embed' is not installed in local Ollama."), "ollama pull fake-embed"),
    ],
)
def test_main_ollama_failures_report_actionable_messages(tmp_path, capsys, error, expected):
    """Verify Ollama connection and missing-model errors give exit code 1 with an actionable hint."""
    settings = make_settings(tmp_path)
    write_course_documents(Path(settings.documents_path))

    exit_code = main(embedder=FakeEmbedder(error=error), settings=settings)

    err = capsys.readouterr().err
    assert exit_code == 1
    assert expected in err
    assert "No index was written." in err
    assert not Path(settings.vector_store_path).exists()


def test_main_invalid_chunk_settings_report_clear_error(tmp_path, capsys):
    """Verify invalid chunk settings give exit code 1 and point to the relevant variables."""
    settings = make_settings(tmp_path, rag_chunk_size=100, rag_chunk_overlap=200)
    write_course_documents(Path(settings.documents_path))

    exit_code = main(embedder=FakeEmbedder(), settings=settings)

    err = capsys.readouterr().err
    assert exit_code == 1
    assert "RAG_CHUNK_SIZE and RAG_CHUNK_OVERLAP" in err


def test_main_defaults_to_configured_ollama_embedder(tmp_path, monkeypatch, capsys):
    """Verify main() builds an OllamaEmbedder from the configured base URL and embedding model."""
    created = {}

    class RecordingEmbedder(FakeEmbedder):
        def __init__(self, base_url, model_name):
            super().__init__()
            created.update(base_url=base_url, model_name=model_name)
            self.model_name = model_name

    monkeypatch.setattr("src.rag.ingest.OllamaEmbedder", RecordingEmbedder)
    settings = make_settings(tmp_path, ollama_base_url="http://ollama:1", embedding_model="my-embed")
    write_course_documents(Path(settings.documents_path))

    assert main(settings=settings) == 0
    assert created == {"base_url": "http://ollama:1", "model_name": "my-embed"}
    assert VectorStore.load(settings.vector_store_path).embedding_model == "my-embed"


def test_module_entry_point_reports_missing_directory_without_traceback(tmp_path):
    """Verify `python -m src.rag.ingest` exits with code 1 and a clean message (no Ollama needed)."""
    env = dict(os.environ)
    env["DOCUMENTS_PATH"] = str(tmp_path / "missing_documents")
    env["VECTOR_STORE_PATH"] = str(tmp_path / "vector_store")

    result = subprocess.run(
        [sys.executable, "-m", "src.rag.ingest"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 1
    assert "Documents directory not found" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "vector_store").exists()


# ---------------------------------------------------------------- console encoding robustness


def gbk_stream():
    """A text stream that, like a GBK Windows console, cannot encode Swedish å/ä/ö."""
    import io

    return io.TextIOWrapper(io.BytesIO(), encoding="gbk", errors="strict")


def test_main_does_not_crash_on_swedish_names_with_gbk_console(tmp_path, monkeypatch):
    """Regression: Swedish characters in paths and skipped-file names must not raise UnicodeEncodeError."""
    out, err = gbk_stream(), gbk_stream()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    settings = make_settings(tmp_path, documents_path=str(tmp_path / "språkdokument"))
    write_course_documents(Path(settings.documents_path))
    (Path(settings.documents_path) / "Svenska språkexamen.pptx").write_bytes(b"binary")

    exit_code = main(embedder=FakeEmbedder(), settings=settings)

    out.flush()
    printed = out.buffer.getvalue().decode("gbk")
    assert exit_code == 0
    assert "Svenska spr?kexamen.pptx: unsupported file type '.pptx'" in printed
    assert RESTART_REMINDER in printed


def test_main_error_path_does_not_crash_with_gbk_console(tmp_path, monkeypatch):
    """Regression: an error message containing Swedish characters is printed, not raised."""
    out, err = gbk_stream(), gbk_stream()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    settings = make_settings(tmp_path, documents_path=str(tmp_path / "saknade_filer_på_svenska"))

    exit_code = main(embedder=FakeEmbedder(), settings=settings)

    err.flush()
    assert exit_code == 1
    assert "Documents directory not found" in err.buffer.getvalue().decode("gbk")


def test_module_entry_point_survives_gbk_console_encoding(tmp_path):
    """Regression for the documented workflow: `python -m src.rag.ingest` under a GBK console
    with a Swedish path exits cleanly (code 1, readable message) instead of a traceback."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "gbk"
    env["DOCUMENTS_PATH"] = str(tmp_path / "språkdokument_saknas")
    env["VECTOR_STORE_PATH"] = str(tmp_path / "vector_store")

    result = subprocess.run(
        [sys.executable, "-m", "src.rag.ingest"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        timeout=60,
    )

    stderr = result.stderr.decode("gbk", errors="replace")
    assert result.returncode == 1
    assert "Documents directory not found" in stderr
    assert "Traceback" not in stderr and "UnicodeEncodeError" not in stderr
