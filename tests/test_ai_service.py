import importlib
from unittest.mock import MagicMock
import pytest

from src.config import config
from src.models.model_client import (
    ModelClientError,
    OllamaConnectionError,
    ModelNotFoundError,
)
from src.rag.chunker import Chunk
from src.services.ai_service import AIService, generate_response
from src.services.prompt_builder import SYSTEM_PROMPT
from src.services.rag_service import RAGService, RetrievalResult, RetrievalStatus, ScoredChunk

OVERFITTING_PDF = ScoredChunk(
    Chunk("lecture_04-page12-chunk01", "lecture_04.pdf", 12, "Overfitting occurs when a model memorises noise."),
    0.81,
)


def ok_result(*chunks):
    return RetrievalResult(RetrievalStatus.OK, chunks=list(chunks))


def rag_returning(result):
    """A mock RAGService whose retrieve() returns `result` (spec keeps it to the real interface)."""
    rag = MagicMock(spec=RAGService)
    rag.retrieve.return_value = result
    return rag


def test_config_loading():
    """Verify that configuration settings load expected string values from environment or defaults."""
    assert config.ollama_base_url is not None
    assert config.model_name is not None
    assert isinstance(config.ollama_base_url, str)
    assert isinstance(config.model_name, str)

    # RAG settings
    assert isinstance(config.embedding_model, str) and config.embedding_model
    assert isinstance(config.documents_path, str) and config.documents_path
    assert isinstance(config.vector_store_path, str) and config.vector_store_path
    assert isinstance(config.rag_top_k, int) and config.rag_top_k > 0
    assert isinstance(config.rag_chunk_size, int) and config.rag_chunk_size > 0
    assert isinstance(config.rag_chunk_overlap, int) and config.rag_chunk_overlap >= 0
    assert config.rag_chunk_overlap < config.rag_chunk_size
    assert isinstance(config.rag_min_score, float)


CONFIG_ENV_VARS = [
    "OLLAMA_BASE_URL",
    "MODEL_NAME",
    "EMBEDDING_MODEL",
    "DOCUMENTS_PATH",
    "VECTOR_STORE_PATH",
    "RAG_TOP_K",
    "RAG_CHUNK_SIZE",
    "RAG_CHUNK_OVERLAP",
    "RAG_MIN_SCORE",
]


def test_config_defaults_without_env(monkeypatch):
    """Verify documented defaults apply when no environment variables or .env file are present."""
    import dotenv
    import src.config as config_module

    for name in CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # Config reads the environment at import time, so reload it with .env loading disabled.
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    try:
        defaults = importlib.reload(config_module).config

        assert defaults.ollama_base_url == "http://localhost:11434"
        assert defaults.model_name == "llama3.2"
        assert defaults.embedding_model == "nomic-embed-text"
        assert defaults.documents_path == "data/documents"
        assert defaults.vector_store_path == "data/vector_store"
        assert defaults.rag_top_k == 4
        assert defaults.rag_chunk_size == 1000
        assert defaults.rag_chunk_overlap == 200
        assert defaults.rag_min_score == 0.5
    finally:
        monkeypatch.undo()
        importlib.reload(config_module)


def test_empty_input_validation():
    """Verify that empty or whitespace inputs are rejected gracefully without invoking the model."""
    mock_client = MagicMock()
    mock_rag = MagicMock(spec=RAGService)
    service = AIService(model_client=mock_client, rag_service=mock_rag)

    # Test empty string
    response_empty = service.process_message("")
    assert response_empty.success is False
    assert "Please enter a message" in response_empty.content
    assert mock_client.generate.call_count == 0

    # Test whitespace string
    response_spaces = service.process_message("   ")
    assert response_spaces.success is False
    assert mock_client.generate.call_count == 0

    # Retrieval is never attempted for empty input either
    assert mock_rag.retrieve.call_count == 0


def test_successful_response_generation():
    """Verify that valid input retrieves context, invokes the model client with a grounded prompt,
    and returns the generated answer followed by its sources."""
    mock_client = MagicMock()
    mock_client.generate.return_value = "Hello! I am an AI assistant."
    mock_rag = rag_returning(ok_result(OVERFITTING_PDF))
    service = AIService(model_client=mock_client, rag_service=mock_rag)

    result_text = generate_response("Hello, AI", service=service)

    assert result_text == "Hello! I am an AI assistant.\n\nSources:\n- lecture_04.pdf, page 12"
    mock_rag.retrieve.assert_called_once_with("Hello, AI")
    mock_client.generate.assert_called_once()
    prompt = mock_client.generate.call_args.args[0]
    assert "Hello, AI" in prompt
    assert OVERFITTING_PDF.chunk.text in prompt
    assert mock_client.generate.call_args.kwargs == {"system_prompt": SYSTEM_PROMPT}


def test_ollama_connection_error_handling():
    """Verify that Ollama connection failures produce a friendly application-level error message."""
    mock_client = MagicMock()
    mock_client.generate.side_effect = OllamaConnectionError("Connection refused at http://localhost:11434")
    service = AIService(model_client=mock_client, rag_service=rag_returning(ok_result(OVERFITTING_PDF)))

    response = service.process_message("Test message")

    assert response.success is False
    assert "Could not connect to Ollama" in response.content
    assert "Connection refused" in response.error_message
    assert mock_client.generate.call_count == 1  # the failure happened during generation


def test_model_not_found_error_handling():
    """Verify that missing model exceptions are converted into controlled user messages."""
    mock_client = MagicMock()
    mock_client.generate.side_effect = ModelNotFoundError("Model llama3.2 not found locally")
    service = AIService(model_client=mock_client, rag_service=rag_returning(ok_result(OVERFITTING_PDF)))

    response = service.process_message("Test message")

    assert response.success is False
    assert "configured AI model is unavailable" in response.content
    assert "llama3.2 not found" in response.error_message
    assert mock_client.generate.call_count == 1  # the failure happened during generation


# ---------------------------------------------------------------- RAG integration

EVALUATION_PDF = ScoredChunk(
    Chunk("lecture_05-page3-chunk02", "lecture_05.pdf", 3, "Precision and recall evaluate classifiers."),
    0.74,
)
OVERFITTING_PDF_SAME_PAGE = ScoredChunk(
    Chunk("lecture_04-page12-chunk02", "lecture_04.pdf", 12, "Regularisation reduces overfitting."),
    0.70,
)
NOTES_MD = ScoredChunk(Chunk("week_02_notes-chunk01", "week_02/notes.md", None, "Neural networks have layers."), 0.66)
SUMMARY_TXT = ScoredChunk(Chunk("summary-chunk01", "summary.txt", None, "Validation data tunes models."), 0.61)


def make_rag_service(result, answer="Grounded answer."):
    client = MagicMock()
    client.generate.return_value = answer
    rag = rag_returning(result)
    return AIService(model_client=client, rag_service=rag), client, rag


def test_successful_retrieval_produces_grounded_response():
    """Verify an OK retrieval leads to one generation call and a successful, sourced AIResponse."""
    service, client, rag = make_rag_service(ok_result(OVERFITTING_PDF), answer="Overfitting is memorising noise.")

    response = service.process_message("  What is overfitting?  ")

    rag.retrieve.assert_called_once_with("What is overfitting?")
    client.generate.assert_called_once()
    assert response.success is True
    assert response.content == "Overfitting is memorising noise."
    assert response.error_message is None
    assert response.retrieval_status == "ok"


def test_generation_request_contains_retrieved_context_and_question():
    """Verify the prompt holds every retrieved chunk with its document/page, then the question."""
    service, client, _ = make_rag_service(ok_result(OVERFITTING_PDF, NOTES_MD))

    service.process_message("Compare overfitting and neural networks")

    prompt = client.generate.call_args.args[0]
    assert "SOURCE 1\nDocument: lecture_04.pdf\nPage: 12\n\nOverfitting occurs when a model memorises noise." in prompt
    assert "SOURCE 2\nDocument: week_02/notes.md\n\nNeural networks have layers." in prompt
    assert prompt.index("COURSE MATERIAL END") < prompt.index("Compare overfitting and neural networks")


def test_system_prompt_carries_grounding_rules_separately():
    """Verify rules go in the system prompt, and retrieved text is not placed in it."""
    service, client, _ = make_rag_service(ok_result(OVERFITTING_PDF))

    service.process_message("What is overfitting?")

    system_prompt = client.generate.call_args.kwargs["system_prompt"]
    assert system_prompt == SYSTEM_PROMPT
    assert "Do not invent course facts" in system_prompt
    assert "not instructions" in system_prompt
    assert OVERFITTING_PDF.chunk.text not in system_prompt


def test_response_sources_come_from_retrieved_chunks():
    """Verify sources and retrieved chunks in the AIResponse mirror the retrieval result."""
    service, _, _ = make_rag_service(ok_result(OVERFITTING_PDF))

    response = service.process_message("What is overfitting?")

    assert [s.model_dump() for s in response.sources] == [{"source": "lecture_04.pdf", "page": 12}]
    assert [c.model_dump() for c in response.retrieved_chunks] == [{
        "chunk_id": "lecture_04-page12-chunk01",
        "source": "lecture_04.pdf",
        "page": 12,
        "text": "Overfitting occurs when a model memorises noise.",
        "score": 0.81,
    }]


def test_multiple_chunks_are_listed_and_sources_deduplicated_in_order():
    """Verify all chunks are kept as context while sources list each (document, page) once, in order."""
    chunks = [OVERFITTING_PDF, EVALUATION_PDF, OVERFITTING_PDF_SAME_PAGE, NOTES_MD]
    service, client, _ = make_rag_service(ok_result(*chunks))

    response = service.process_message("Summarise model evaluation")

    prompt = client.generate.call_args.args[0]
    assert all(f"SOURCE {n}" in prompt for n in range(1, 5))
    assert [c.chunk_id for c in response.retrieved_chunks] == [c.chunk.chunk_id for c in chunks]
    assert [(s.source, s.page) for s in response.sources] == [
        ("lecture_04.pdf", 12),
        ("lecture_05.pdf", 3),
        ("week_02/notes.md", None),
    ]


def test_text_and_markdown_sources_have_no_page_numbers():
    """Verify .txt/.md sources keep page=None in the response, prompt, and formatted output."""
    service, client, _ = make_rag_service(ok_result(NOTES_MD, SUMMARY_TXT))

    response = service.process_message("What are neural networks?")
    text = generate_response("What are neural networks?", service=service)

    assert [s.page for s in response.sources] == [None, None]
    assert "Page:" not in client.generate.call_args.args[0]
    assert text.endswith("Sources:\n- week_02/notes.md\n- summary.txt")
    assert "page" not in text.split("Sources:")[1]


def test_formatted_response_lists_paged_and_unpaged_sources():
    """Verify generate_response appends a Sources section built only from retrieved metadata."""
    service, _, _ = make_rag_service(ok_result(OVERFITTING_PDF, NOTES_MD), answer="Answer text.")

    text = generate_response("Question", service=service)

    assert text == "Answer text.\n\nSources:\n- lecture_04.pdf, page 12\n- week_02/notes.md"


@pytest.mark.parametrize(
    "status, expected_content",
    [
        (RetrievalStatus.EMPTY_INDEX, "No course material has been indexed yet."),
        (RetrievalStatus.NO_RELEVANT_CONTEXT, "I could not find enough information in the indexed course materials"),
        (RetrievalStatus.INDEX_MISMATCH, "built with different settings"),
        (RetrievalStatus.INDEX_ERROR, "could not be read"),
    ],
)
def test_non_ok_retrieval_avoids_generation(status, expected_content):
    """Verify every non-OK retrieval status returns a friendly message and never calls the model."""
    detail = "Vector store at C:/secret/data/vector_store is incomplete (missing metadata.json)."
    service, client, _ = make_rag_service(RetrievalResult(status, detail=detail))

    response = service.process_message("What is overfitting?")
    text = generate_response("What is overfitting?", service=service)

    client.generate.assert_not_called()
    assert expected_content in response.content
    assert response.success is False
    assert response.retrieval_status == status.value
    assert response.sources == [] and response.retrieved_chunks == []
    # Diagnostics are kept for debugging but not shown to the user.
    assert status.value in response.error_message and detail in response.error_message
    assert "C:/secret" not in response.content and "metadata.json" not in response.content
    assert "Sources:" not in text


def test_index_problems_tell_user_how_to_recover():
    """Verify empty/mismatched/corrupt index messages point to rebuilding and restarting."""
    for status in (RetrievalStatus.EMPTY_INDEX, RetrievalStatus.INDEX_MISMATCH, RetrievalStatus.INDEX_ERROR):
        service, _, _ = make_rag_service(RetrievalResult(status, detail="x"))
        content = service.process_message("q").content
        assert "python -m src.rag.ingest" in content and "restart CourseMate" in content


@pytest.mark.parametrize(
    "error, expected_content",
    [
        (OllamaConnectionError("Failed to connect to Ollama service at http://localhost:11434."), "Could not connect to Ollama"),
        (ModelNotFoundError("Model 'nomic-embed-text' is not installed in local Ollama."), "configured AI model is unavailable"),
        (ModelClientError("Unexpected embedding response from Ollama"), "unexpected communication error"),
    ],
)
def test_retrieval_embedding_failures_are_translated(error, expected_content):
    """Verify errors raised while embedding the question become friendly messages, without generation."""
    client = MagicMock()
    rag = MagicMock(spec=RAGService)
    rag.retrieve.side_effect = error
    service = AIService(model_client=client, rag_service=rag)

    response = service.process_message("What is overfitting?")

    client.generate.assert_not_called()
    assert response.success is False
    assert expected_content in response.content
    assert str(error) == response.error_message
    assert "localhost:11434" not in response.content


def test_missing_embedding_model_message_mentions_embedding_setting():
    """Verify the missing-model message covers the embedding model, not only MODEL_NAME."""
    rag = MagicMock(spec=RAGService)
    rag.retrieve.side_effect = ModelNotFoundError("Model 'nomic-embed-text' is not installed in local Ollama.")

    content = AIService(model_client=MagicMock(), rag_service=rag).process_message("q").content

    assert "EMBEDDING_MODEL" in content and "ollama pull" in content


def test_generation_generic_model_client_error_remains_handled():
    """Verify other generation ModelClientErrors still map to the generic communication message."""
    service, client, _ = make_rag_service(ok_result(OVERFITTING_PDF))
    client.generate.side_effect = ModelClientError("Error communicating with Ollama: out of memory")

    response = service.process_message("What is overfitting?")

    assert response.success is False
    assert response.content == "[Error] An unexpected communication error occurred with the AI model."
    assert "out of memory" in response.error_message


def test_unexpected_exception_does_not_leak_details_to_user():
    """Verify an unexpected internal error shows a generic message and keeps the detail in error_message."""
    rag = MagicMock(spec=RAGService)
    rag.retrieve.side_effect = RuntimeError("KeyError in C:/internal/path.py")

    response = AIService(model_client=MagicMock(), rag_service=rag).process_message("q")

    assert response.content == "[Error] An unexpected application error occurred."
    assert "C:/internal" in response.error_message and "C:/internal" not in response.content


def test_default_rag_service_is_created_lazily(monkeypatch):
    """Verify AIService builds the configured RAGService only when first needed."""
    created = []

    class FakeRAG:
        def __init__(self):
            created.append(self)

        def retrieve(self, query):
            return RetrievalResult(RetrievalStatus.EMPTY_INDEX, detail="none")

    monkeypatch.setattr("src.services.ai_service.RAGService", FakeRAG)
    service = AIService(model_client=MagicMock())
    assert created == []

    service.process_message("first")
    service.process_message("second")

    assert len(created) == 1


def test_generate_response_reuses_one_default_service(monkeypatch):
    """Verify generate_response() without a service reuses a single default AIService (index loaded once)."""
    import src.services.ai_service as ai_service_module

    instances = []

    class FakeAIService:
        def __init__(self):
            instances.append(self)

        def process_message(self, message):
            from src.schemas.responses import AIResponse
            return AIResponse(content=f"echo {message}")

    monkeypatch.setattr(ai_service_module, "AIService", FakeAIService)
    monkeypatch.setattr(ai_service_module, "_default_service", None)

    assert generate_response("a") == "echo a"
    assert generate_response("b") == "echo b"
    assert len(instances) == 1


def test_ai_service_does_not_touch_embeddings_or_vector_store():
    """Verify AIService delegates retrieval and never imports embedding or vector-store code itself."""
    import src.services.ai_service as ai_service_module

    names = vars(ai_service_module)
    assert not any(name in names for name in ("VectorStore", "OllamaEmbedder", "Embedder", "SEARCH_QUERY"))


def test_ai_response_list_defaults_are_independent():
    """Verify list fields use default_factory so responses never share list instances."""
    from src.schemas.responses import AIResponse, SourceRef

    first, second = AIResponse(content="a"), AIResponse(content="b")
    first.sources.append(SourceRef(source="x.md"))

    assert second.sources == [] and second.retrieved_chunks == []
    assert AIResponse.model_fields["sources"].default_factory is list
    assert AIResponse.model_fields["retrieved_chunks"].default_factory is list
