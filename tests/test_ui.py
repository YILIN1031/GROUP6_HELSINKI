import ast
from pathlib import Path
from unittest.mock import MagicMock

import gradio as gr
import pytest

import app.ui as ui
import src.services.ai_service as ai_service_module
from src.rag.chunker import Chunk
from src.services.ai_service import AIService
from src.services.rag_service import RAGService, RetrievalResult, RetrievalStatus, ScoredChunk

REPO_ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_UI_IMPORTS = (
    "ollama",
    "src.models",
    "src.rag",
    "src.services.rag_service",
    "src.services.prompt_builder",
)


def use_default_service(monkeypatch, retrieval=None, answer="Grounded answer.", rag_error=None):
    """Installs a mock-backed AIService as the service layer's default, so no Ollama is needed."""
    client = MagicMock()
    client.generate.return_value = answer
    rag = MagicMock(spec=RAGService)
    if rag_error is not None:
        rag.retrieve.side_effect = rag_error
    else:
        rag.retrieve.return_value = retrieval
    monkeypatch.setattr(ai_service_module, "_default_service", AIService(model_client=client, rag_service=rag))
    return client, rag


def registered_events(demo):
    """(function, trigger component, event name) for every event registered on the Blocks app."""
    events = []
    for block_fn in demo.fns.values():
        for block_id, event_name in block_fn.targets:
            events.append((block_fn.fn, demo.blocks.get(block_id), event_name))
    return events


def test_build_ui_constructs_successfully():
    """Verify build_ui() returns a Gradio Blocks app titled CourseMate without launching a server."""
    demo = ui.build_ui()

    assert isinstance(demo, gr.Blocks)
    assert demo.title == "CourseMate"


def test_answer_output_is_markdown_and_question_input_is_textbox():
    """Verify the answer is shown in a Markdown component so the Sources list renders formatted."""
    demo = ui.build_ui()
    respond_events = [e for e in registered_events(demo) if e[0] is ui.respond]
    block_fn = next(fn for fn in demo.fns.values() if fn.fn is ui.respond)

    assert [type(c) for c in block_fn.inputs] == [gr.Textbox]
    assert [type(c) for c in block_fn.outputs] == [gr.Markdown]
    assert block_fn.outputs[0].sanitize_html is True
    assert len(respond_events) == 2


def test_button_click_and_textbox_submit_route_through_respond():
    """Verify both the Ask button and pressing Enter call the UI callback, and nothing else runs the model."""
    demo = ui.build_ui()
    events = registered_events(demo)

    triggers = {(type(component), name) for fn, component, name in events if fn is ui.respond}
    assert triggers == {(gr.Button, "click"), (gr.Textbox, "submit")}
    # The only other registered event is Gradio's built-in example loader (fills the textbox).
    other = {getattr(fn, "__name__", repr(fn)) for fn, _, _ in events if fn is not ui.respond}
    assert other <= {"load_example"}


def test_respond_delegates_to_service_layer(monkeypatch):
    """Verify the callback passes the message to generate_response and returns its text unchanged."""
    calls = []
    monkeypatch.setattr(ai_service_module, "generate_response", lambda message: calls.append(message) or "ok")

    assert ui.respond("What is overfitting?") == "ok"
    assert calls == ["What is overfitting?"]


def test_grounded_answer_renders_with_sources_section(monkeypatch):
    """Verify a successful answer is displayed with a clean Sources list and renders in Markdown."""
    chunks = [
        ScoredChunk(Chunk("lecture_04-page12-chunk01", "lecture_04.pdf", 12, "Overfitting ..."), 0.8),
        ScoredChunk(Chunk("week_02_notes-chunk01", "week_02/notes.md", None, "Networks ..."), 0.7),
    ]
    use_default_service(monkeypatch, RetrievalResult(RetrievalStatus.OK, chunks=chunks), answer="Overfitting is memorising noise.")

    text = ui.respond("What is overfitting?")
    rendered = gr.Markdown(line_breaks=True).postprocess(text)

    assert text == (
        "Overfitting is memorising noise.\n\n"
        "Sources:\n- lecture_04.pdf, page 12\n- week_02/notes.md"
    )
    assert "Sources:" in rendered and "- lecture_04.pdf, page 12" in rendered
    # No retrieval internals reach the screen.
    for internal in ("chunk01", "0.8", "0.7", "score"):
        assert internal not in text


def test_empty_input_renders_friendly_message_without_model_calls(monkeypatch):
    """Verify submitting an empty box shows the validation message and never retrieves or generates."""
    client, rag = use_default_service(monkeypatch, RetrievalResult(RetrievalStatus.OK))

    for empty in ("", "   ", None):
        text = ui.respond(empty)
        assert text == "Please enter a message before sending."
        gr.Markdown().postprocess(text)

    rag.retrieve.assert_not_called()
    client.generate.assert_not_called()


@pytest.mark.parametrize(
    "status, expected",
    [
        (RetrievalStatus.EMPTY_INDEX, "No course material has been indexed yet."),
        (RetrievalStatus.NO_RELEVANT_CONTEXT, "I could not find enough information"),
        (RetrievalStatus.INDEX_MISMATCH, "built with different settings"),
        (RetrievalStatus.INDEX_ERROR, "could not be read"),
    ],
)
def test_retrieval_states_render_user_friendly_text_only(monkeypatch, status, expected):
    """Verify each retrieval state shows its friendly message and none of the technical detail."""
    detail = "Vector store at C:/Users/x/data/vector_store is incomplete (missing metadata.json); best score 0.499824"
    client, _ = use_default_service(monkeypatch, RetrievalResult(status, detail=detail))

    text = ui.respond("What is overfitting?")
    gr.Markdown(line_breaks=True).postprocess(text)

    assert expected in text
    assert "Sources:" not in text
    for leaked in ("C:/Users", "vector_store", "metadata.json", "0.499824", "Retrieval status", status.value):
        assert leaked not in text
    client.generate.assert_not_called()


@pytest.mark.parametrize(
    "error_name, expected",
    [
        ("OllamaConnectionError", "Could not connect to Ollama"),
        ("ModelNotFoundError", "configured AI model is unavailable"),
        ("ModelClientError", "unexpected communication error"),
    ],
)
def test_ollama_and_model_errors_render_without_internals(monkeypatch, error_name, expected):
    """Verify Ollama/model failures show a friendly message, not the raw exception text."""
    import src.models.model_client as model_client

    raw = "Failed at http://localhost:11434 -- Traceback: KeyError in C:/internal/model_client.py"
    use_default_service(monkeypatch, rag_error=getattr(model_client, error_name)(raw))

    text = ui.respond("What is overfitting?")
    gr.Markdown().postprocess(text)

    assert expected in text
    for leaked in ("localhost:11434", "Traceback", "C:/internal", "KeyError"):
        assert leaked not in text


def test_examples_do_not_run_the_model_at_startup(monkeypatch):
    """Verify building the UI (including examples) makes no retrieval or generation calls."""
    client, rag = use_default_service(monkeypatch, RetrievalResult(RetrievalStatus.OK))

    ui.build_ui()

    rag.retrieve.assert_not_called()
    client.generate.assert_not_called()


def test_ui_does_not_expose_ingestion_or_upload_controls():
    """Verify the UI has no file upload or index-rebuild controls (ingestion stays a CLI step)."""
    demo = ui.build_ui()

    assert not any(isinstance(block, (gr.File, gr.UploadButton)) for block in demo.blocks.values())
    button_labels = [block.value for block in demo.blocks.values() if isinstance(block, gr.Button)]
    assert button_labels == ["Ask CourseMate"]


@pytest.mark.parametrize("relative_path", ["app/ui.py", "app/main.py"])
def test_ui_layer_has_no_infrastructure_imports(relative_path):
    """Verify the UI layer imports only Gradio and the AI service layer, never infrastructure code."""
    tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported.add(module)
            imported.update(f"{module}.{alias.name}" for alias in node.names)

    forbidden = {name for name in imported if name.startswith(FORBIDDEN_UI_IMPORTS)}
    assert forbidden == set()
    assert imported <= {"gradio", "src.services", "src.services.ai_service", "app.ui", "app.ui.build_ui"}
