from unittest.mock import patch

import ollama
import pytest

from src.config import config
from src.models.model_client import (
    ModelClientError,
    ModelNotFoundError,
    OllamaConnectionError,
    OllamaModelClient,
    translate_ollama_error,
)


@pytest.fixture
def mock_ollama_client():
    """Replaces ollama.Client so no Ollama server is contacted; yields the fake instance."""
    with patch("ollama.Client") as client_class:
        yield client_class.return_value


def test_client_uses_configured_model_and_host():
    """Verify the default client reads the base URL and model name from config."""
    with patch("ollama.Client") as client_class:
        client = OllamaModelClient()

    client_class.assert_called_once_with(host=config.ollama_base_url)
    assert client.model_name == config.model_name


def test_generate_with_dict_response(mock_ollama_client):
    """Verify text is extracted from a dict-shaped chat response."""
    mock_ollama_client.chat.return_value = {"message": {"role": "assistant", "content": "Hi there"}}

    assert OllamaModelClient(model_name="llama3.2").generate("Hello") == "Hi there"


def test_generate_with_chat_response_object(mock_ollama_client):
    """Verify text is extracted from the ollama library's ChatResponse object."""
    mock_ollama_client.chat.return_value = ollama.ChatResponse(
        message=ollama.Message(role="assistant", content="Object reply")
    )

    assert OllamaModelClient().generate("Hello") == "Object reply"


def test_generate_without_system_prompt_sends_only_user_message(mock_ollama_client):
    """Verify backward compatibility: without a system prompt only the user message is sent."""
    mock_ollama_client.chat.return_value = {"message": {"content": "ok"}}

    OllamaModelClient(model_name="llama3.2").generate("What is overfitting?")

    mock_ollama_client.chat.assert_called_once_with(
        model="llama3.2",
        messages=[{"role": "user", "content": "What is overfitting?"}],
    )


def test_generate_with_system_prompt_sends_system_then_user(mock_ollama_client):
    """Verify a system prompt is sent as a separate system message before the user message."""
    mock_ollama_client.chat.return_value = {"message": {"content": "ok"}}

    OllamaModelClient(model_name="llama3.2").generate("Question", system_prompt="Rules")

    mock_ollama_client.chat.assert_called_once_with(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "Rules"},
            {"role": "user", "content": "Question"},
        ],
    )


def test_unreachable_server_raises_connection_error(mock_ollama_client):
    """Verify the ollama library's ConnectionError becomes OllamaConnectionError naming the host."""
    original = ConnectionError("Failed to connect to Ollama.")
    mock_ollama_client.chat.side_effect = original

    with pytest.raises(OllamaConnectionError, match="http://example:11434") as excinfo:
        OllamaModelClient(base_url="http://example:11434").generate("Hello")

    assert excinfo.value.__cause__ is original


def test_connection_refused_message_fallback(mock_ollama_client):
    """Verify untyped errors mentioning a refused connection still map to OllamaConnectionError."""
    mock_ollama_client.chat.side_effect = RuntimeError("[WinError 10061] connection refused")

    with pytest.raises(OllamaConnectionError):
        OllamaModelClient().generate("Hello")


def test_missing_model_404_raises_model_not_found(mock_ollama_client):
    """Verify a 404 ResponseError becomes ModelNotFoundError naming the configured model."""
    mock_ollama_client.chat.side_effect = ollama.ResponseError("model 'llama9' not found", 404)

    with pytest.raises(ModelNotFoundError, match="llama9"):
        OllamaModelClient(model_name="llama9").generate("Hello")


def test_not_found_message_fallback(mock_ollama_client):
    """Verify errors whose message says 'not found' still map to ModelNotFoundError."""
    mock_ollama_client.chat.side_effect = ollama.ResponseError("model not found")

    with pytest.raises(ModelNotFoundError):
        OllamaModelClient().generate("Hello")


def test_other_server_errors_raise_generic_model_client_error(mock_ollama_client):
    """Verify unrelated server failures become a plain ModelClientError with the original detail."""
    mock_ollama_client.chat.side_effect = ollama.ResponseError("out of memory", 500)

    with pytest.raises(ModelClientError, match="out of memory") as excinfo:
        OllamaModelClient().generate("Hello")

    assert type(excinfo.value) is ModelClientError


def test_unexpected_response_shapes_keep_existing_behavior(mock_ollama_client):
    """Verify unexpected response shapes are handled as before: missing content -> '', other -> str()."""
    client = OllamaModelClient()

    mock_ollama_client.chat.return_value = {"unexpected": "shape"}
    assert client.generate("Hello") == ""

    mock_ollama_client.chat.return_value = 12345
    assert client.generate("Hello") == "12345"


def test_translate_ollama_error_mapping():
    """Verify the shared helper maps each error family to the right exception and message."""
    base_url, model = "http://localhost:11434", "nomic-embed-text"

    connection = translate_ollama_error(ConnectionError("down"), model, base_url)
    missing = translate_ollama_error(ollama.ResponseError("missing", 404), model, base_url)
    other = translate_ollama_error(ValueError("boom"), model, base_url)

    assert isinstance(connection, OllamaConnectionError) and base_url in str(connection)
    assert isinstance(missing, ModelNotFoundError) and model in str(missing)
    assert type(other) is ModelClientError and "boom" in str(other)
