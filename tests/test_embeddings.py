from unittest.mock import patch

import ollama
import pytest

from src.config import config
from src.models.model_client import ModelClientError, ModelNotFoundError, OllamaConnectionError
from src.rag.embeddings import (
    SEARCH_DOCUMENT,
    SEARCH_QUERY,
    Embedder,
    OllamaEmbedder,
    apply_task_prefix,
    uses_nomic_task_prefixes,
)


def fake_embed(model, input):
    """Returns a deterministic 3-dimensional vector per input text, like ollama's EmbedResponse."""
    return ollama.EmbedResponse(embeddings=[[float(len(text)), 1.0, 0.0] for text in input])


@pytest.fixture
def mock_ollama_client():
    """Replaces ollama.Client so no Ollama server is contacted; yields the fake instance."""
    with patch("ollama.Client") as client_class:
        client_class.return_value.embed.side_effect = fake_embed
        yield client_class.return_value


def test_embedder_uses_configured_model_and_host():
    """Verify the default embedder reads the base URL and embedding model from config."""
    with patch("ollama.Client") as client_class:
        embedder = OllamaEmbedder()

    client_class.assert_called_once_with(host=config.ollama_base_url)
    assert embedder.model_name == config.embedding_model


def test_embed_returns_one_vector_per_text(mock_ollama_client):
    """Verify each input text gets a float vector, in input order, using the embedding model."""
    vectors = OllamaEmbedder(model_name="nomic-embed-text").embed(["a", "bbb"])

    assert vectors == [[1.0, 1.0, 0.0], [3.0, 1.0, 0.0]]
    assert all(isinstance(value, float) for vector in vectors for value in vector)
    mock_ollama_client.embed.assert_called_once_with(model="nomic-embed-text", input=["a", "bbb"])


def test_embed_batches_requests_and_preserves_order(mock_ollama_client):
    """Verify large inputs are sent in batches and results are concatenated in order."""
    texts = ["a" * n for n in range(1, 6)]

    vectors = OllamaEmbedder(batch_size=2).embed(texts)

    assert [call.kwargs["input"] for call in mock_ollama_client.embed.call_args_list] == [
        texts[0:2], texts[2:4], texts[4:5]
    ]
    assert [vector[0] for vector in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_embed_empty_input_returns_empty_list_without_calling_ollama(mock_ollama_client):
    """Verify embedding nothing returns [] and makes no request."""
    assert OllamaEmbedder().embed([]) == []
    mock_ollama_client.embed.assert_not_called()


def test_embed_accepts_dict_response(mock_ollama_client):
    """Verify a dict-shaped embed response is also supported."""
    mock_ollama_client.embed.side_effect = None
    mock_ollama_client.embed.return_value = {"embeddings": [[0.5, 0.25]]}

    assert OllamaEmbedder().embed(["text"]) == [[0.5, 0.25]]


def test_embed_connection_failure_raises_connection_error(mock_ollama_client):
    """Verify an unreachable server raises OllamaConnectionError naming the host."""
    mock_ollama_client.embed.side_effect = ConnectionError("Failed to connect to Ollama.")

    with pytest.raises(OllamaConnectionError, match="http://example:11434"):
        OllamaEmbedder(base_url="http://example:11434").embed(["text"])


def test_embed_missing_model_raises_model_not_found(mock_ollama_client):
    """Verify a missing embedding model raises ModelNotFoundError naming that model."""
    mock_ollama_client.embed.side_effect = ollama.ResponseError("model not found", 404)

    with pytest.raises(ModelNotFoundError, match="nomic-embed-text"):
        OllamaEmbedder(model_name="nomic-embed-text").embed(["text"])


@pytest.mark.parametrize("response", [{"embeddings": [[1.0]]}, {"other": "shape"}])
def test_embed_wrong_vector_count_raises_model_client_error(mock_ollama_client, response):
    """Verify a response without exactly one vector per text raises ModelClientError."""
    mock_ollama_client.embed.side_effect = None
    mock_ollama_client.embed.return_value = response

    with pytest.raises(ModelClientError, match="expected 2 vectors"):
        OllamaEmbedder().embed(["one", "two"])


def test_invalid_batch_size_is_rejected():
    """Verify a non-positive batch size raises ValueError."""
    with patch("ollama.Client"):
        with pytest.raises(ValueError):
            OllamaEmbedder(batch_size=0)


def test_ollama_embedder_satisfies_embedder_protocol():
    """Verify OllamaEmbedder can be used wherever the Embedder interface is expected."""
    with patch("ollama.Client"):
        assert isinstance(OllamaEmbedder(), Embedder)


# ---------------------------------------------------------------- retrieval task prefixes


def sent_inputs(mock_client):
    """All texts actually sent to Ollama, across every embed call, in order."""
    return [text for call in mock_client.embed.call_args_list for text in call.kwargs["input"]]


def test_document_task_sends_search_document_prefix(mock_ollama_client):
    """Verify indexed passages are sent to a Nomic model as 'search_document: <text>'."""
    OllamaEmbedder(model_name="nomic-embed-text").embed(["Overfitting is memorisation."], task=SEARCH_DOCUMENT)

    assert sent_inputs(mock_ollama_client) == ["search_document: Overfitting is memorisation."]


def test_query_task_sends_search_query_prefix(mock_ollama_client):
    """Verify user questions are sent to a Nomic model as 'search_query: <text>'."""
    OllamaEmbedder(model_name="nomic-embed-text").embed(["What is overfitting?"], task=SEARCH_QUERY)

    assert sent_inputs(mock_ollama_client) == ["search_query: What is overfitting?"]


def test_raw_embedding_without_task_is_unchanged(mock_ollama_client):
    """Verify backward compatibility: embed(texts) with no task sends text exactly as given."""
    OllamaEmbedder(model_name="nomic-embed-text").embed(["What is overfitting?"])

    assert sent_inputs(mock_ollama_client) == ["What is overfitting?"]


def test_task_prefix_with_batching_preserves_order(mock_ollama_client):
    """Verify every batch gets the prefix and vectors still come back in input order."""
    texts = ["a" * n for n in range(1, 6)]

    vectors = OllamaEmbedder(model_name="nomic-embed-text", batch_size=2).embed(texts, task=SEARCH_DOCUMENT)

    assert mock_ollama_client.embed.call_count == 3
    assert sent_inputs(mock_ollama_client) == [f"search_document: {t}" for t in texts]
    prefix_length = len("search_document: ")
    assert [vector[0] for vector in vectors] == [float(prefix_length + n) for n in range(1, 6)]


def test_prefix_is_applied_exactly_once(mock_ollama_client):
    """Verify already-prefixed text is not prefixed again and the caller's list is not modified."""
    texts = ["search_query: already prefixed", "plain question"]
    embedder = OllamaEmbedder(model_name="nomic-embed-text")

    embedder.embed(texts, task=SEARCH_QUERY)
    embedder.embed(texts, task=SEARCH_QUERY)

    assert texts == ["search_query: already prefixed", "plain question"]
    assert sent_inputs(mock_ollama_client) == [
        "search_query: already prefixed",
        "search_query: plain question",
    ] * 2
    assert all(text.count("search_query: ") == 1 for text in sent_inputs(mock_ollama_client))


def test_non_nomic_models_receive_text_unchanged_even_with_task(mock_ollama_client):
    """Verify the Nomic prefix is not applied to unrelated embedding models."""
    OllamaEmbedder(model_name="mxbai-embed-large").embed(["What is overfitting?"], task=SEARCH_QUERY)

    assert sent_inputs(mock_ollama_client) == ["What is overfitting?"]


def test_unknown_task_is_rejected_before_calling_ollama(mock_ollama_client):
    """Verify an unsupported task name raises ValueError and makes no request."""
    with pytest.raises(ValueError, match="Unknown retrieval task"):
        OllamaEmbedder(model_name="nomic-embed-text").embed(["text"], task="classification")

    mock_ollama_client.embed.assert_not_called()


@pytest.mark.parametrize(
    "model_name, expected",
    [
        ("nomic-embed-text", True),
        ("nomic-embed-text:latest", True),
        ("nomic-embed-text:v1.5", True),
        ("nomic-embed-text-v2-moe", True),
        ("NOMIC-EMBED-TEXT", True),
        ("hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF", True),
        ("mxbai-embed-large", False),
        ("all-minilm", False),
        ("llama3.2", False),
    ],
)
def test_nomic_model_detection(model_name, expected):
    """Verify only Nomic Embed text models are treated as needing task prefixes."""
    assert uses_nomic_task_prefixes(model_name) is expected


def test_apply_task_prefix_helper():
    """Verify the centralized helper handles no task, Nomic tasks, and other models."""
    assert apply_task_prefix("nomic-embed-text", ["x"], None) == ["x"]
    assert apply_task_prefix("nomic-embed-text", ["x"], SEARCH_DOCUMENT) == ["search_document: x"]
    assert apply_task_prefix("nomic-embed-text", ["x"], SEARCH_QUERY) == ["search_query: x"]
    assert apply_task_prefix("all-minilm", ["x"], SEARCH_QUERY) == ["x"]
