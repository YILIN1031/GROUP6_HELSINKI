from typing import Optional, Protocol, Sequence, runtime_checkable

import ollama

from src.config import config
from src.models.model_client import ModelClientError, translate_ollama_error

# Retrieval roles. Callers state what the text is; each embedder decides how
# (or whether) its model needs that role encoded.
SEARCH_DOCUMENT = "search_document"  # passages stored in the index
SEARCH_QUERY = "search_query"  # user questions used to search the index
RETRIEVAL_TASKS = (SEARCH_DOCUMENT, SEARCH_QUERY)

# Nomic Embed models are trained to expect these task prefixes on every input.
_NOMIC_TASK_PREFIXES = {
    SEARCH_DOCUMENT: "search_document: ",
    SEARCH_QUERY: "search_query: ",
}


def uses_nomic_task_prefixes(model_name: str) -> bool:
    """
    True for Nomic Embed text models, e.g. 'nomic-embed-text', 'nomic-embed-text:v1.5',
    'nomic-embed-text-v2-moe', or a namespaced 'hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF'.
    """
    name = model_name.strip().lower().rsplit("/", 1)[-1]
    return name.startswith("nomic-embed-text")


def apply_task_prefix(model_name: str, texts: Sequence[str], task: Optional[str]) -> list[str]:
    """
    Returns `texts` prepared for `model_name` and retrieval `task`.

    With no task, texts are returned unchanged (raw embedding). For Nomic Embed
    models the matching prefix is added once; text that already starts with it
    is left as is. Other models receive the text unchanged.
    """
    if task is not None and task not in RETRIEVAL_TASKS:
        raise ValueError(f"Unknown retrieval task {task!r}; expected one of {RETRIEVAL_TASKS}.")
    if task is None or not uses_nomic_task_prefixes(model_name):
        return list(texts)

    prefix = _NOMIC_TASK_PREFIXES[task]
    return [text if text.startswith(prefix) else prefix + text for text in texts]


@runtime_checkable
class Embedder(Protocol):
    """
    Minimal interface the RAG pipeline depends on, so the embedding model or
    provider can be swapped (or faked in tests) without touching other code.
    """

    model_name: str

    def embed(self, texts: Sequence[str], task: Optional[str] = None) -> list[list[float]]:
        """
        Returns one embedding vector per input text, in the same order.

        `task` is SEARCH_DOCUMENT for indexed passages, SEARCH_QUERY for user
        questions, or None for raw text.
        """
        ...


class OllamaEmbedder:
    """Produces embeddings for document chunks and queries using a local Ollama model."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        batch_size: int = 32,
    ):
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}.")
        self.base_url = base_url or config.ollama_base_url
        self.model_name = model_name or config.embedding_model
        self.batch_size = batch_size
        self._client = ollama.Client(host=self.base_url)

    def embed(self, texts: Sequence[str], task: Optional[str] = None) -> list[list[float]]:
        """
        Embeds `texts` in batches and returns one vector per text, in input order.

        `task` (SEARCH_DOCUMENT or SEARCH_QUERY) applies the model's retrieval
        prefix where the model needs one (Nomic Embed); None embeds raw text.

        Raises ValueError for an unknown task, and ModelClientError subclass if
        Ollama is unreachable, the embedding model is missing, or the response
        does not contain one vector per text.
        """
        texts = apply_task_prefix(self.model_name, texts, task)
        vectors: list[list[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            try:
                response = self._client.embed(model=self.model_name, input=batch)
            except Exception as err:
                raise translate_ollama_error(err, self.model_name, self.base_url) from err

            if isinstance(response, dict):
                embeddings = response.get("embeddings")
            else:
                embeddings = getattr(response, "embeddings", None)

            if embeddings is None or len(embeddings) != len(batch):
                received = "none" if embeddings is None else len(embeddings)
                raise ModelClientError(
                    f"Unexpected embedding response from Ollama: expected {len(batch)} "
                    f"vectors, received {received}."
                )
            vectors.extend([float(value) for value in vector] for vector in embeddings)

        return vectors
