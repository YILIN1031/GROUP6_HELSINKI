from typing import Optional
import ollama
from src.config import config


class ModelClientError(Exception):
    """Base exception for model client failures."""
    pass


class OllamaConnectionError(ModelClientError):
    """Exception raised when Ollama service is unreachable."""
    pass


class ModelNotFoundError(ModelClientError):
    """Exception raised when the requested model is not found on local Ollama."""
    pass


def translate_ollama_error(err: Exception, model_name: str, base_url: str) -> ModelClientError:
    """
    Converts a low-level Ollama/transport exception into a ModelClientError subclass.

    Shared by the generation client and the embedding client. Typed checks come
    first (the ollama library raises ConnectionError when the server is unreachable
    and ResponseError with status 404 for a missing model); message matching is
    kept as a fallback for other error types. The caller raises the result `from err`.
    """
    if isinstance(err, ConnectionError):
        return OllamaConnectionError(f"Failed to connect to Ollama service at {base_url}.")
    if isinstance(err, ollama.ResponseError) and err.status_code == 404:
        return ModelNotFoundError(f"Model '{model_name}' is not installed in local Ollama.")

    err_str = str(err).lower()
    if "connection" in err_str or "connect" in err_str or "refused" in err_str:
        return OllamaConnectionError(f"Failed to connect to Ollama service at {base_url}.")
    if "not found" in err_str or "404" in err_str:
        return ModelNotFoundError(f"Model '{model_name}' is not installed in local Ollama.")
    return ModelClientError(f"Error communicating with Ollama: {err}")


class OllamaModelClient:
    """Encapsulates interaction with local Ollama inference server."""

    def __init__(self, base_url: Optional[str] = None, model_name: Optional[str] = None):
        self.base_url = base_url or config.ollama_base_url
        self.model_name = model_name or config.model_name
        self._client = ollama.Client(host=self.base_url)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Sends a user prompt to local Ollama model and returns generated response text.

        If `system_prompt` is given, it is sent as a separate system message before
        the user message, keeping application instructions apart from user content.

        Raises ModelClientError subclass if connection or execution fails.
        """
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat(
                model=self.model_name,
                messages=messages,
            )

            # Standardize extraction from dict or chat response object
            if isinstance(response, dict):
                return response.get("message", {}).get("content", "")
            elif hasattr(response, "message") and hasattr(response.message, "content"):
                return response.message.content
            return str(response)

        except Exception as err:
            raise translate_ollama_error(err, self.model_name, self.base_url) from err
