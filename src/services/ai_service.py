from typing import Optional
from src.models.model_client import (
    OllamaModelClient,
    ModelClientError,
    OllamaConnectionError,
    ModelNotFoundError,
)
from src.schemas.responses import UserRequest, AIResponse, RetrievedChunk, SourceRef
from src.services.prompt_builder import SYSTEM_PROMPT, build_user_prompt, format_source_label
from src.services.rag_service import RAGService, RetrievalResult, RetrievalStatus, ScoredChunk

REBUILD_HINT = "Please rebuild it by running 'python -m src.rag.ingest', then restart CourseMate."

# User-facing messages for retrieval outcomes that must not reach the generation model.
RETRIEVAL_MESSAGES = {
    RetrievalStatus.EMPTY_INDEX: (
        "No course material has been indexed yet.\n\n"
        "Add PDF, TXT, or Markdown course documents, run 'python -m src.rag.ingest', "
        "then restart CourseMate."
    ),
    RetrievalStatus.NO_RELEVANT_CONTEXT: (
        "I could not find enough information in the indexed course materials "
        "to answer that question reliably."
    ),
    RetrievalStatus.INDEX_MISMATCH: (
        "[Error] The course-material index was built with different settings than CourseMate "
        f"is currently using.\n\n{REBUILD_HINT}"
    ),
    RetrievalStatus.INDEX_ERROR: (
        f"[Error] The course-material index could not be read.\n\n{REBUILD_HINT}"
    ),
}


def _sources_from_chunks(chunks: list[ScoredChunk]) -> list[SourceRef]:
    """Distinct (document, page) pairs in retrieval order, taken only from retrieved chunks."""
    seen = set()
    sources = []
    for item in chunks:
        key = (item.chunk.source, item.chunk.page)
        if key not in seen:
            seen.add(key)
            sources.append(SourceRef(source=item.chunk.source, page=item.chunk.page))
    return sources


def _retrieved_chunks(chunks: list[ScoredChunk]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=item.chunk.chunk_id,
            source=item.chunk.source,
            page=item.chunk.page,
            text=item.chunk.text,
            score=item.score,
        )
        for item in chunks
    ]


class AIService:
    """
    Application service layer responsible for validating user input,
    orchestrating retrieval and model client requests, and catching
    exceptions gracefully.
    """

    def __init__(
        self,
        model_client: Optional[OllamaModelClient] = None,
        rag_service: Optional[RAGService] = None,
    ):
        # Allow injecting custom/mock model_client and rag_service for simple testing
        self.model_client = model_client
        self.rag_service = rag_service

    def _get_client(self) -> OllamaModelClient:
        """Returns active model client, initializing default client if none provided."""
        if self.model_client is None:
            self.model_client = OllamaModelClient()
        return self.model_client

    def _get_rag_service(self) -> RAGService:
        """Returns active RAG service, initializing the configured default if none provided."""
        if self.rag_service is None:
            self.rag_service = RAGService()
        return self.rag_service

    def _retrieval_response(self, retrieval: RetrievalResult) -> AIResponse:
        """Converts a non-OK retrieval result into a user-facing response (no generation)."""
        return AIResponse(
            content=RETRIEVAL_MESSAGES[retrieval.status],
            success=False,
            error_message=f"Retrieval status '{retrieval.status.value}': {retrieval.detail}",
            retrieval_status=retrieval.status.value,
        )

    def process_message(self, user_message: str) -> AIResponse:
        """
        Processes a raw user message string and returns a structured AIResponse.
        Catches technical failures and converts them to friendly user-facing messages.
        """
        # 1. Validate empty input
        if not user_message or not user_message.strip():
            return AIResponse(
                content="Please enter a message before sending.",
                success=False,
                error_message="User message was empty.",
            )

        try:
            # 2. Schema validation
            request = UserRequest(message=user_message.strip())

            # 3. Retrieve course context; generation only runs on usable context
            retrieval = self._get_rag_service().retrieve(request.message)
            if not retrieval.ok:
                return self._retrieval_response(retrieval)

            # 4. Grounded generation from the retrieved chunks
            prompt = build_user_prompt(request.message, retrieval.chunks)
            client = self._get_client()
            response_text = client.generate(prompt, system_prompt=SYSTEM_PROMPT)

            # 5. Sources come from the retrieved chunks, never from model output
            return AIResponse(
                content=response_text,
                success=True,
                sources=_sources_from_chunks(retrieval.chunks),
                retrieved_chunks=_retrieved_chunks(retrieval.chunks),
                retrieval_status=retrieval.status.value,
            )

        except OllamaConnectionError as err:
            return AIResponse(
                content=(
                    "[Error] Could not connect to Ollama.\n\n"
                    "Please verify that Ollama is installed and running locally on your machine."
                ),
                success=False,
                error_message=str(err),
            )

        except ModelNotFoundError as err:
            return AIResponse(
                content=(
                    "[Error] The configured AI model is unavailable in Ollama.\n\n"
                    "Please verify your MODEL_NAME and EMBEDDING_MODEL settings, and install "
                    "missing models with 'ollama pull <model_name>'."
                ),
                success=False,
                error_message=str(err),
            )

        except ModelClientError as err:
            return AIResponse(
                content="[Error] An unexpected communication error occurred with the AI model.",
                success=False,
                error_message=str(err),
            )

        except Exception as err:
            return AIResponse(
                content="[Error] An unexpected application error occurred.",
                success=False,
                error_message=str(err),
            )


def format_response(response: AIResponse) -> str:
    """Renders an AIResponse as display text: the content, then a Sources list when present."""
    if not response.sources:
        return response.content
    lines = [f"- {format_source_label(ref.source, ref.page)}" for ref in response.sources]
    return f"{response.content}\n\nSources:\n" + "\n".join(lines)


_default_service: Optional[AIService] = None


def _get_default_service() -> AIService:
    """Creates the default AIService once so the index is loaded once per application run."""
    global _default_service
    if _default_service is None:
        _default_service = AIService()
    return _default_service


def generate_response(user_message: str, service: Optional[AIService] = None) -> str:
    """
    Main reusable service entry point used by the UI layer.

    Accepts user input message, passes it to the AI service, and returns
    the generated text response with its sources (or a friendly error message).
    """
    active_service = service or _get_default_service()
    response = active_service.process_message(user_message)
    return format_response(response)
