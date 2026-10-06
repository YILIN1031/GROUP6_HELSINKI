from typing import Optional

from pydantic import BaseModel, Field


class UserRequest(BaseModel):
    """Minimal schema for validating incoming user input."""

    message: str = Field(..., description="The user's prompt or message.")


class SourceRef(BaseModel):
    """A course document that supplied context for an answer."""

    source: str = Field(..., description="Document path relative to the course-material directory.")
    page: Optional[int] = Field(None, description="1-based PDF page number; None for text/Markdown files.")


class RetrievedChunk(BaseModel):
    """A retrieved passage used as generation context (diagnostics; not shown to users by default)."""

    chunk_id: str
    source: str
    page: Optional[int] = None
    text: str
    score: float = Field(..., description="Cosine similarity between the question and the chunk.")


class AIResponse(BaseModel):
    """Minimal schema for structured response output from the AI service layer."""

    content: str = Field(..., description="The generated response text or user-friendly error message.")
    success: bool = Field(True, description="Flag indicating if the operation succeeded.")
    error_message: str | None = Field(None, description="Detailed error description if success is False.")
    sources: list[SourceRef] = Field(
        default_factory=list, description="Distinct documents (and pages) the answer was grounded in."
    )
    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list, description="Chunks supplied to the model as context."
    )
    retrieval_status: str | None = Field(
        None, description="RAG retrieval outcome (e.g. 'ok', 'no_relevant_context'); None if retrieval did not run."
    )
