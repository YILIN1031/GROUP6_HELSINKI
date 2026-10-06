from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Optional

from src.rag.document_loader import Document


@dataclass
class Chunk:
    """A retrievable piece of a document page, carrying its source metadata."""

    chunk_id: str
    source: str
    page: Optional[int]
    text: str


def _validate_parameters(size: int, overlap: int) -> None:
    if size <= 0:
        raise ValueError(f"Chunk size must be positive, got {size}.")
    if overlap < 0:
        raise ValueError(f"Chunk overlap must not be negative, got {overlap}.")
    if overlap >= size:
        raise ValueError(f"Chunk overlap ({overlap}) must be smaller than chunk size ({size}).")


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """
    Splits text into windows of at most `size` characters, where consecutive
    windows share roughly `overlap` characters.

    Window ends prefer the last whitespace in the second half of the window so
    words are not cut; a hard cut is used only when no such whitespace exists.
    Window starts are moved forward to the next word boundary.
    """
    _validate_parameters(size, overlap)
    text = text.strip()
    chunks: list[str] = []
    # A break point must leave the window longer than the overlap so each step makes progress.
    min_break = max(size // 2, overlap + 1)
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            break_at = text.rfind(" ", start + min_break, end + 1)
            break_at = max(break_at, text.rfind("\n", start + min_break, end + 1))
            if break_at != -1:
                end = break_at

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break

        next_start = end - overlap
        if next_start > 0 and not text[next_start - 1].isspace():
            boundary = min(
                (i for i in (text.find(" ", next_start, end), text.find("\n", next_start, end)) if i != -1),
                default=-1,
            )
            if boundary != -1:
                next_start = boundary + 1
        start = next_start

    return chunks


def _source_ids(documents: list[Document]) -> dict[str, str]:
    """
    Maps each source to a readable ID prefix: the path without its extension,
    with folders joined by '_' (e.g. 'week_02/notes.md' -> 'week_02_notes').
    Sources that would collide (e.g. notes.md and notes.txt) keep their extension.
    """
    bases: dict[str, list[str]] = {}
    for document in documents:
        path = PurePosixPath(document.source)
        base = "_".join(path.with_suffix("").parts)
        bases.setdefault(base, []).append(document.source)

    ids: dict[str, str] = {}
    for base, sources in bases.items():
        for source in sources:
            suffix = PurePosixPath(source).suffix.lstrip(".")
            ids[source] = f"{base}_{suffix}" if len(sources) > 1 and suffix else base
    return ids


def chunk_documents(documents: list[Document], size: int, overlap: int) -> list[Chunk]:
    """
    Splits every page of every document into overlapping chunks.

    Each chunk keeps its document's source and page number. Chunk IDs follow
    the pattern '<source>-page<N>-chunk<NN>' for PDFs and '<source>-chunk<NN>'
    for text/Markdown files, with chunks numbered from 01 within each page.
    """
    _validate_parameters(size, overlap)
    source_ids = _source_ids(documents)
    chunks: list[Chunk] = []

    for document in documents:
        prefix = source_ids[document.source]
        for page in document.pages:
            page_prefix = f"{prefix}-page{page.page}" if page.page is not None else prefix
            for number, piece in enumerate(chunk_text(page.text, size, overlap), start=1):
                chunks.append(
                    Chunk(
                        chunk_id=f"{page_prefix}-chunk{number:02d}",
                        source=page.source,
                        page=page.page,
                        text=piece,
                    )
                )

    return chunks
