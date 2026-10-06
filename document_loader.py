import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass
class Page:
    """A unit of extracted text with the metadata needed to cite it."""

    source: str
    page: Optional[int]  # 1-based page number for PDFs; None for text/Markdown files
    text: str


@dataclass
class Document:
    """A loaded course document made up of one or more non-empty pages."""

    source: str
    pages: list[Page]


@dataclass
class SkippedFile:
    """A file that was found but could not be used, with a human-readable reason."""

    source: str
    reason: str


def normalize_text(text: str) -> str:
    """
    Applies light cleaning without removing meaningful academic content:
    unifies line endings, collapses runs of spaces/tabs, strips trailing
    whitespace on each line, and limits consecutive blank lines to one.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# A PDF page counts as fragmented when most of its lines hold a single word,
# which happens when a PDF's text layer places every word on its own line.
_FRAGMENTED_MIN_LINES = 20
_FRAGMENTED_SINGLE_WORD_SHARE = 0.8


def _is_fragmented(lines: list[str]) -> bool:
    words_per_line = [len(line.split()) for line in lines if line.strip()]
    if len(words_per_line) < _FRAGMENTED_MIN_LINES:
        return False
    single = sum(1 for count in words_per_line if count == 1)
    return single / len(words_per_line) >= _FRAGMENTED_SINGLE_WORD_SHARE


def repair_fragmented_pdf_text(text: str) -> str:
    """
    Rejoins PDF text whose words were extracted one per line, separated by a
    single blank (or whitespace-only) line. On such pages one blank line sits
    between words of the same sentence, so it becomes a space; two or more
    blank lines are kept as a paragraph break, and a plain line break stays a
    line break. Pages that are not fragmented are returned unchanged.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not _is_fragmented(lines):
        return text

    pieces: list[str] = []
    blank_run = 0
    for line in lines:
        if not line.strip():
            blank_run += 1
            continue
        if pieces:
            pieces.append(" " if blank_run == 1 else "\n\n" if blank_run >= 2 else "\n")
        pieces.append(line.strip())
        blank_run = 0
    return "".join(pieces)


def _load_pdf(path: Path, source: str) -> list[Page]:
    reader = PdfReader(path)
    if reader.is_encrypted:
        reader.decrypt("")
    pages = []
    for number, pdf_page in enumerate(reader.pages, start=1):
        text = normalize_text(repair_fragmented_pdf_text(pdf_page.extract_text() or ""))
        if text:
            pages.append(Page(source=source, page=number, text=text))
    return pages


def _load_text(path: Path, source: str) -> list[Page]:
    text = normalize_text(path.read_text(encoding="utf-8-sig"))
    return [Page(source=source, page=None, text=text)] if text else []


def load_documents(directory: str | Path) -> tuple[list[Document], list[SkippedFile]]:
    """
    Loads every supported document (.pdf, .txt, .md) under `directory`, recursively.

    Returns the loaded documents and a list of skipped files. Unsupported, empty,
    or unreadable files are reported in the skipped list instead of raising.
    Hidden files (e.g. .gitkeep) are ignored silently.

    Raises FileNotFoundError if `directory` does not exist or is not a directory.
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {root}")

    documents: list[Document] = []
    skipped: list[SkippedFile] = []

    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue

        source = relative.as_posix()
        extension = path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            skipped.append(SkippedFile(source, f"unsupported file type '{extension or '(none)'}'"))
            continue

        try:
            pages = _load_pdf(path, source) if extension == ".pdf" else _load_text(path, source)
        except UnicodeDecodeError:
            skipped.append(SkippedFile(source, "text file is not valid UTF-8"))
            continue
        except Exception as err:
            skipped.append(SkippedFile(source, f"could not be read: {err}"))
            continue

        if not pages:
            reason = (
                "no extractable text (the PDF may be scanned or image-only)"
                if extension == ".pdf"
                else "file is empty"
            )
            skipped.append(SkippedFile(source, reason))
            continue

        documents.append(Document(source=source, pages=pages))

    return documents, skipped
