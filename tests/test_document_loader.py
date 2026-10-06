import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from src.rag.document_loader import load_documents, normalize_text, repair_fragmented_pdf_text


def make_pdf(path, page_texts):
    """Writes a PDF with one page per entry; None produces a page with no text (like a scan)."""
    writer = PdfWriter()
    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        if text is None:
            continue
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
        })
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1"))
        page.replace_contents(stream)
    writer.write(path)


def skipped_reasons(skipped):
    return {item.source: item.reason for item in skipped}


def test_loads_txt_file(tmp_path):
    """Verify a text file becomes one document with a single page and no page number."""
    (tmp_path / "notes.txt").write_text("Supervised learning uses labelled data.", encoding="utf-8")

    documents, skipped = load_documents(tmp_path)

    assert skipped == []
    assert len(documents) == 1
    document = documents[0]
    assert document.source == "notes.txt"
    assert len(document.pages) == 1
    assert document.pages[0].page is None
    assert document.pages[0].source == "notes.txt"
    assert document.pages[0].text == "Supervised learning uses labelled data."


def test_loads_markdown_file(tmp_path):
    """Verify Markdown files are loaded as text, keeping their content."""
    (tmp_path / "revision.md").write_text("# Overfitting\n\nA model memorises noise.", encoding="utf-8")

    documents, skipped = load_documents(tmp_path)

    assert skipped == []
    assert [d.source for d in documents] == ["revision.md"]
    assert documents[0].pages[0].text == "# Overfitting\n\nA model memorises noise."


def test_loads_pdf_with_page_numbers(tmp_path):
    """Verify PDF pages keep 1-based page numbers and text-less pages are dropped."""
    make_pdf(tmp_path / "lecture_04.pdf", ["Overfitting occurs early.", None, "Regularisation helps."])

    documents, skipped = load_documents(tmp_path)

    assert skipped == []
    assert len(documents) == 1
    pages = documents[0].pages
    assert [p.page for p in pages] == [1, 3]
    assert [p.source for p in pages] == ["lecture_04.pdf", "lecture_04.pdf"]
    assert pages[0].text == "Overfitting occurs early."
    assert pages[1].text == "Regularisation helps."


def test_pdf_without_extractable_text_is_skipped(tmp_path):
    """Verify an image-only style PDF is reported as skipped rather than loaded empty."""
    make_pdf(tmp_path / "scanned.pdf", [None, None])

    documents, skipped = load_documents(tmp_path)

    assert documents == []
    assert "no extractable text" in skipped_reasons(skipped)["scanned.pdf"]


def test_empty_and_whitespace_files_are_skipped(tmp_path):
    """Verify empty or whitespace-only text files are skipped with a reason."""
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    (tmp_path / "blank.md").write_text("   \n\n\t  \n", encoding="utf-8")

    documents, skipped = load_documents(tmp_path)

    assert documents == []
    reasons = skipped_reasons(skipped)
    assert reasons == {"blank.md": "file is empty", "empty.txt": "file is empty"}


def test_unsupported_extension_is_skipped(tmp_path):
    """Verify unsupported file types are reported, not loaded, and do not stop loading."""
    (tmp_path / "slides.pptx").write_bytes(b"binary")
    (tmp_path / "notes.txt").write_text("Valid content.", encoding="utf-8")

    documents, skipped = load_documents(tmp_path)

    assert [d.source for d in documents] == ["notes.txt"]
    assert "unsupported file type '.pptx'" in skipped_reasons(skipped)["slides.pptx"]


def test_missing_directory_raises_clear_error(tmp_path):
    """Verify a missing documents directory raises FileNotFoundError naming the path."""
    missing = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError, match="does_not_exist"):
        load_documents(missing)


def test_corrupt_pdf_is_skipped_without_raising(tmp_path):
    """Verify a corrupt PDF is reported as skipped while other files still load."""
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4 this is not a real pdf")
    (tmp_path / "notes.txt").write_text("Still loaded.", encoding="utf-8")

    documents, skipped = load_documents(tmp_path)

    assert [d.source for d in documents] == ["notes.txt"]
    assert skipped_reasons(skipped)["broken.pdf"].startswith("could not be read")


def test_non_utf8_text_file_is_skipped(tmp_path):
    """Verify a text file with invalid UTF-8 bytes is skipped with a clear reason."""
    (tmp_path / "legacy.txt").write_bytes(b"caf\xe9 \xff\xfe")

    documents, skipped = load_documents(tmp_path)

    assert documents == []
    assert skipped_reasons(skipped) == {"legacy.txt": "text file is not valid UTF-8"}


def test_hidden_files_are_ignored_and_subdirectories_are_loaded(tmp_path):
    """Verify .gitkeep-style files are ignored silently and nested files use relative sources."""
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
    (tmp_path / "week_02").mkdir()
    (tmp_path / "week_02" / "notes.md").write_text("Neural networks.", encoding="utf-8")

    documents, skipped = load_documents(tmp_path)

    assert skipped == []
    assert [d.source for d in documents] == ["week_02/notes.md"]


def test_documents_are_returned_in_sorted_order(tmp_path):
    """Verify loading order is deterministic regardless of file creation order."""
    for name in ["c.txt", "a.txt", "b.md"]:
        (tmp_path / name).write_text(f"Content of {name}", encoding="utf-8")

    documents, _ = load_documents(tmp_path)

    assert [d.source for d in documents] == ["a.txt", "b.md", "c.txt"]


def test_normalize_text_cleans_whitespace_but_keeps_content():
    """Verify normalisation collapses spacing and blank lines without dropping words."""
    raw = "  Overfitting\t\toccurs   when\r\n\r\n\r\n\r\na model   \nmemorises\xa0noise.  "

    assert normalize_text(raw) == "Overfitting occurs when\n\na model\nmemorises noise."


# ---------------------------------------------------------------- fragmented PDF text repair

def fragmented(words, separator="\n \n"):
    """Mimics a PDF text layer that emits one word per line, separated by whitespace-only lines."""
    return separator.join(words)


SENTENCE_1 = "Writing on the exam takes fifty five minutes for three texts in total today".split()
SENTENCE_2 = "The opinion text is the longest one and should have about one hundred twenty words".split()


def test_fragmented_words_are_joined_into_sentences():
    """Verify single whitespace-only lines between words become spaces."""
    text = fragmented(SENTENCE_1 + SENTENCE_2)

    assert repair_fragmented_pdf_text(text) == " ".join(SENTENCE_1 + SENTENCE_2)


def test_fragmented_text_keeps_real_paragraph_breaks():
    """Verify two or more blank lines on a fragmented page remain a paragraph break."""
    text = fragmented(SENTENCE_1) + "\n \n \n" + fragmented(SENTENCE_2)

    repaired = normalize_text(repair_fragmented_pdf_text(text))

    assert repaired == " ".join(SENTENCE_1) + "\n\n" + " ".join(SENTENCE_2)


def test_fragmented_text_keeps_plain_line_breaks():
    """Verify a direct line break (no blank line) is kept, e.g. between a date and a heading."""
    text = fragmented(SENTENCE_1) + "\nOct 13, 2025\nSuperintensive YKI November\n" + fragmented(SENTENCE_2)

    repaired = repair_fragmented_pdf_text(text)

    assert repaired == (
        " ".join(SENTENCE_1) + "\nOct 13, 2025\nSuperintensive YKI November\n" + " ".join(SENTENCE_2)
    )


def test_normal_prose_page_is_unchanged():
    """Verify ordinary PDF text, where one blank line is a real paragraph break, is not altered."""
    text = (
        "EN ANNONS \n \nSnygg och bekväm lägenhet söker pålitlig hyresgäst \n \nHej! \n \n"
        "Jag har en fin lägenhet som är tillgänglig för uthyrning och letar efter en pålitlig hyresgäst.\n"
    ) * 5

    assert repair_fragmented_pdf_text(text) == text
    assert "\n\nSnygg och bekväm" in normalize_text(repair_fragmented_pdf_text(text))


def test_short_page_is_never_treated_as_fragmented():
    """Verify pages with too few lines (e.g. a title page) are left alone."""
    text = fragmented(["Förbered", "dig", "för", "allmän", "språkexamen"])

    assert repair_fragmented_pdf_text(text) == text


def test_mostly_multi_word_lines_are_not_treated_as_fragmented():
    """Verify a page is only repaired when at least 80% of its lines are single words."""
    lines = SENTENCE_1[:15] + ["two words"] * 10  # 15 of 25 lines single-word = 60%
    text = fragmented(lines)

    assert repair_fragmented_pdf_text(text) == text


def test_loader_repairs_fragmented_pdf_pages(tmp_path):
    """Verify load_documents applies the repair to PDF pages and preserves the page number."""
    make_pdf(tmp_path / "notes.pdf", ["Intro page", "Second page"])
    import src.rag.document_loader as loader

    class FakePage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class FakeReader:
        is_encrypted = False

        def __init__(self, path):
            self.pages = [FakePage("Short title"), FakePage(fragmented(SENTENCE_1 + SENTENCE_2))]

    original = loader.PdfReader
    loader.PdfReader = FakeReader
    try:
        documents, skipped = load_documents(tmp_path)
    finally:
        loader.PdfReader = original

    assert skipped == []
    pages = documents[0].pages
    assert [p.page for p in pages] == [1, 2]
    assert pages[1].text == " ".join(SENTENCE_1 + SENTENCE_2)


def test_text_files_are_not_affected_by_pdf_repair(tmp_path):
    """Verify .txt/.md files keep their line structure even if they look fragmented."""
    content = "\n\n".join(SENTENCE_1 + SENTENCE_2)
    (tmp_path / "vocab.txt").write_text(content, encoding="utf-8")

    documents, _ = load_documents(tmp_path)

    assert documents[0].pages[0].text == content
