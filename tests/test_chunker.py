import pytest

from src.rag.chunker import chunk_documents, chunk_text
from src.rag.document_loader import Document, Page

WORDS = " ".join(f"word{i:03d}" for i in range(400))  # 3,199 characters, words of 7 letters


def make_document(source, page_texts, paged=True):
    pages = [
        Page(source=source, page=(number if paged else None), text=text)
        for number, text in enumerate(page_texts, start=1)
    ]
    return Document(source=source, pages=pages)


def test_normal_text_is_split_into_bounded_chunks():
    """Verify long text is split into several chunks, none longer than the chunk size."""
    chunks = chunk_text(WORDS, size=1000, overlap=200)

    assert len(chunks) >= 4
    assert all(0 < len(chunk) <= 1000 for chunk in chunks)
    assert chunks[0].startswith("word000")
    assert chunks[-1].endswith("word399")


def test_chunks_do_not_cut_words():
    """Verify chunk boundaries fall on whitespace when whitespace is available."""
    words = set(WORDS.split())

    for chunk in chunk_text(WORDS, size=250, overlap=50):
        assert all(token in words for token in chunk.split())


def test_consecutive_chunks_overlap():
    """Verify each chunk begins with text repeated from the end of the previous chunk."""
    chunks = chunk_text(WORDS, size=300, overlap=80)

    for previous, current in zip(chunks, chunks[1:]):
        shared = current[:40]
        assert shared in previous[-80:]


def test_zero_overlap_covers_text_exactly_once():
    """Verify that with no overlap every word appears exactly once across chunks."""
    chunks = chunk_text(WORDS, size=300, overlap=0)

    assert " ".join(chunks).split() == WORDS.split()


def test_all_content_is_covered_with_overlap():
    """Verify no words are lost when chunks overlap."""
    chunks = chunk_text(WORDS, size=300, overlap=100)

    covered = set(" ".join(chunks).split())
    assert covered == set(WORDS.split())


def test_text_shorter_than_one_chunk_yields_single_chunk():
    """Verify short text produces exactly one chunk equal to the stripped text."""
    assert chunk_text("  Overfitting is memorisation.  ", size=1000, overlap=200) == [
        "Overfitting is memorisation."
    ]


def test_very_short_text():
    """Verify a single-character text is still returned as one chunk."""
    assert chunk_text("A", size=1000, overlap=200) == ["A"]


def test_empty_and_whitespace_text_yield_no_chunks():
    """Verify empty or whitespace-only text produces no chunks."""
    assert chunk_text("", size=1000, overlap=200) == []
    assert chunk_text(" \n\t ", size=1000, overlap=200) == []


def test_text_without_whitespace_is_hard_cut():
    """Verify text with no spaces is still split at the size limit and fully covered."""
    text = "x" * 2500

    chunks = chunk_text(text, size=1000, overlap=200)

    assert [len(chunk) for chunk in chunks] == [1000, 1000, 900]


@pytest.mark.parametrize(
    "size, overlap",
    [(0, 0), (-10, 0), (100, -1), (100, 100), (100, 150)],
)
def test_invalid_parameters_are_rejected(size, overlap):
    """Verify non-positive sizes, negative overlaps, and overlap >= size raise ValueError."""
    with pytest.raises(ValueError):
        chunk_text("some text", size=size, overlap=overlap)
    with pytest.raises(ValueError):
        chunk_documents([], size=size, overlap=overlap)


def test_metadata_is_preserved_for_each_chunk():
    """Verify every chunk keeps the source and page number of the page it came from."""
    document = make_document("lecture_04.pdf", [WORDS, "Short page two."])

    chunks = chunk_documents([document], size=1000, overlap=200)

    page_one = [c for c in chunks if c.page == 1]
    page_two = [c for c in chunks if c.page == 2]
    assert len(page_one) >= 4
    assert [c.text for c in page_two] == ["Short page two."]
    assert all(c.source == "lecture_04.pdf" for c in chunks)


def test_chunk_ids_follow_readable_pattern():
    """Verify chunk IDs look like 'lecture_04-page12-chunk02' and 'notes-chunk01'."""
    pdf = Document(
        source="lecture_04.pdf",
        pages=[Page(source="lecture_04.pdf", page=12, text=WORDS)],
    )
    notes = make_document("notes.md", ["Plain notes."], paged=False)

    chunks = chunk_documents([pdf, notes], size=1000, overlap=200)

    assert chunks[0].chunk_id == "lecture_04-page12-chunk01"
    assert chunks[1].chunk_id == "lecture_04-page12-chunk02"
    assert chunks[-1].chunk_id == "notes-chunk01"
    assert chunks[-1].page is None


def test_chunk_ids_are_unique_across_colliding_sources():
    """Verify IDs stay unique for same-named files in different folders or with different extensions."""
    documents = [
        make_document("notes.md", [WORDS], paged=False),
        make_document("notes.txt", [WORDS], paged=False),
        make_document("week_02/notes.md", [WORDS], paged=False),
        make_document("week_03/notes.md", [WORDS], paged=False),
        make_document("lecture.pdf", [WORDS, WORDS]),
    ]

    chunks = chunk_documents(documents, size=500, overlap=100)
    ids = [c.chunk_id for c in chunks]

    assert len(ids) == len(set(ids))
    assert any(i.startswith("notes_md-") for i in ids)
    assert any(i.startswith("notes_txt-") for i in ids)
    assert any(i.startswith("week_02_notes-") for i in ids)


def test_empty_document_list_yields_no_chunks():
    """Verify chunking nothing returns an empty list."""
    assert chunk_documents([], size=1000, overlap=200) == []
