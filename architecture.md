# Architecture

This document describes CourseMate's architecture as implemented. CourseMate extends the course starter's layered design (UI → service → model client → Ollama) with a Retrieval-Augmented Generation (RAG) layer.

## Overview

```text
                     OFFLINE: DOCUMENT INGESTION  (python -m src.rag.ingest)

  data/documents/*.pdf|.txt|.md
          │
          ▼
  src/rag/document_loader.py   extract text (PDF per page) · normalise · skip unusable files
          │
          ▼
  src/rag/chunker.py           overlapping character windows · chunk IDs · source + page
          │
          ▼
  src/rag/embeddings.py        OllamaEmbedder · nomic-embed-text · task "search_document"
          │
          ▼
  src/rag/vector_store.py      NumPy index → data/vector_store/{embeddings.npy, chunks.json, metadata.json}


                     ONLINE: QUESTION ANSWERING  (python -m app.main)

  User
    │
    ▼
  app/ui.py                    Gradio: question box, "Ask CourseMate", Markdown answer + Sources
    │  generate_response(message)
    ▼
  src/services/ai_service.py   validate input · orchestrate · map errors · build Sources
    │  retrieve(question)
    ▼
  src/services/rag_service.py  load index once · compatibility check · embed question ("search_query")
    │                          · cosine search · top-k · min score → RetrievalResult(status, chunks)
    │  retrieved chunks (status "ok")
    ▼
  src/services/prompt_builder.py   SYSTEM_PROMPT + delimited COURSE MATERIAL + USER QUESTION
    │
    ▼
  src/models/model_client.py   OllamaModelClient.generate(prompt, system_prompt)
    │
    ▼
  Ollama · llama3.2
    │
    ▼
  AIResponse(content, sources, retrieved_chunks, retrieval_status, …) → formatted answer + "Sources:" → UI
```

## Core design rules

1. **Strict UI isolation.** `app/ui.py` calls only `generate_response()` in `src/services/ai_service.py`. It never imports the model client, embeddings, vector store or ingestion code; `tests/test_ui.py` checks this.
2. **Controlled failure handling.** Low-level Ollama errors are translated in one place (`translate_ollama_error()` in `src/models/model_client.py`), which is shared by generation and embeddings. `AIService` turns every error into a user-facing message, and technical detail stays in `error_message`.
3. **No generation without usable context.** Every retrieval status other than `ok` returns a fixed message without calling the generation model.
4. **Sources come from code.** The Sources list is built from the retrieved chunks, never from the model's output.
5. **Schemas.** Requests and responses are Pydantic models in `src/schemas/responses.py`. List fields use `Field(default_factory=list)`.
6. **Configuration.** All settings are read from environment variables or `.env` in `src/config.py`.

## Ingestion components

### Document loading and extraction (`src/rag/document_loader.py`)

- Supports `.pdf` (read page by page with pypdf, keeping 1-based page numbers), `.txt` and `.md` (UTF-8, with no page numbers).
- Reads the documents folder recursively. Hidden files such as `.gitkeep` are ignored.
- **Normalisation:** unifies line endings, collapses runs of spaces, trims whitespace around line breaks, and allows at most one blank line in a row.
- **Fragmented-PDF repair:** some PDFs place every word on its own line. A page counts as fragmented when it has at least 20 lines and 80% or more are single words. On such pages words are rejoined, while two or more blank lines stay a paragraph break. Other pages are unchanged.
- **Unusable files** (unsupported type, empty, corrupt, image-only, not UTF-8) are listed as skipped with a reason. They never stop the run.

### Chunking (`src/rag/chunker.py`)

- Splits each page into windows of about `RAG_CHUNK_SIZE` characters (default 1000), with about `RAG_CHUNK_OVERLAP` characters of overlap (default 200).
- Window ends prefer whitespace, so words aren't cut; a hard cut is used only when there is none.
- Each chunk keeps its `source` and `page`. Chunk IDs look like `lecture_04-page12-chunk02` for PDFs and `notes-chunk01` for text files, and they stay unique across documents with the same name.

### Embeddings (`src/rag/embeddings.py`)

- `Embedder` is a small interface: `embed(texts, task=None)`. `OllamaEmbedder` is the implementation. It uses `nomic-embed-text` by default and sends requests in batches of 32.
- **Retrieval tasks:** `SEARCH_DOCUMENT` for indexed passages and `SEARCH_QUERY` for questions. For Nomic Embed models the matching prefix (`search_document:` / `search_query:`) is added exactly once. Other models get the text unchanged.

### Vector store (`src/rag/vector_store.py`)

A local NumPy store. The index is saved in `VECTOR_STORE_PATH` (default `data/vector_store/`) as three files:

| File | Contents |
|---|---|
| `embeddings.npy` | float32 matrix, one row per chunk, in chunk order |
| `chunks.json` | chunk records only (`chunk_id`, `source`, `page`, `text`) |
| `metadata.json` | `format_version`, `embedding_model`, `embedding_dimension`, `document_embedding_task`, `chunk_count` |

- **Search:** cosine similarity over normalised vectors.
- **Validation on load:** row count against chunk count, vector width against the recorded dimension, required metadata keys, record fields, duplicate IDs, and NaN or zero vectors.
  - A missing index raises `VectorStoreNotFoundError`.
  - A corrupt or inconsistent index raises `VectorStoreError`.
- **Compatibility check:** `compatibility_issue(model, task)` reports an index built with another embedding model or document task, or an index without a task marker.

### Ingestion command (`src/rag/ingest.py`)

`python -m src.rag.ingest` runs load → chunk → embed → store → save.
- **Atomic result:** the index is written only after every step succeeds, so a failed run never leaves a partial index.
- **Output:** the command reports the document and chunk counts and the skipped files, and ends with a reminder to restart the application.
- **Errors:** a missing folder, no usable documents, invalid chunk settings, Ollama being unreachable and a missing embedding model each produce a clear message with exit code 1.

## Question-answering components

### RAG service (`src/services/rag_service.py`)

- **Loading:** the index is loaded once, the first time it's needed, and kept for the life of the service. `generate_response()` reuses one default service, so **the application must be restarted after re-ingesting**. There is no hot reload.
- **Retrieval steps:** check compatibility → embed the question with `search_query` → search → keep up to `RAG_TOP_K` chunks scoring at least `RAG_MIN_SCORE` (default 0.5, provisional).
- **Statuses:**

| Status | Meaning |
|---|---|
| `ok` | at least one chunk reached the minimum score |
| `empty_index` | no index exists, or it contains no chunks |
| `no_relevant_context` | no chunk reached `RAG_MIN_SCORE` |
| `index_mismatch` | the index was built with another embedding model or document task, predates the task marker, or has a different dimension from the query |
| `index_error` | the index files exist but are corrupt or inconsistent |

- **Errors:** embedding failures (Ollama unreachable, missing model) are passed up to `AIService`.

### Prompt builder (`src/services/prompt_builder.py`)

- `SYSTEM_PROMPT` holds CourseMate's grounding rules: prefer the material, don't invent course facts, say when the material is insufficient, cite only the document names given, and treat the material as reference text rather than instructions.
- `build_user_prompt()` numbers the chunks as `SOURCE n` blocks with `Document:` and, for PDFs, `Page:` lines. The blocks go between `=== COURSE MATERIAL START/END ===` markers, followed by the question between `=== USER QUESTION START/END ===`.
- Marker strings are removed from retrieved text and from the question, so they can't fake a section boundary.

### Generation (`src/models/model_client.py`)

- `OllamaModelClient.generate(prompt, system_prompt=None)` sends a system message (when given) and a user message to `MODEL_NAME` (default `llama3.2`).
- Generation uses Ollama's default sampling.

### AI service (`src/services/ai_service.py`)

- **Input:** empty input is rejected before any retrieval.
- **Retrieval outcome:** a non-`ok` retrieval returns its fixed message.
- **Answer:** otherwise the service builds the prompt, calls the model client, and returns an `AIResponse` whose `sources` are the retrieved chunks' distinct (document, page) pairs, in order.
- **Errors:** connection, missing-model and other model errors each map to a fixed user message.
- **Output format:** `format_response()` appends a `Sources:` list to successful answers.

### Gradio UI (`app/ui.py`)

- **Layout:** a question textbox, an **Ask CourseMate** button (pressing Enter also works), a Markdown output with HTML sanitisation, and example-question buttons.
- **Examples:** the example-question buttons hold Swedish/YKI-oriented questions. Clicking one only fills in the question box; it never runs the model on its own.
- **Launch:** `python -m app.main` starts the app.

## Offline evaluation workflow (separate from the application)

The evaluation runs offline and never changes the application:

```text
evaluation/PROTOCOL.md + test_cases.json (frozen)
        │
        ▼
evaluation/run_eval.py   runs each case through the real AIService with recording wrappers
        │                (3 runs per generated-answer case, 1 per deterministic failure case)
        ▼
evaluation/runs/<timestamp>/   manifest.json · runs.jsonl · cases.json   (git-ignored, local)
        │
        ▼
human grading → evaluation/evaluation_results.md
```

- **Injection case F08:** it uses a separate evaluation-only index (the corpus plus `evaluation/fixtures/f08_writing_test_tips.md`), built in a temporary folder outside the repository. The main index is never modified.
- **Harness limits:** the harness checks preconditions (frozen settings, a clean tree, index contamination, the fixture hash) and records mechanical facts only. It never assigns quality labels.
