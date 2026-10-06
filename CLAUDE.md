# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CourseMate — a local RAG study assistant for course documents (Python 3.12, Gradio UI, local inference via Ollama: `llama3.2` for generation, `nomic-embed-text` for embeddings). The RAG pipeline is fully implemented (`src/rag/`, `src/services/`), and a frozen, human-graded evaluation has been completed (`evaluation/`). The evaluation corpus is local Swedish/YKI exam-preparation material that is not in the repository.

## Commands

```bash
conda env create -f environment.yml   # env name: dev-ai-project
conda activate dev-ai-project
copy .env.example .env                # Windows PowerShell
cp .env.example .env                  # Linux/macOS
ollama pull llama3.2
ollama pull nomic-embed-text

python -m src.rag.ingest              # build the index from DOCUMENTS_PATH into VECTOR_STORE_PATH
python -m app.main                    # launches Gradio at http://localhost:7860 (run from repo root)
pytest                                # all tests
pytest tests/test_ai_service.py::test_empty_input_validation   # single test
```

- Restart `python -m app.main` after re-running ingestion: the running app loads the index once.
- On consoles that cannot print Swedish characters, prefix commands with `PYTHONIOENCODING=utf-8`.
- New Python dependencies go in the `pip:` section of `environment.yml`.

## Architecture

**Ingestion** (`src/rag/ingest.py`): `document_loader` (PDF per page / TXT / MD, normalisation, fragmented-PDF repair) → `chunker` (overlapping character windows) → `embeddings` (`OllamaEmbedder`, task `search_document`) → `vector_store` (NumPy; writes `embeddings.npy`, `chunks.json`, `metadata.json`).

**Questions:** `app/ui.py` → `generate_response()` → `AIService.process_message()` in `src/services/ai_service.py`:
1. validate the input;
2. `RAGService.retrieve()`: embed the question with `search_query`, cosine search, `RAG_TOP_K`, `RAG_MIN_SCORE`;
3. `prompt_builder.build_user_prompt()` with `SYSTEM_PROMPT`;
4. `OllamaModelClient.generate()`;
5. build the Sources list from the retrieved chunks.

Rules:
- **UI never touches Ollama, the model client, embeddings, or the vector store directly.** It only calls `generate_response()`. Dependency direction: UI → services → RAG/retrieval components → model/infrastructure.
- **Retrieval outcomes.** `RetrievalStatus` values: `ok`, `empty_index`, `no_relevant_context`, `index_mismatch`, `index_error`. Every non-`ok` status returns a fixed user message and **does not call the model**. `RAGService` checks the index's recorded embedding model and `document_embedding_task` via `VectorStore.compatibility_issue()`.
- **Nomic task prefixes** (`search_document:` / `search_query:`) are applied only in `src/rag/embeddings.py`. Don't add prefix strings elsewhere.
- **Error translation is layered.** `translate_ollama_error()` in `model_client.py` maps Ollama errors: typed checks for `ConnectionError` and a 404 `ResponseError` first, then message matching as a fallback, producing `OllamaConnectionError` / `ModelNotFoundError` / `ModelClientError`. It is shared by generation and embeddings. `AIService` turns these into an `AIResponse` (`src/schemas/responses.py`) with a user-friendly `content`; diagnostics go in `error_message`. Services should not raise to the UI.
- **Sources** are built by code from the retrieved chunks, never from model output.
- **Config** comes from `src/config.py`: a module-level `config` dataclass populated from `.env` via python-dotenv at import time. Add new settings there rather than reading `os.getenv` elsewhere. `RAG_MIN_SCORE=0.5` is provisional and untuned.

## Testing

Tests must not require Gradio, a running Ollama, real embeddings, or the course documents. The pattern is dependency injection with fakes and mocks: `AIService(model_client=..., rag_service=...)`, `RAGService(embedder=..., store=...)`, and `generate_response(msg, service=...)`. `ollama.Client` is patched in the model-client and embedding tests.

## Evaluation (frozen — do not modify)

- `evaluation/PROTOCOL.md`, `evaluation/test_cases.json` and `evaluation/fixtures/` are frozen. Do not edit the cases, gold sources, key facts, thresholds or rubric.
- `evaluation/run_eval.py` is the mechanical harness. It never assigns labels; grading is done by a human.
- The final results are in `evaluation/evaluation_results.md`.
- Raw runs and human grades live under the git-ignored `evaluation/runs/`.
- Don't re-run the formal evaluation, and don't change settings or prompts to change historical results.

## Other project artifacts

- `src/capabilities/` — These are optional-capability placeholder modules from the starter template. Implement or modify only the capability needed for this project. Leave unrelated starter-template placeholders unchanged unless their removal is explicitly approved.
- `docs/project-decisions.md` — team decision log; `docs/architecture.md` should be updated when the architecture changes.
- `data/` — repository-safety rules:
  - Do not commit course documents unless redistribution is permitted.
  - Never commit PII, secrets, credentials, or other sensitive data.
  - Avoid committing generated vector-store data or unnecessarily large local files.
