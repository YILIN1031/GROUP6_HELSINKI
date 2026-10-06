# Project decisions

This decision log records significant technical and architectural choices made during development. Each entry describes what was decided and why, as supported by the implementation and repository history.

## Decision 1 — Model selection

**Decision:** Use local models served by Ollama: `llama3.2` for answer generation (`MODEL_NAME`) and `nomic-embed-text` for embeddings (`EMBEDDING_MODEL`).

**Alternatives considered:** Hosted cloud LLM APIs, ruled out by the project's local-first requirement and the course constraints (no cloud LLM APIs). Other Ollama-compatible models remain configurable through the environment, but were not evaluated.

**Why we chose this:**
- The course documents stay on the local machine.
- No paid API is needed.
- It is compatible with the course starter project, which already used `llama3.2` through Ollama.
- `nomic-embed-text` is a local embedding model available through the same Ollama runtime.

The frozen evaluation later showed that grounding with `llama3.2` is often unreliable (see [`evaluation/evaluation_results.md`](../evaluation/evaluation_results.md)); the model choice itself was not changed after results were observed.

---

## Decision 2 — Additional AI capability

**Decision:** Retrieval-Augmented Generation (RAG) is the project's additional AI capability.

**Why it is needed:**
- CourseMate must answer questions about a student's own course documents instead of the model's general knowledge.
- Retrieval supplies the relevant passages, lets the application show which documents an answer is based on, and allows the material to be updated by re-running ingestion.

**Alternatives considered:** The starter template's other optional capabilities (tools / external APIs, MCP, agent workflows, persistent memory, multimodal). None addresses grounding answers in local documents, and agents, MCP, cloud APIs and authentication were explicitly out of scope. Their placeholder modules in `src/capabilities/` are left unimplemented.

---

## Decision 3 — Architecture

**Decision:** Keep the starter template's layered architecture and add a RAG layer beneath the service layer:
- **UI:** `app/ui.py` calls only `generate_response()`.
- **Application service:** `AIService`.
- **Retrieval service:** `RAGService`, which sits on the loader, chunker, embeddings and vector store in `src/rag/`.
- **Model client:** talks to Ollama.

Retrieval runs before generation, and the Sources list is built by code from the retrieved chunks.

**Why:**
- It preserves the course requirement that the UI never talks directly to Ollama, the model client, embeddings or the vector store.
- It keeps each component independently testable, with fakes injected in the tests.
- Building Sources in code prevents the model from inventing source names in the Sources list.

---

## Decision 4 — Local NumPy vector store instead of ChromaDB

**Decision:** Implement a small local vector store with NumPy (`src/rag/vector_store.py`). The index is persisted as `embeddings.npy`, `chunks.json` (chunk records only) and `metadata.json` (index-level metadata), and searched by cosine similarity.

**Alternatives considered:** ChromaDB, which the original project specification listed as an option.

**Why:**
- It adds no heavy dependencies; only NumPy is needed.
- It is transparent and easy to explain.
- It can be fully unit-tested with hand-made vectors.
- It is sufficient for a single-user, course-sized document collection.

Loading validates the index's integrity (row and dimension consistency, required metadata), so corrupt indexes fail clearly.

---

## Decision 5 — Explicit ingestion command, with restart after ingesting

**Decision:** Documents are indexed by an explicit command, `python -m src.rag.ingest`, rather than automatically at application start-up. The application loads the index once, so it must be restarted after re-ingesting. No hot reload is implemented.

**Why:**
- **Predictable startup:** the application starts quickly and behaves the same way every time.
- **Clearer errors:** ingestion failures (missing documents, Ollama unavailable, missing embedding model) are reported by the command rather than at app start-up.
- **No partial index:** the index is written only after ingestion fully succeeds.
- **Simpler:** avoiding hot reload keeps the application simple.

---

## Decision 6 — Retrieval task prefixes for `nomic-embed-text`, recorded in the index

**Decision:** Embed indexed passages with the `search_document` task and questions with the `search_query` task. For Nomic Embed models this adds the `search_document:` / `search_query:` prefixes that the model expects. The prefix logic lives only in `src/rag/embeddings.py`, and `metadata.json` records the document task (`document_embedding_task`).

**Why:**
- Nomic Embed models are designed to be used with these task prefixes.
- Recording the task lets `RAGService` detect an index built with a different embedding model or task, or one that predates the marker. It then returns `index_mismatch` instead of silently comparing incompatible vectors.
- The prefixes are applied only for Nomic Embed models.

---

## Decision 7 — Single controlled RAG prompt

**Decision:** Use one controlled prompt for all requests: questions, summaries, explanations and revision questions. It has two parts:
- the application rules in a system prompt;
- the retrieved material and the question in clearly delimited sections of the user message.

There is no separate task classifier.

**Why:** It is simpler, and it keeps application instructions separate from untrusted document content and the user's question.

---

## Decision 8 — Provisional relevance threshold

**Decision:** `RAG_MIN_SCORE=0.5` is a provisional starting value chosen before any evaluation. It was **not tuned** on the evaluation data, and the frozen evaluation used it unchanged.

**Why:** Tuning it on the same cases used to report final results would overfit the evaluation. The frozen evaluation found that 0.5 did not filter the out-of-scope test questions for this corpus. Any future tuning should use a separate development set.

---

## Decision 9 — Frozen evaluation protocol before results

**Decision:**
- **Freeze first:** the evaluation protocol, the 25 cases, the gold sources, the key facts, the thresholds, the rubric and the repeat policy were frozen and committed ([`evaluation/PROTOCOL.md`](../evaluation/PROTOCOL.md), [`evaluation/test_cases.json`](../evaluation/test_cases.json)) before the formal run.
- **Run once:** the formal run was executed once, with no re-runs.
- **Human grading:** every case was graded by a human after the run.
- **Mechanical harness:** the harness ([`evaluation/run_eval.py`](../evaluation/run_eval.py)) records mechanical facts only and never assigns quality labels.

**Why:**
- It prevents choosing questions, criteria or settings after seeing results.
- It separates objective mechanical checks from subjective quality judgements.
- It lets the reported results (6 PASS, 5 PARTIAL and 14 FAIL of 25) be trusted as a fair outcome of the fixed suite.

---

## Decision 10 — Keep course documents, indexes and raw evaluation output local

**Decision:** Course documents (`data/documents/`), generated indexes (`data/vector_store/`) and raw evaluation runs and human grades (`evaluation/runs/`) are git-ignored and never committed. Committed evaluation files contain only definitions, paraphrased facts and summarised results.

**Why:**
- The evaluation documents are copyright-protected and must not be redistributed.
- They contain personal data, such as owner watermarks and other students' names.
- Raw outputs can reproduce both.
