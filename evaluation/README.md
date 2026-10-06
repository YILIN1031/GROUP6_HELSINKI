# Evaluation Framework

Systematic evaluation checks whether CourseMate reliably answers questions from its indexed course documents, and fails safely when it cannot.

## Files

| File | Purpose |
|---|---|
| [`PROTOCOL.md`](PROTOCOL.md) | The frozen evaluation protocol: scope, frozen configuration, all 25 cases, the rubric, the repeat policy, the execution procedure and the privacy rules. |
| [`test_cases.json`](test_cases.json) | The machine-readable case definitions, with result fields that stay empty until each case is executed and graded. |
| [`evaluation_results.md`](evaluation_results.md) | The summary of graded results, written only after the evaluation has been run and graded. |
| [`fixtures/f08_writing_test_tips.md`](fixtures/f08_writing_test_tips.md) | A test-only document for case F08 (injection from a document). It is only indexed into a separate test index, never the main corpus. |
| `runs/` | Local raw run output. It is **git-ignored** and never committed, because it may contain copyrighted text and personal data from the source documents. |

## Case categories

The frozen set contains 25 cases (`category` values in `test_cases.json`):

1. **Successful cases** (`successful_case`, S01–S10): typical questions the documents answer.
2. **Difficult cases** (`difficult_case`, D01–D07): paraphrased, multi-document, ambiguous, long, half-answerable, Swedish-language and Chinese-language questions.
3. **Failure / adversarial cases** (`failure_case`, F01–F08): empty input, out-of-scope questions, injection from the user, injection from a document, missing Ollama, missing model and an empty index.

The conflicting-material category is not applicable to the evaluation corpus; see `PROTOCOL.md`.

## Labels

Each case is graded by a human on the dimensions retrieval (R), grounding (G), sources (S) and failure handling (H), then combined into an overall label:

- **PASS**: every applicable dimension passes.
- **PARTIAL**: no dimension fails, but at least one is only partial.
- **FAIL**: at least one dimension fails, including any critical failure.

`status` in `test_cases.json` uses the lowercase values `pending`, `pass`, `partial`, `fail` and `not_exercised` (F08 only). The exact criteria are in `PROTOCOL.md`, section 5.

## Rules

- The protocol and cases are frozen. Cases are not rewritten, removed or re-scoped after results are seen, and no setting is tuned during the evaluation run.
- Nothing is recorded in `test_cases.json` or `evaluation_results.md` until the corresponding case has actually been run and graded.
- The evaluation harness only performs mechanical checks; it never assigns qualitative labels.
