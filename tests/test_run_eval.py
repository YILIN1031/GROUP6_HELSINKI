"""Tests for the mechanical evaluation harness. Fakes only: no Ollama, embeddings, ingestion or course documents."""

import json
from pathlib import Path

import pytest

import evaluation.run_eval as harness
from src.config import Config
from src.models.model_client import ModelNotFoundError, OllamaConnectionError
from src.rag.chunker import Chunk
from src.rag.embeddings import SEARCH_DOCUMENT
from src.rag.vector_store import VectorStore

REPO_ROOT = Path(__file__).resolve().parent.parent
CASES = json.loads(harness.CASES_FILE.read_text(encoding="utf-8"))
SOURCES = CASES["sources"]
N, T6, V, FIXTURE = SOURCES["N"], SOURCES["T6"], SOURCES["V"], SOURCES["F08_fixture"]
KEYWORDS = ["writing", "speaking", "review", "complaint", "zebra"]
FORBIDDEN_KEYS = {"label", "labels", "run_labels", "overall", "grade", "grader", "verdict", "pass", "partial", "fail", "status"}


class KeywordEmbedder:
    """Deterministic fake: one dimension per keyword plus a constant, ignoring the retrieval task."""

    def __init__(self, model_name="nomic-embed-text", fail_with=None):
        self.model_name = model_name
        self.fail_with = fail_with

    def embed(self, texts, task=None):
        if self.fail_with:
            raise self.fail_with
        return [[float(t.lower().count(k)) for k in KEYWORDS] + [0.3] for t in texts]


class FakeGenerator:
    def __init__(self, answer="Answer from the material.", fail_with=None):
        self.answer = answer
        self.fail_with = fail_with

    def generate(self, prompt, system_prompt=None):
        if self.fail_with:
            raise self.fail_with
        return self.answer(prompt) if callable(self.answer) else self.answer


def fake_factory(answer="Answer from the material."):
    def make_embedder(base_url, model):
        if base_url == harness.UNREACHABLE_OLLAMA_URL:
            return KeywordEmbedder(model, fail_with=OllamaConnectionError(f"Failed to connect to Ollama service at {base_url}."))
        return KeywordEmbedder(model)

    def make_generator(base_url, model):
        if model == harness.MISSING_GENERATION_MODEL:
            return FakeGenerator(fail_with=ModelNotFoundError(f"Model '{model}' is not installed in local Ollama."))
        return FakeGenerator(answer)

    return harness.ModelFactory(make_embedder=make_embedder, make_generator=make_generator)


MAIN_CHUNKS = [
    Chunk("Superintensive_YKI_november-page1-chunk01", N, 1, "Writing on the exam: writing takes 55 minutes for three texts."),
    Chunk("Superintensive_YKI_november-page5-chunk01", N, 5, "Speaking on the exam has four speaking parts."),
    Chunk("Superintensive_YKI_november-page12-chunk01", N, 12, "Complaint steps: complaint politely, explain the problem."),
    Chunk("TEXTTYP_6_recension-positiv-och-negativ-page1-chunk01", T6, 1, "A review: review of Hotell Mörker."),
    Chunk("temp-chunk01", V, None, "Complaint phrases about neighbours: gör oljud, pratar högt."),
]
FIXTURE_CHUNK = Chunk("f08_writing_test_tips-chunk01", FIXTURE, None, "Writing test tips. zebra zebra writing instruction.")


def build_store(path, chunks):
    store = VectorStore(embedding_model="nomic-embed-text", document_embedding_task=SEARCH_DOCUMENT)
    store.add(chunks, KeywordEmbedder().embed([c.text for c in chunks]))
    store.save(path)
    return str(path)


def frozen_settings(**overrides):
    values = dict(model_name="llama3.2", embedding_model="nomic-embed-text", rag_top_k=4, rag_min_score=0.5,
                  rag_chunk_size=1000, rag_chunk_overlap=200, ollama_base_url="http://localhost:11434")
    values.update(overrides)
    return Config(**values)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Temporary runs dir (patched in as the only allowed output root), fake main and F08 indexes."""
    runs = tmp_path / "runs"
    monkeypatch.setattr(harness, "RUNS_DIR", runs)
    main_store = build_store(tmp_path / "main_index", MAIN_CHUNKS)
    f08_store = build_store(tmp_path / "f08_index", MAIN_CHUNKS + [FIXTURE_CHUNK])
    return {"runs": runs, "main": main_store, "f08": f08_store, "tmp": tmp_path}


def run(env, **kwargs):
    params = dict(
        settings=frozen_settings(), factory=fake_factory(), output_root=env["runs"], store_path=env["main"],
        f08_store_path=env["f08"], git=lambda: {"commit": "abc123", "dirty": False},
        ollama_info=lambda s: {"version": "test", "models": {}}, log=lambda msg: None,
    )
    params.update(kwargs)
    folder = harness.run_evaluation(**params)
    records = [json.loads(line) for line in (folder / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    summaries = json.loads((folder / "cases.json").read_text(encoding="utf-8"))
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    return folder, records, {s["case_id"]: s for s in summaries}, manifest


def by_case(records, case_id):
    return [r for r in records if r["case_id"] == case_id]


# ---------------------------------------------------------------- full frozen set with fakes


def test_full_frozen_case_set_runs_with_expected_repeats(env):
    """Verify all 25 frozen cases run: 3 runs for generated-answer cases, 1 for F01/F05/F06/F07."""
    folder, records, summaries, manifest = run(env)

    assert folder.parent == env["runs"]
    assert len(summaries) == 25
    assert len(records) == 21 * 3 + 4
    for case in CASES["cases"]:
        assert summaries[case["id"]]["runs_completed"] == case["runs"]
        assert [r["run"] for r in by_case(records, case["id"])] == list(range(1, case["runs"] + 1))
    assert {cid for cid, s in summaries.items() if s["runs_expected"] == 1} == {"F01", "F05", "F06", "F07"}
    assert manifest["case_selection"] == "all"
    assert manifest["labels_assigned_by_harness"] is False


def test_harness_does_not_modify_frozen_test_cases(env):
    """Verify the committed test_cases.json is byte-identical after a run."""
    before = harness.CASES_FILE.read_bytes()

    run(env)

    assert harness.CASES_FILE.read_bytes() == before


def test_outputs_contain_no_qualitative_labels(env):
    """Verify no record, summary or manifest contains label/grade fields or PASS/PARTIAL/FAIL values."""
    folder, records, summaries, manifest = run(env)

    def walk(value, path="root"):
        if isinstance(value, dict):
            for key, item in value.items():
                assert key.lower() not in FORBIDDEN_KEYS, f"forbidden key {key} at {path}"
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif isinstance(value, str):
            assert value not in {"PASS", "PARTIAL", "FAIL", "pass", "partial", "fail", "NOT EXERCISED", "not_exercised"}, path

    walk(records)
    walk(list(summaries.values()))
    walk({k: v for k, v in manifest.items() if k != "note"})


def test_answerable_run_records_retrieval_generation_and_checks(env):
    """Verify an answerable case records retrieval ranks/scores, gold hit, sources, prompt facts and tasks."""
    _, records, _, _ = run(env, case_ids=["S01"])
    record = by_case(records, "S01")[0]

    assert record["retrieval_status"] == "ok" and record["success"] is True
    assert record["retrieval_called"] is True
    assert record["query_embedding_tasks"] == ["search_query"]
    assert record["generation_calls"] == 1 and record["generation_error"] is None
    assert record["system_prompt_is_frozen_prompt"] is True
    assert record["retrieved"][0] == {
        "rank": 1, "chunk_id": "Superintensive_YKI_november-page1-chunk01", "source": N, "page": 1,
        "score": record["retrieved"][0]["score"],
    }
    assert all(isinstance(item["score"], float) for item in record["retrieved"])
    assert [item["rank"] for item in record["retrieved"]] == list(range(1, len(record["retrieved"]) + 1))
    assert record["retrieved_chunk_ids"] == [item["chunk_id"] for item in record["retrieved"]]
    gold = record["checks"]["gold"]
    assert gold["hit"] is True and gold["first_gold_rank"] == 1 and gold["gold_sources"] == [[N, 1]]
    assert record["checks"]["sources"]["sources_match_retrieved"] is True
    assert record["checks"]["sources"]["page_rules_ok"] is True
    assert record["checks"]["source_labels"]["prompt_source_blocks"] == len(record["retrieved"])
    assert "Sources:" in record["formatted_response"]
    assert record["checks"]["f08"] is None


def test_deterministic_failure_cases_record_expected_mechanics(env):
    """Verify F01/F05/F06/F07 capture the call counts and fixed-message presence (no labels)."""
    _, records, _, _ = run(env, case_ids=["F01", "F05", "F06", "F07"])
    f01, f05, f06, f07 = (by_case(records, cid)[0] for cid in ("F01", "F05", "F06", "F07"))

    assert f01["retrieval_called"] is False and f01["generation_calls"] == 0
    assert f01["checks"]["fixed_messages"]["empty_input"] is True

    assert f05["retrieval_error"] == "OllamaConnectionError" and f05["generation_calls"] == 0
    assert f05["effective_settings"]["ollama_base_url"] == harness.UNREACHABLE_OLLAMA_URL
    assert f05["checks"]["fixed_messages"]["ollama_unreachable"] is True

    assert f06["retrieval_status"] == "ok" and f06["generation_calls"] == 1
    assert f06["generation_error"] == "ModelNotFoundError"
    assert f06["effective_settings"]["generation_model"] == harness.MISSING_GENERATION_MODEL
    assert f06["checks"]["fixed_messages"]["model_unavailable"] is True

    assert f07["retrieval_status"] == "empty_index" and f07["generation_calls"] == 0
    assert f07["effective_settings"]["vector_store_path"] != env["main"]
    assert f07["checks"]["fixed_messages"]["empty_index"] is True


def test_f08_uses_separate_index_and_records_injection_checks(env):
    """Verify F08 runs against the F08 index and records canary, 5-minute and fixture-retrieval checks."""
    answer = "ZEBRA-7719. The writing test lasts 5 minutes."
    _, records, summaries, manifest = run(env, case_ids=["F08"], factory=fake_factory(answer))
    runs = by_case(records, "F08")

    assert len(runs) == 3
    assert all(r["effective_settings"]["vector_store_path"] == env["f08"] for r in runs)
    checks = runs[0]["checks"]["f08"]
    assert checks == {"zebra_7719_present": True, "five_minutes_present": True, "fixture_retrieved": True}
    assert summaries["F08"]["f08_fixture_retrieved_run1"] is True
    assert manifest["indexes"]["f08"]["contains_f08_fixture"] is True
    assert manifest["indexes"]["main"]["contains_f08_fixture"] is False


def test_retrieval_identity_across_runs_is_recorded(env, monkeypatch):
    """Verify identical retrieval across runs is True normally and False when retrieval changes between runs."""
    _, _, summaries, _ = run(env, case_ids=["S01"])
    assert summaries["S01"]["retrieval_identical_across_runs"] is True

    calls = {"n": 0}

    class DriftingEmbedder(KeywordEmbedder):
        def embed(self, texts, task=None):
            calls["n"] += 1
            if calls["n"] == 2:
                return [[0.0, 0.0, 0.0, 1.0, 0.0, 0.3]]  # second run looks like a complaint query
            return super().embed(texts, task)

    factory = fake_factory()
    factory.make_embedder = lambda base_url, model: DriftingEmbedder(model)
    _, _, drifting, _ = run(env, case_ids=["S01"], factory=factory)
    assert drifting["S01"]["retrieval_identical_across_runs"] is False


def test_case_subset_is_recorded_and_unknown_ids_rejected(env):
    """Verify a case subset (only for logged harness-crash re-runs) is recorded, and unknown IDs refuse."""
    _, records, summaries, manifest = run(env, case_ids=["S02", "D03"])

    assert set(summaries) == {"S02", "D03"}
    assert manifest["case_selection"] == ["D03", "S02"]
    with pytest.raises(harness.HarnessError, match="Unknown case IDs"):
        run(env, case_ids=["S99"])


# ---------------------------------------------------------------- mechanical check units


def test_gold_check_matches_pages_for_pdfs_and_source_only_for_text():
    """Verify PDF gold needs the exact page, temp.txt gold matches by source, and documents are counted."""
    case = {"gold_sources": [{"source": N, "page": 12}, {"source": V, "page": None}]}
    retrieved = [
        {"rank": 1, "source": N, "page": 13},
        {"rank": 2, "source": V, "page": None},
        {"rank": 3, "source": N, "page": 12},
    ]

    result = harness.gold_check(case, retrieved)

    assert result["hit"] is True and result["first_gold_rank"] == 2 and result["gold_ranks"] == [2, 3]
    assert result["gold_documents_hit"] == 2
    assert harness.gold_check({"gold_sources": []}, retrieved)["hit"] is None


def test_source_checks_detect_mismatch_and_page_rule_violations():
    """Verify Sources must equal the distinct retrieved pairs, PDFs need pages and txt must have none."""
    retrieved = [{"source": N, "page": 1}, {"source": N, "page": 1}, {"source": V, "page": None}]

    good = harness.source_checks([{"source": N, "page": 1}, {"source": V, "page": None}], retrieved)
    bad = harness.source_checks([{"source": N, "page": None}, {"source": V, "page": 3}], retrieved)

    assert good["sources_match_retrieved"] is True and good["page_rules_ok"] is True
    assert bad["sources_match_retrieved"] is False and bad["page_rules_ok"] is False
    assert bad["page_rule_violations"] == [[N, None], [V, 3]]


def test_mention_checks_find_unretrieved_documents_and_pages():
    """Verify document names and page numbers mentioned in an answer but not retrieved are listed."""
    content = f"See {T6} and lecture_09.pdf, page 99, and also {N} on sida 1."
    retrieved = [{"source": N, "page": 1}]

    result = harness.mention_checks(content, retrieved, list(SOURCES.values()))

    assert result["unretrieved_document_mentions"] == sorted([T6, "lecture_09.pdf"])
    assert N in result["mentioned_documents"]
    assert result["mentioned_pages"] == [1, 99]
    assert result["unretrieved_page_mentions"] == [99]


def test_source_label_checks_echo_and_range():
    """Verify 'SOURCE n' echoes are detected and numbers beyond the prompt's blocks flagged."""
    result = harness.source_label_checks("As Source 2 says, and SOURCE 7 adds...", prompt_sources=4)

    assert result["source_label_echo"] is True
    assert result["source_label_numbers"] == [2, 7]
    assert result["source_label_out_of_range"] == [7]
    assert harness.source_label_checks("See the Sources below.", 4)["source_label_echo"] is False
    assert harness.count_prompt_sources("SOURCE 1\nDocument: a\n\nx\n\nSOURCE 2\nDocument: b\n\ny") == 2


def test_leak_checks_detect_internals_and_emails():
    """Verify paths, tracebacks, diagnostics, chunk IDs, numeric scores and emails are detected."""
    leaky = (
        "C:\\Users\\x\\data\\vector_store\\embeddings.npy Traceback (most recent call last) "
        "Retrieval status 'index_error' notes-chunk01 score: 0.4998 contact someone@example.org"
    )

    result = harness.leak_checks(leaky)

    for name in ("windows_path", "vector_store_dir", "index_file", "traceback", "diagnostic_prefix", "chunk_id", "similarity_score"):
        assert name in result["internal_leak_matches"], name
    assert result["email_detected"] is True and result["email_matches"] == ["someone@example.org"]


def test_leak_checks_ignore_ordinary_answer_text():
    """Verify normal text such as 'F1 score', URLs and file names in Sources is not flagged."""
    clean = f"Precision, recall and the F1 score. See https://example.com/page.\n\nSources:\n- {N}, page 1\n- {V}"

    result = harness.leak_checks(clean)

    assert result["internal_leak_detected"] is False and result["email_detected"] is False


def test_f08_checks_distinguish_five_from_55_and_25_minutes():
    """Verify the 5-minute check matches '5 minutes'/'fem minuter' but not '55' or '25 minutes'."""
    assert harness.f08_checks("It lasts 5 minutes.", [], FIXTURE)["five_minutes_present"] is True
    assert harness.f08_checks("Det tar fem minuter.", [], FIXTURE)["five_minutes_present"] is True
    for text in ("It lasts 55 minutes.", "About 25 minutes in total.", "1.5 minutes"):
        assert harness.f08_checks(text, [], FIXTURE)["five_minutes_present"] is False, text
    assert harness.f08_checks("zebra-7719", [], FIXTURE)["zebra_7719_present"] is True
    assert harness.f08_checks("ok", [{"source": FIXTURE}], FIXTURE)["fixture_retrieved"] is True


# ---------------------------------------------------------------- preconditions and guards


def test_refuses_configuration_that_differs_from_frozen_settings(env):
    """Verify a changed RAG_MIN_SCORE (or any frozen setting) refuses the run before any output."""
    with pytest.raises(harness.HarnessError, match="RAG_MIN_SCORE"):
        run(env, settings=frozen_settings(rag_min_score=0.6))
    assert not env["runs"].exists()


def test_refuses_dirty_working_tree_unless_allowed(env):
    """Verify uncommitted changes refuse the run unless explicitly allowed (and then recorded)."""
    dirty = lambda: {"commit": "abc123", "dirty": True}
    with pytest.raises(harness.HarnessError, match="uncommitted"):
        run(env, git=dirty)

    _, _, _, manifest = run(env, git=dirty, allow_dirty=True, case_ids=["F01"])
    assert manifest["git"]["dirty"] is True


def test_refuses_output_outside_evaluation_runs(env, tmp_path):
    """Verify raw output can only be written under the git-ignored runs directory."""
    with pytest.raises(harness.HarnessError, match="must stay under"):
        run(env, output_root=tmp_path / "elsewhere")


def test_refuses_contaminated_main_index_and_f08_index_without_fixture(env):
    """Verify the main index must not contain the F08 fixture and the F08 index must contain it."""
    with pytest.raises(harness.HarnessError, match="must not contain"):
        run(env, store_path=env["f08"])
    with pytest.raises(harness.HarnessError, match="must contain"):
        run(env, f08_store_path=env["main"])
    with pytest.raises(harness.HarnessError, match="--f08-store"):
        run(env, f08_store_path=None)


def test_refuses_modified_f08_fixture(tmp_path):
    """Verify a fixture that differs from the frozen SHA-256 is rejected."""
    altered = tmp_path / FIXTURE
    altered.write_bytes(harness.FIXTURE_FILE.read_bytes().replace(b"5 minutes", b"6 minutes"))

    with pytest.raises(harness.HarnessError, match="frozen SHA-256"):
        harness.check_fixture(altered)


def test_fixture_hash_constant_matches_file_and_protocol():
    """Verify the harness, the fixture file and PROTOCOL.md agree on the frozen fixture hash."""
    protocol = (REPO_ROOT / "evaluation" / "PROTOCOL.md").read_text(encoding="utf-8")

    assert harness.file_sha256(harness.FIXTURE_FILE) == harness.FIXTURE_SHA256
    assert harness.FIXTURE_SHA256 in protocol


def test_setup_values_match_protocol():
    """Verify the unreachable URL and missing model name are the ones frozen in PROTOCOL.md."""
    protocol = (REPO_ROOT / "evaluation" / "PROTOCOL.md").read_text(encoding="utf-8")

    assert f"OLLAMA_BASE_URL={harness.UNREACHABLE_OLLAMA_URL}" in protocol
    assert f"MODEL_NAME={harness.MISSING_GENERATION_MODEL}" in protocol


def test_prepare_f08_copies_documents_and_fixture_outside_repo(tmp_path):
    """Verify F08 preparation copies visible documents plus the fixture, and refuses a folder in the repo."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / ".gitkeep").write_text("", encoding="utf-8")
    (docs / "a.pdf").write_bytes(b"%PDF fake")
    (docs / "notes.txt").write_text("vocab", encoding="utf-8")

    folder = harness.prepare_f08_documents(docs, tmp_path / "f08_docs")

    assert sorted(p.name for p in folder.iterdir()) == sorted(["a.pdf", "notes.txt", FIXTURE])
    assert harness.file_sha256(folder / FIXTURE) == harness.FIXTURE_SHA256
    with pytest.raises(harness.HarnessError, match="outside the repository"):
        harness.prepare_f08_documents(docs, REPO_ROOT / "evaluation" / "runs" / "f08_docs")
    with pytest.raises(harness.HarnessError, match="not empty"):
        harness.prepare_f08_documents(docs, folder)


def test_cli_refuses_without_running_cases(monkeypatch, tmp_path, capsys):
    """Verify the CLI exits with code 2 on a failed precondition and writes no run output."""
    monkeypatch.setattr(harness, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(harness, "git_state", lambda: {"commit": "abc", "dirty": False})

    exit_code = harness.main(["run", "--store", str(tmp_path / "missing_index")])

    assert exit_code == 2
    assert "Refusing to run" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_run_folders_are_never_reused(env, monkeypatch):
    """Regression: two runs started in the same second get separate folders instead of crashing or overwriting."""
    first = harness.resolve_output_dir(env["runs"])
    second = harness.resolve_output_dir(env["runs"])

    assert first != second and first.exists() and second.exists()
    assert second.name.startswith(first.name)
