from __future__ import annotations

from pathlib import Path

import pytest

from genlesson import tracing
from genlesson.classify import run_classify
from genlesson.config import (
    GenerationOptions,
    PromptOptions,
    ReviewOptions,
    VerificationOptions,
)
from genlesson.errors import GenLessonError
from genlesson.generate import run_generate
from genlesson.ingest import run_ingest
from genlesson.models import (
    PosterSpecDocument,
    ReviewDocument,
    RunManifest,
    VerificationDocument,
)
from genlesson.plan import run_plan
from genlesson.prompts import run_prompts
from genlesson.review import run_review
from genlesson.storage import read_model
from genlesson.verify import normalize_compare, run_verify

from .helpers import (
    FakeImageProvider,
    FakePlanner,
    FakePosterPlanner,
    FakeVisionTranscriber,
    default_options,
    make_lesson_docx,
    plan_options,
    planner_options,
    start_run,
)
from .test_tracing import _enable


@pytest.fixture(autouse=True)
def _clear_tracing():
    yield
    tracing.shutdown()


def _generated_run(
    tmp_path: Path,
    formats=("portrait", "landscape"),
    fail_formats=None,
    run_id: str = "test-run",
):
    path = make_lesson_docx(
        tmp_path / "lesson.docx",
        [
            "Байланыс дегеніміз — адамдар арасындағы ақпарат алмасу.",
            "Байланыстың ауызша, жазбаша және визуалды түрлері бар.",
        ],
    )
    ingest_options = default_options()
    store, paths, manifest = start_run(tmp_path, path, ingest_options, run_id=run_id)
    run_ingest(paths, manifest, store, ingest_options)
    run_classify(paths, manifest, store, planner_options(), planner=FakePlanner())
    run_plan(
        paths,
        manifest,
        store,
        plan_options(),
        planner=FakePosterPlanner(detail_count=2),
    )
    run_review(paths, manifest, store, ReviewOptions(mode="off"))
    run_prompts(
        paths,
        manifest,
        store,
        PromptOptions(formats=formats, image_model="gpt-image-1"),  # type: ignore[arg-type]
    )
    run_generate(
        paths,
        manifest,
        store,
        GenerationOptions(api_key="test-key", timeout_s=30, max_retries=1),
        provider=FakeImageProvider(fail_formats=fail_formats or set()),
    )
    review = read_model(paths.stage_dir("review") / "review.json", ReviewDocument)
    spec = read_model(paths.resolve(review.approved_spec_path), PosterSpecDocument)
    return store, paths, manifest, spec


def _exact_transcript(spec: PosterSpecDocument) -> str:
    return "\n".join(spec.strings_to_render)


def _verify(paths, manifest, store, transcriber, image_provider=None, options=None):
    return run_verify(
        paths,
        manifest,
        store,
        options or VerificationOptions(api_key="test-key", similarity_threshold=0.9, max_retries=2),
        transcriber=transcriber,
        image_provider=image_provider,
    )


def test_normalize_compare_nfc_casefold_and_punctuation() -> None:
    assert normalize_compare("  «БАЙЛАНЫС ТҮРЛЕРІ!»  ") == "байланыс түрлері"
    assert normalize_compare("Cafe\u0301") == normalize_compare("Café")
    assert normalize_compare("...") == ""


def test_verification_passes_when_text_matches(tmp_path: Path) -> None:
    store, paths, manifest, spec = _generated_run(tmp_path)
    vision = FakeVisionTranscriber(_exact_transcript(spec))

    result = _verify(paths, manifest, store, vision)

    assert result.needs_attention == 0
    assert all(item.status == "passed" for item in result.document.formats)
    assert result.document.stats.passed == 2
    assert result.document.stats.retries == 0
    for item in result.document.formats:
        assert item.best_attempt == 1
        assert item.best_image_path is not None
        assert paths.resolve(item.best_image_path).is_file()
    saved = read_model(
        paths.stage_dir("verify") / "verification.json", VerificationDocument
    )
    assert saved.stats.passed == 2
    assert read_model(paths.manifest_path, RunManifest).stages["verify"].status == "complete"


def test_mismatch_retries_with_correction_note(tmp_path: Path) -> None:
    store, paths, manifest, spec = _generated_run(tmp_path, formats=("portrait",))
    exact = _exact_transcript(spec)
    bad = exact.replace("Түр 2", "", 1) + "\nWRONG EXTRA"
    vision = FakeVisionTranscriber([bad, exact])
    retry_provider = FakeImageProvider()

    result = _verify(paths, manifest, store, vision, retry_provider)

    item = result.document.formats[0]
    assert item.status == "passed"
    assert item.attempts[0].passed is False
    assert "Түр 2" in item.attempts[0].exact_mismatches
    assert any(extra.text == "WRONG EXTRA" for extra in item.attempts[0].extra_strings)
    assert item.attempts[1].passed is True
    assert item.best_attempt == 2
    assert len(vision.calls) == 2
    assert len(retry_provider.calls) == 1
    retry_prompt = retry_provider.calls[0][1]
    assert "CORRECTION NOTE" in retry_prompt
    assert "Түр 2" in retry_prompt
    assert "WRONG EXTRA" in retry_prompt
    assert result.document.stats.retries == 1


def test_retries_exhausted_saves_best_and_marks_needs_attention(tmp_path: Path) -> None:
    store, paths, manifest, spec = _generated_run(tmp_path, formats=("portrait",))
    exact = _exact_transcript(spec)
    bad = exact.replace("Түр 2", "", 1)
    vision = FakeVisionTranscriber(bad)
    retry_provider = FakeImageProvider()
    options = VerificationOptions(
        api_key="test-key", similarity_threshold=0.9, max_retries=1
    )

    result = _verify(paths, manifest, store, vision, retry_provider, options)

    item = result.document.formats[0]
    assert item.status == "needs_attention"
    assert len(item.attempts) == 2
    assert item.best_attempt is not None
    assert item.best_image_path is not None
    assert paths.resolve(item.best_image_path).is_file()
    assert result.needs_attention == 1
    assert any("needs attention" in warning for warning in result.document.warnings)


def test_extra_strings_are_flagged(tmp_path: Path) -> None:
    store, paths, manifest, spec = _generated_run(tmp_path, formats=("portrait",))
    vision = FakeVisionTranscriber(_exact_transcript(spec) + "\nUNKNOWN EXTRA TEXT")
    options = VerificationOptions(
        api_key="test-key", similarity_threshold=0.9, max_retries=0
    )

    result = _verify(paths, manifest, store, vision, options=options)

    item = result.document.formats[0]
    assert item.status == "needs_attention"
    assert item.attempts[0].extra_strings
    assert any("UNKNOWN EXTRA" in extra.text for extra in item.attempts[0].extra_strings)


def test_generation_failed_format_is_recorded(tmp_path: Path) -> None:
    store, paths, manifest, spec = _generated_run(
        tmp_path, formats=("portrait", "landscape"), fail_formats={"landscape"}
    )
    vision = FakeVisionTranscriber(_exact_transcript(spec))

    result = _verify(paths, manifest, store, vision)

    by_format = {item.format: item for item in result.document.formats}
    assert by_format["portrait"].status == "passed"
    assert by_format["landscape"].status == "generation_failed"
    assert result.document.stats.generation_failed == 1


def test_verifier_falls_back_when_planner_has_no_vision(tmp_path: Path) -> None:
    store, paths, manifest, spec = _generated_run(tmp_path, formats=("portrait",))
    manifest.config["planner"] = {"provider": "deepseek", "model": "deepseek-text"}
    vision = FakeVisionTranscriber(_exact_transcript(spec))

    result = _verify(paths, manifest, store, vision)

    assert result.document.config.fallback_used is True
    assert result.document.config.planner_model == "deepseek-text"
    assert result.document.config.verifier_model != "deepseek-text"


def test_verify_requires_api_key_without_injected_transcriber(tmp_path: Path) -> None:
    store, paths, manifest, _ = _generated_run(tmp_path, formats=("portrait",))

    with pytest.raises(GenLessonError) as excinfo:
        run_verify(
            paths,
            manifest,
            store,
            VerificationOptions(api_key=None, max_retries=0),
        )

    assert excinfo.value.code == "vision_provider_required"


def test_verify_resume_reuses_report(tmp_path: Path) -> None:
    store, paths, manifest, spec = _generated_run(tmp_path, formats=("portrait",))
    first_vision = FakeVisionTranscriber(_exact_transcript(spec))
    first = _verify(paths, manifest, store, first_vision)

    saved = read_model(paths.manifest_path, RunManifest)
    second_vision = FakeVisionTranscriber(_exact_transcript(spec))
    second = _verify(paths, saved, store, second_vision)

    assert second.resumed is True
    assert second.document == first.document
    assert second_vision.calls == []


def test_sentence_threshold_is_configurable(tmp_path: Path) -> None:
    from genlesson.verify import _compare

    store, paths, manifest, spec = _generated_run(tmp_path, formats=("portrait",))
    transcript = _exact_transcript(spec).replace(
        spec.central_idea.definition, "Байланыс — мүлде басқа бір сөйлем."
    )

    checks, extra, exact, below, score, passed = _compare(spec, transcript, 0.99)

    definition_check = next(
        check for check in checks if check.expected == spec.central_idea.definition
    )
    assert definition_check.status == "missing"
    assert spec.central_idea.definition in below
    assert not exact
    assert not passed


def test_wrapped_lines_and_list_markers_are_not_extra(tmp_path: Path) -> None:
    store, paths, manifest, spec = _generated_run(tmp_path)
    wrapped: list[str] = []
    for index, value in enumerate(spec.strings_to_render, start=1):
        words = value.split()
        chunks = [" ".join(words[i : i + 4]) for i in range(0, len(words), 4)] or [value]
        wrapped.append(f"{index}. {chunks[0]}")
        wrapped.extend(chunks[1:])
    vision = FakeVisionTranscriber("\n".join(wrapped))

    result = _verify(paths, manifest, store, vision)

    assert result.document.stats.passed == 2
    assert result.needs_attention == 0
    assert all(
        not attempt.extra_strings
        for item in result.document.formats
        for attempt in item.attempts
    )


def test_kazakh_letters_remain_distinct_after_normalization() -> None:
    pairs = [
        ("ә", "а"),
        ("ғ", "г"),
        ("қ", "к"),
        ("ң", "н"),
        ("ө", "о"),
        ("ұ", "у"),
        ("ү", "у"),
        ("һ", "х"),
        ("і", "и"),
    ]
    for kazakh, lookalike in pairs:
        assert normalize_compare(f"Сөз{kazakh}") != normalize_compare(f"Сөз{lookalike}")
    assert normalize_compare("ӘҒҚҢӨҰҮҺІ") == "әғқңөұүһі"
    assert normalize_compare("  ҚАЗАҚ!  ") == "қазақ"


def test_verify_opens_format_attempt_and_retry_spans(monkeypatch, tmp_path: Path) -> None:
    store, paths, manifest, spec = _generated_run(tmp_path, formats=("portrait",))
    backend = _enable(monkeypatch)
    exact = _exact_transcript(spec)
    bad = exact.replace("Түр 2", "", 1) + "\nWRONG EXTRA"
    vision = FakeVisionTranscriber([bad, exact])

    result = _verify(paths, manifest, store, vision, FakeImageProvider())

    assert result.document.formats[0].status == "passed"
    entries = list(zip(backend.created, backend.observations))
    by_name = {created["name"]: (created, observation) for created, observation in entries}
    format_entry, format_observation = by_name["verify:portrait"]
    attempt1_entry, _attempt1_observation = by_name["verify.attempt:1"]
    attempt2_entry, attempt2_observation = by_name["verify.attempt:2"]
    retry_entry, _ = by_name["verify.retry_image"]

    assert format_entry["_parent"] is None
    assert attempt1_entry["_parent"] == format_observation.id
    assert attempt2_entry["_parent"] == format_observation.id
    assert retry_entry["_parent"] == attempt2_observation.id
    assert format_entry["input"]["format"] == "portrait"

    scores = {
        score["name"]: score
        for score in backend.scores
        if score.get("observation_id") == format_observation.id
    }
    assert scores["verify_status"]["value"] == "passed"
    assert scores["verify_status"]["data_type"] == "CATEGORICAL"
    assert scores["text_match"]["value"] == 1.0
    assert scores["exact_mismatch_count"]["value"] == 0.0
    assert scores["extra_strings_count"]["value"] == 0.0
    assert scores["retries_used"]["value"] == 1.0
