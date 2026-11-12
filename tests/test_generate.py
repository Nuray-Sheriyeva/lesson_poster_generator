from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from genlesson import tracing
from genlesson.classify import run_classify
from genlesson.config import GenerationOptions, PromptOptions, ReviewOptions
from genlesson.errors import GenLessonError
from genlesson.generate import run_generate
from genlesson.image.openai_image import OpenAIImageProvider
from genlesson.ingest import run_ingest
from genlesson.models import GeneratedFormat, GenerationDocument, RunManifest
from genlesson.plan import run_plan
from genlesson.prompts import run_prompts
from genlesson.review import run_review
from genlesson.storage import read_model

from .helpers import (
    FakeImageProvider,
    FakePlanner,
    FakePosterPlanner,
    _png_bytes,
    default_options,
    make_lesson_docx,
    plan_options,
    planner_options,
    start_run,
)
from .test_tracing import _enable


def _prompted_run(tmp_path: Path, formats: tuple[str, ...]):
    path = make_lesson_docx(
        tmp_path / "lesson.docx",
        [
            "Байланыс дегеніміз — адамдар арасындағы ақпарат алмасу.",
            "Байланыстың ауызша, жазбаша және визуалды түрлері бар.",
        ],
    )
    ingest_options = default_options()
    store, paths, manifest = start_run(tmp_path, path, ingest_options)
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
    return store, paths, manifest


@pytest.fixture(autouse=True)
def _clear_tracing():
    yield
    tracing.shutdown()


def _options() -> GenerationOptions:
    return GenerationOptions(api_key="test-key", timeout_s=30, max_retries=2)


def _generate(paths, manifest, store, provider, options=None):
    return run_generate(
        paths,
        manifest,
        store,
        options or _options(),
        provider=provider,
    )


def test_generate_all_formats_separately(tmp_path: Path) -> None:
    store, paths, manifest = _prompted_run(
        tmp_path, ("portrait", "landscape", "square")
    )
    provider = FakeImageProvider()

    result = _generate(paths, manifest, store, provider)

    assert result.generated == 3
    assert result.failed == 0
    assert len(provider.calls) == 3
    assert [item.format for item in result.document.formats] == [
        "portrait",
        "landscape",
        "square",
    ]
    assert [item.size for item in result.document.formats] == [
        "1024x1536",
        "1536x1024",
        "1024x1024",
    ]
    for item in result.document.formats:
        assert item.status == "generated"
        assert item.image_path is not None
        image_path = paths.resolve(item.image_path)
        assert image_path.is_file()
        assert image_path.read_bytes()
        assert item.image_sha256
        assert item.attempts == 1
        assert item.moderation_retry is False
        assert item.prompt_variant == "original"
    saved = read_model(paths.stage_dir("generate") / "generation.json", GenerationDocument)
    assert saved.stats.generated == 3
    assert read_model(paths.manifest_path, RunManifest).stages["generate"].status == "complete"
    assert read_model(paths.manifest_path, RunManifest).status == "in_progress"


def test_one_failed_format_does_not_fail_run(tmp_path: Path) -> None:
    store, paths, manifest = _prompted_run(tmp_path, ("portrait", "landscape"))
    provider = FakeImageProvider(fail_formats={"landscape"})

    result = _generate(paths, manifest, store, provider)

    assert result.generated == 1
    assert result.failed == 1
    by_format = {item.format: item for item in result.document.formats}
    assert by_format["portrait"].status == "generated"
    assert by_format["landscape"].status == "failed"
    assert by_format["landscape"].error is not None
    assert by_format["landscape"].error.code == "image_failed"
    assert any("landscape" in warning for warning in result.document.warnings)
    assert read_model(paths.manifest_path, RunManifest).stages["generate"].status == "complete"
    assert read_model(paths.manifest_path, RunManifest).status == "in_progress"


def test_moderation_refusal_retries_once_with_toned_down_prompt(tmp_path: Path) -> None:
    store, paths, manifest = _prompted_run(tmp_path, ("portrait",))
    provider = FakeImageProvider(moderate_formats={"portrait"})

    result = _generate(paths, manifest, store, provider)

    item = result.document.formats[0]
    assert item.status == "generated"
    assert item.attempts == 2
    assert item.moderation_retry is True
    assert item.prompt_variant == "toned_down"
    assert len(provider.calls) == 2
    assert provider.calls[0][1] != provider.calls[1][1]
    assert any("moderation" in warning for warning in result.document.warnings)


def test_moderation_twice_fails_only_that_format(tmp_path: Path) -> None:
    store, paths, manifest = _prompted_run(tmp_path, ("portrait", "square"))
    provider = FakeImageProvider(moderate_twice_formats={"portrait"})

    result = _generate(paths, manifest, store, provider)

    by_format = {item.format: item for item in result.document.formats}
    assert by_format["portrait"].status == "failed"
    assert by_format["portrait"].attempts == 2
    assert by_format["portrait"].moderation_retry is True
    assert by_format["portrait"].error is not None
    assert by_format["portrait"].error.moderation is True
    assert by_format["square"].status == "generated"
    assert result.generated == 1
    assert result.failed == 1
    assert read_model(paths.manifest_path, RunManifest).status == "in_progress"


def test_all_formats_failed_marks_stage_failed(tmp_path: Path) -> None:
    store, paths, manifest = _prompted_run(tmp_path, ("portrait", "landscape"))
    provider = FakeImageProvider(fail_formats={"portrait", "landscape"})

    with pytest.raises(GenLessonError) as excinfo:
        _generate(paths, manifest, store, provider)

    assert excinfo.value.code == "all_formats_failed"
    saved = read_model(paths.manifest_path, RunManifest)
    assert saved.stages["generate"].status == "failed"
    assert saved.status == "failed"
    assert (paths.stage_dir("generate") / "generation.json").is_file()


def test_generate_resume_reuses_success_and_retries_failed(tmp_path: Path) -> None:
    store, paths, manifest = _prompted_run(tmp_path, ("portrait", "landscape"))
    first = FakeImageProvider(fail_formats={"landscape"})
    _generate(paths, manifest, store, first)
    assert [fmt for fmt, _, _ in first.calls] == ["portrait", "landscape"]

    saved = read_model(paths.manifest_path, RunManifest)
    second = FakeImageProvider()
    result = _generate(paths, saved, store, second)

    assert [fmt for fmt, _, _ in second.calls] == ["landscape"]
    assert result.generated == 2
    assert result.failed == 0
    assert all(item.status == "generated" for item in result.document.formats)


def test_invalid_image_bytes_fail_only_that_format(tmp_path: Path) -> None:
    store, paths, manifest = _prompted_run(tmp_path, ("portrait", "square"))
    provider = FakeImageProvider(invalid_formats={"portrait"})

    result = _generate(paths, manifest, store, provider)

    by_format = {item.format: item for item in result.document.formats}
    assert by_format["portrait"].status == "failed"
    assert by_format["portrait"].error is not None
    assert by_format["portrait"].error.code == "invalid_image"
    assert by_format["square"].status == "generated"


def test_generate_requires_completed_prompts(tmp_path: Path) -> None:
    path = make_lesson_docx(tmp_path / "lesson.docx", ["Байланыс туралы мәтін."])
    ingest_options = default_options()
    store, paths, manifest = start_run(tmp_path, path, ingest_options)
    run_ingest(paths, manifest, store, ingest_options)
    run_classify(paths, manifest, store, planner_options(), planner=FakePlanner())
    run_plan(
        paths,
        manifest,
        store,
        plan_options(),
        planner=FakePosterPlanner(detail_count=1),
    )

    with pytest.raises(GenLessonError) as excinfo:
        _generate(paths, manifest, store, FakeImageProvider())

    assert excinfo.value.code == "stage_incomplete"


class _UsageProvider(FakeImageProvider):
    def generate(self, prompt: str, size: str) -> bytes:
        data = super().generate(prompt, size)
        self.last_call = {
            "usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
            "latency_ms": 12.5,
            "request_id": "req-generate",
        }
        return data


def test_generate_copies_last_call_and_opens_format_span(monkeypatch, tmp_path: Path) -> None:
    backend = _enable(monkeypatch)
    store, paths, manifest = _prompted_run(tmp_path, ("portrait",))

    result = _generate(paths, manifest, store, _UsageProvider())

    item = result.document.formats[0]
    assert item.usage == {
        "input_tokens": 5,
        "output_tokens": 7,
        "total_tokens": 12,
    }
    assert item.last_call is not None
    assert item.last_call["latency_ms"] == 12.5
    assert item.last_call["request_id"] == "req-generate"
    assert item.cost_usd is None

    saved = read_model(paths.stage_dir("generate") / "generation.json", GenerationDocument)
    assert saved.formats[0].last_call == item.last_call
    assert saved.formats[0].usage == item.usage

    span = next(item for item in backend.created if item["name"] == "generate:portrait")
    assert span["_parent"] is None
    assert span["input"]["format"] == "portrait"


def test_generated_format_accepts_legacy_payload() -> None:
    item = GeneratedFormat.model_validate(
        {
            "format": "portrait",
            "width": 1024,
            "height": 1536,
            "size": "1024x1536",
            "status": "failed",
            "attempts": 1,
            "prompt_sha256": "a" * 64,
        }
    )

    assert item.usage is None
    assert item.cost_usd is None
    assert item.last_call is None


class _ModerationThenSuccessClient:
    def __init__(self, response) -> None:
        self.response = response
        self.calls = 0
        self.images = SimpleNamespace(generate=self._generate)

    def _generate(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("moderation_blocked: content policy")
        return self.response


def test_moderation_retry_gets_its_own_image_observation(monkeypatch, tmp_path: Path) -> None:
    store, paths, manifest = _prompted_run(tmp_path, ("portrait",))
    backend = _enable(monkeypatch)
    response = SimpleNamespace(
        data=[
            SimpleNamespace(
                b64_json=base64.b64encode(_png_bytes()).decode("ascii"),
                url=None,
            )
        ],
        usage=SimpleNamespace(input_tokens=2, output_tokens=3, total_tokens=5),
        request_id="req-moderation-retry",
    )
    client = _ModerationThenSuccessClient(response)
    provider = OpenAIImageProvider(
        model="gpt-image-1",
        api_key="",
        client=client,
        max_retries=0,
    )

    result = _generate(paths, manifest, store, provider)

    item = result.document.formats[0]
    assert item.status == "generated"
    assert item.attempts == 2
    assert item.moderation_retry is True
    observations = [
        observation
        for observation in backend.observations
        if observation.name == "generate.image"
    ]
    assert len(observations) == 2
    assert observations[0].updates[-1]["level"] == "WARNING"
    assert provider.last_call is not None
    assert provider.last_call["request_id"] == "req-moderation-retry"
