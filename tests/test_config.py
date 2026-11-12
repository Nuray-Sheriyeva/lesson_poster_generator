from __future__ import annotations

import pytest

from genlesson.config import IngestOptions, stage_dir_name
from genlesson.errors import GenLessonError


def test_invalid_env_dpi_is_ignored_when_flag_overrides(monkeypatch) -> None:
    monkeypatch.setenv("GENLESSON_RENDER_DPI", "not-a-number")
    options = IngestOptions.from_env(render_dpi=144)
    assert options.render_dpi == 144


def test_invalid_env_dpi_raises_without_override(monkeypatch) -> None:
    monkeypatch.setenv("GENLESSON_RENDER_DPI", "not-a-number")
    with pytest.raises(GenLessonError) as excinfo:
        IngestOptions.from_env()
    assert excinfo.value.code == "invalid_arguments"


def test_unknown_stage_is_clear() -> None:
    with pytest.raises(GenLessonError) as excinfo:
        stage_dir_name("bogus")
    assert excinfo.value.code == "unknown_stage"


def test_options_validation() -> None:
    with pytest.raises(GenLessonError):
        IngestOptions(render_dpi=10).validate()
    with pytest.raises(GenLessonError):
        IngestOptions(vision_timeout_s=0).validate()
    with pytest.raises(GenLessonError):
        IngestOptions(vision_max_retries=99).validate()


def test_planner_env_max_images_override(monkeypatch) -> None:
    from genlesson.config import PlannerOptions

    monkeypatch.setenv("GENLESSON_MAX_PAGE_IMAGES", "not-a-number")
    options = PlannerOptions.from_env(max_page_images=3)
    assert options.max_page_images == 3


def test_planner_options_validation() -> None:
    from genlesson.config import PlannerOptions

    with pytest.raises(GenLessonError):
        PlannerOptions(model="").validate()
    with pytest.raises(GenLessonError):
        PlannerOptions(timeout_s=0).validate()
    with pytest.raises(GenLessonError):
        PlannerOptions(max_page_images=99).validate()


def test_plan_env_overrides_and_validation(monkeypatch) -> None:
    from genlesson.config import PlanOptions

    monkeypatch.setenv("GENLESSON_MAX_DETAILS", "not-a-number")
    options = PlanOptions.from_env(max_details=4)
    assert options.max_details == 4

    with pytest.raises(GenLessonError):
        PlanOptions(max_details=0).validate()
    with pytest.raises(GenLessonError):
        PlanOptions(context_char_limit=100).validate()
    with pytest.raises(GenLessonError):
        PlanOptions(context_char_limit=4000, chunk_char_limit=9000).validate()


def test_review_and_prompt_options() -> None:
    from genlesson.config import PromptOptions, ReviewOptions

    ReviewOptions(mode="on").validate()
    with pytest.raises(GenLessonError):
        ReviewOptions(mode="sometimes").validate()

    PromptOptions(formats=("portrait", "square"), image_model="gpt-image-1").validate()
    with pytest.raises(GenLessonError):
        PromptOptions(formats=(), image_model="gpt-image-1").validate()
    with pytest.raises(GenLessonError):
        PromptOptions(formats=("portrait", "portrait"), image_model="gpt-image-1").validate()
    with pytest.raises(GenLessonError):
        PromptOptions(formats=("portrait",), image_model="").validate()


def test_generation_options_validation(monkeypatch) -> None:
    from genlesson.config import GenerationOptions

    GenerationOptions(api_key=None, timeout_s=30, max_retries=2).validate()
    with pytest.raises(GenLessonError):
        GenerationOptions(timeout_s=1).validate()
    with pytest.raises(GenLessonError):
        GenerationOptions(max_retries=99).validate()

    monkeypatch.setenv("GENLESSON_IMAGE_TIMEOUT", "not-a-number")
    options = GenerationOptions.from_env(timeout_s=30)
    assert options.timeout_s == 30


def test_verification_options_validation(monkeypatch) -> None:
    from genlesson.config import VerificationOptions

    VerificationOptions(api_key=None, similarity_threshold=0.9, max_retries=2).validate()
    with pytest.raises(GenLessonError):
        VerificationOptions(similarity_threshold=0.4).validate()
    with pytest.raises(GenLessonError):
        VerificationOptions(max_retries=99).validate()
    with pytest.raises(GenLessonError):
        VerificationOptions(vision_timeout_s=1).validate()

    monkeypatch.setenv("GENLESSON_VERIFY_SIMILARITY", "not-a-number")
    options = VerificationOptions.from_env(similarity_threshold=0.8)
    assert options.similarity_threshold == 0.8
