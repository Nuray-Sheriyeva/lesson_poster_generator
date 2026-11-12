from __future__ import annotations

from pathlib import Path

import pytest

from genlesson.app_config import clear_config_cache, load_app_config, set_active_config_path
from genlesson.errors import GenLessonError


@pytest.fixture(autouse=True)
def _clear():
    clear_config_cache()
    set_active_config_path(None)
    yield
    clear_config_cache()
    set_active_config_path(None)


def test_default_config_uses_current_model_names() -> None:
    config = load_app_config()

    assert config.planner.provider == "openai"
    assert config.planner.model == "gpt-6-luna"
    assert config.image.provider == "openai"
    assert config.image.model == "gpt-image-2"
    assert config.formats == ("portrait",)
    assert config.review == "off"
    assert config.review_on_conflict is False
    assert config.max_details == 8
    assert config.max_retries == 2
    assert config.tracing.enabled == "auto"
    assert config.tracing.environment == "local"
    assert config.tracing.max_input_chars == 8000
    assert config.tracing.mask_paths is True
    assert config.tracing.pricing_image == {}
    assert config.out_dir == Path("outputs")


def test_environment_overrides_config(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("POSTER_REVIEW", "on-conflict")
    monkeypatch.setenv("POSTER_REVIEW_ON_CONFLICT", "1")
    monkeypatch.setenv("POSTER_PLANNER_MODEL", "planner-from-env")
    monkeypatch.setenv("POSTER_IMAGE_MODEL", "image-from-env")
    monkeypatch.setenv("POSTER_MAX_DETAILS", "5")
    monkeypatch.setenv("POSTER_MAX_RETRIES", "1")
    monkeypatch.setenv("POSTER_OUT_DIR", str(tmp_path / "posters"))

    config = load_app_config()

    assert config.review == "on_conflict"
    assert config.review_on_conflict is True
    assert config.planner.model == "planner-from-env"
    assert config.image.model == "image-from-env"
    assert config.max_details == 5
    assert config.max_retries == 1
    assert config.out_dir == tmp_path / "posters"


def test_explicit_config_path(tmp_path: Path) -> None:
    path = tmp_path / "custom.toml"
    path.write_text(
        """
[planner]
provider = "openai"
model = "planner-custom"

[image]
provider = "openai"
model = "gpt-image-2"

[image.sizes.gpt-image-2]
portrait = "1024x1536"
landscape = "1536x1024"
square = "1024x1024"

[vision]
provider = "openai"
model = "vision-custom"

[defaults]
formats = ["square", "portrait"]
review = "on"
max_details = 4
max_retries = 0
out_dir = "custom-outputs"
""",
        encoding="utf-8",
    )

    config = load_app_config(path)

    assert config.planner.model == "planner-custom"
    assert config.vision.model == "vision-custom"
    assert config.formats == ("square", "portrait")
    assert config.review == "on"
    assert config.max_details == 4
    assert config.max_retries == 0
    assert config.out_dir == Path("custom-outputs")


def test_poster_config_env_path(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "env.toml"
    path.write_text(
        """
[planner]
provider = "openai"
model = "env-planner"
[image]
provider = "openai"
model = "gpt-image-2"
[image.sizes.gpt-image-2]
portrait = "1024x1536"
landscape = "1536x1024"
square = "1024x1024"
[vision]
provider = "openai"
model = "env-vision"
[defaults]
formats = ["portrait"]
review = "off"
max_details = 8
max_retries = 2
out_dir = "outputs"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("POSTER_CONFIG", str(path))

    config = load_app_config()

    assert config.planner.model == "env-planner"
    assert config.vision.model == "env-vision"


def _write_config(path: Path, tracing_table: str = "") -> Path:
    path.write_text(
        f"""
[planner]
provider = "openai"
model = "planner-custom"
[image]
provider = "openai"
model = "gpt-image-2"
[image.sizes.gpt-image-2]
portrait = "1024x1536"
[vision]
provider = "openai"
model = "vision-custom"
[defaults]
formats = ["portrait"]
review = "off"
max_details = 8
max_retries = 2
out_dir = "outputs"
{tracing_table}
""",
        encoding="utf-8",
    )
    return path


def test_tracing_config_parsed(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path / "tracing.toml",
        """
[tracing]
enabled = "on"
environment = "staging"
max_input_chars = 100
mask_paths = false

[tracing.pricing.image]
gpt-image-2 = 0.04
""",
    )

    config = load_app_config(path)

    assert config.tracing.enabled == "on"
    assert config.tracing.environment == "staging"
    assert config.tracing.max_input_chars == 100
    assert config.tracing.mask_paths is False
    assert config.tracing.pricing_image == {"gpt-image-2": 0.04}


def test_invalid_tracing_config_is_rejected(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path / "bad-tracing.toml",
        """
[tracing]
enabled = "sometimes"
""",
    )

    with pytest.raises(GenLessonError) as excinfo:
        load_app_config(path)

    assert excinfo.value.code == "config_invalid"
    assert "tracing.enabled" in excinfo.value.message
