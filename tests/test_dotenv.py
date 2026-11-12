from __future__ import annotations

import os
from pathlib import Path

from genlesson.dotenv import load_dotenv


def test_dotenv_loads_credentials_and_ignores_legacy_config(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("POSTER_REVIEW", raising=False)
    monkeypatch.delenv("GENLESSON_IMAGE_MODEL", raising=False)
    path = tmp_path / ".env"
    path.write_text(
        "OPENAI_API_KEY=test-key\n"
        "POSTER_REVIEW=on\n"
        "GENLESSON_IMAGE_MODEL=legacy-model\n",
        encoding="utf-8",
    )

    load_dotenv(path)

    try:
        assert os.environ["OPENAI_API_KEY"] == "test-key"
        assert os.environ["POSTER_REVIEW"] == "on"
        assert "GENLESSON_IMAGE_MODEL" not in os.environ
    finally:
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("POSTER_REVIEW", None)


def test_existing_environment_wins(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "already-set")
    path = tmp_path / ".env"
    path.write_text("OPENAI_API_KEY=from-file\n", encoding="utf-8")

    load_dotenv(path)

    assert os.environ["OPENAI_API_KEY"] == "already-set"


def test_dotenv_loads_langfuse_credentials(tmp_path: Path, monkeypatch) -> None:
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / ".env"
    path.write_text(
        "LANGFUSE_PUBLIC_KEY=pk-test\n"
        "LANGFUSE_SECRET_KEY=sk-test\n"
        "LANGFUSE_HOST=https://langfuse.example\n",
        encoding="utf-8",
    )

    try:
        load_dotenv(path)

        assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk-test"
        assert os.environ["LANGFUSE_SECRET_KEY"] == "sk-test"
        assert os.environ["LANGFUSE_HOST"] == "https://langfuse.example"
    finally:
        os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
        os.environ.pop("LANGFUSE_SECRET_KEY", None)
        os.environ.pop("LANGFUSE_HOST", None)
