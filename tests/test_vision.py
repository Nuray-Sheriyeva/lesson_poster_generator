from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from genlesson.errors import GenLessonError
from genlesson.vision import build_transcriber
from genlesson.vision.cache import TranscriptionCache
from genlesson.vision.openai_vision import OpenAIVisionTranscriber, _response_text

from .helpers import default_options


def _png(tmp_path: Path) -> Path:
    import pymupdf

    path = tmp_path / "page.png"
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 2, 2), 0)
    pixmap.clear_with(255)
    pixmap.save(path)
    return path


class _FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_transcriber_sends_page_image_and_returns_text(tmp_path: Path) -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="  Lesson text  ", refusal=None))]
    )
    client = _FakeClient(response)
    transcriber = OpenAIVisionTranscriber(
        model="gpt-6-luna", api_key="", client=client, max_retries=0
    )

    text = transcriber.transcribe(_png(tmp_path), 3)

    assert text == "Lesson text"
    assert client.calls[0]["model"] == "gpt-6-luna"
    content = client.calls[0]["messages"][1]["content"]
    assert content[0]["text"] == "Transcribe page 3."
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_vision_requires_a_key_for_scanned_pages() -> None:
    options = default_options(vision_mode="openai", openai_api_key=None)
    with pytest.raises(GenLessonError) as excinfo:
        build_transcriber(options)
    assert excinfo.value.code == "vision_required"


def test_vision_disabled_raises_when_called() -> None:
    options = default_options(vision_mode="none")
    with pytest.raises(GenLessonError) as excinfo:
        build_transcriber(options)
    assert excinfo.value.code == "vision_required"


def test_empty_refusal_is_reported() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, refusal="cannot read"))]
    )
    with pytest.raises(GenLessonError) as excinfo:
        _response_text(response)
    assert excinfo.value.code == "vision_failed"


def test_cache_is_keyed_by_image_and_model(tmp_path: Path) -> None:
    cache = TranscriptionCache(tmp_path / "cache")
    cache.put(1, image_sha256="a" * 64, model="model-a", text="hello")

    assert cache.get(1, image_sha256="a" * 64, model="model-a") == "hello"
    assert cache.get(1, image_sha256="b" * 64, model="model-a") is None
    assert cache.get(1, image_sha256="a" * 64, model="model-b") is None


class _Retryable(Exception):
    pass


class _OtherApiError(Exception):
    pass


class _FlakyClient:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise _Retryable("temporary")
        return self.response


def test_transcriber_retries_transient_failures(tmp_path: Path) -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Recovered", refusal=None))]
    )
    client = _FlakyClient(response)
    transcriber = OpenAIVisionTranscriber(
        model="gpt-6-luna", api_key="", client=client, max_retries=1, sleep=lambda _s: None
    )
    transcriber._sdk = SimpleNamespace(
        RateLimitError=_Retryable,
        APITimeoutError=_Retryable,
        APIConnectionError=_Retryable,
        APIStatusError=_OtherApiError,
        AuthenticationError=_OtherApiError,
        PermissionDeniedError=_OtherApiError,
        OpenAIError=_OtherApiError,
    )

    assert transcriber.transcribe(_png(tmp_path), 1) == "Recovered"
    assert client.calls == 2
