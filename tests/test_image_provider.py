from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from genlesson import tracing
from genlesson.config import GenerationOptions
from genlesson.errors import GenLessonError
from genlesson.image import ImageProviderError, ModerationRefusal, build_image_provider
from genlesson.image.openai_image import OpenAIImageProvider

from .helpers import _png_bytes
from .test_tracing import _enable


@pytest.fixture(autouse=True)
def _clear_tracing():
    yield
    tracing.shutdown()


class _FakeImages:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _FakeClient:
    def __init__(self, response=None, error=None):
        self.images = _FakeImages(response, error)


def _response(data: bytes):
    return SimpleNamespace(
        data=[
            SimpleNamespace(
                b64_json=base64.b64encode(data).decode("ascii"),
                url=None,
            )
        ]
    )


def test_openai_image_provider_sends_size_and_decodes_bytes() -> None:
    client = _FakeClient(_response(_png_bytes()))
    provider = OpenAIImageProvider(
        model="gpt-image-1",
        api_key="",
        client=client,
        max_retries=0,
    )

    data = provider.generate("a poster prompt", "1024x1536")

    assert data.startswith(b"\x89PNG")
    assert client.images.calls[0]["model"] == "gpt-image-1"
    assert client.images.calls[0]["prompt"] == "a poster prompt"
    assert client.images.calls[0]["size"] == "1024x1536"
    assert client.images.calls[0]["n"] == 1


def test_openai_image_provider_translates_moderation(caplog) -> None:
    client = _FakeClient(error=RuntimeError("moderation_blocked: content policy"))
    provider = OpenAIImageProvider(
        model="gpt-image-1",
        api_key="",
        client=client,
        max_retries=0,
    )

    with (
        caplog.at_level("WARNING", logger="genlesson.image.openai_image"),
        pytest.raises(ModerationRefusal),
    ):
        provider.generate("prompt", "1024x1536")

    assert "moderation" in caplog.text.lower()
    assert provider.last_call is not None
    assert provider.last_call["usage"] is None
    assert provider.last_call["request_id"] is None


def test_openai_image_provider_empty_response() -> None:
    client = _FakeClient(SimpleNamespace(data=[]))
    provider = OpenAIImageProvider(
        model="gpt-image-1",
        api_key="",
        client=client,
        max_retries=0,
    )

    with pytest.raises(ImageProviderError) as excinfo:
        provider.generate("prompt", "1024x1536")
    assert excinfo.value.code == "image_failed"


def test_image_provider_requires_api_key() -> None:
    with pytest.raises(GenLessonError) as excinfo:
        OpenAIImageProvider(model="gpt-image-1", api_key="")
    assert excinfo.value.code == "image_provider_required"

    with pytest.raises(GenLessonError) as excinfo:
        build_image_provider(GenerationOptions(api_key=None), model="gpt-image-1")
    assert excinfo.value.code == "image_provider_required"


class _FlakyImages:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.kwargs: list[dict] = []

    def generate(self, **kwargs):
        self.calls += 1
        self.kwargs.append(kwargs)
        if self.calls == 1:
            raise RuntimeError("temporary timeout")
        return self.response


class _FlakyClient:
    def __init__(self, response):
        self.images = _FlakyImages(response)


def test_image_provider_backoff_retries_transient_errors(monkeypatch) -> None:
    import genlesson.image.openai_image as module

    monkeypatch.setattr(module, "_is_transient", lambda exc: True)
    client = _FlakyClient(_response(_png_bytes()))
    sleeps: list[float] = []
    provider = OpenAIImageProvider(
        model="gpt-image-1",
        api_key="",
        client=client,
        max_retries=1,
        sleep=sleeps.append,
    )

    data = provider.generate("prompt", "1024x1536")

    assert data.startswith(b"\x89PNG")
    assert client.images.calls == 2
    assert sleeps


def _usage_response(data: bytes, request_id: str = "req-1"):
    return SimpleNamespace(
        data=[
            SimpleNamespace(
                b64_json=base64.b64encode(data).decode("ascii"),
                url=None,
            )
        ],
        usage=SimpleNamespace(input_tokens=4, output_tokens=6, total_tokens=10),
        request_id=request_id,
    )


def test_image_provider_records_last_call_and_attempt_observation(monkeypatch) -> None:
    backend = _enable(monkeypatch)
    client = _FakeClient(_usage_response(_png_bytes()))
    provider = OpenAIImageProvider(
        model="gpt-image-1",
        api_key="",
        client=client,
        max_retries=1,
    )

    with tracing.stage_span("generate:portrait", run_id="run-1") as parent:
        data = provider.generate("a poster prompt", "1024x1536")

    assert data.startswith(b"\x89PNG")
    assert provider.last_call is not None
    assert provider.last_call["usage"] == {
        "input_tokens": 4,
        "output_tokens": 6,
        "total_tokens": 10,
    }
    assert provider.last_call["request_id"] == "req-1"
    assert provider.last_call["latency_ms"] >= 0

    observation = next(
        item for item in backend.observations if item.name == "generate.image"
    )
    assert observation.updates[-1]["usage_details"] == {
        "input": 4,
        "output": 6,
        "total": 10,
    }
    assert backend.created[1]["_parent"] == parent.observation_id


def test_image_provider_retries_each_get_an_observation(monkeypatch) -> None:
    import genlesson.image.openai_image as module

    backend = _enable(monkeypatch)
    monkeypatch.setattr(module, "_is_transient", lambda exc: True)
    client = _FlakyClient(_usage_response(_png_bytes()))
    provider = OpenAIImageProvider(
        model="gpt-image-1",
        api_key="",
        client=client,
        max_retries=1,
        sleep=lambda _s: None,
    )

    with tracing.stage_span("generate:portrait", run_id="run-1"):
        data = provider.generate("prompt", "1024x1536")

    assert data.startswith(b"\x89PNG")
    observations = [
        item for item in backend.observations if item.name == "generate.image"
    ]
    assert len(observations) == 2
    assert provider.last_call["request_id"] == "req-1"
