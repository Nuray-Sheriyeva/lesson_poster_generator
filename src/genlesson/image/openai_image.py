"""OpenAI Images API implementation of the image-provider interface."""

from __future__ import annotations

import base64
import logging
import random
import time
import urllib.request
from typing import Any

from .. import tracing
from ..errors import GenLessonError
from .base import ImageProvider, ImageProviderError, ModerationRefusal

log = logging.getLogger(__name__)


class OpenAIImageProvider(ImageProvider):
    name = "openai"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout_s: float = 180.0,
        max_retries: int = 3,
        client: Any | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        if not api_key and client is None:
            raise GenLessonError(
                "image_provider_required",
                "Image generation needs an OpenAI API key.",
                hint="Set OPENAI_API_KEY and run generate again.",
            )
        self.model = model
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._sleep = sleep
        self._sdk = None
        self.last_call: dict[str, Any] | None = None
        if client is not None:
            self._client = client
        else:
            try:
                import openai
            except ImportError as exc:
                raise GenLessonError(
                    "image_provider_unavailable",
                    "The openai package is not installed, so images cannot be generated.",
                    hint="Install it with: uv pip install openai",
                ) from exc
            self._sdk = openai
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout_s,
                max_retries=0,
            )

    def generate(self, prompt: str, size: str) -> bytes:
        self.last_call = None
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            self.last_call = None
            started = time.monotonic()
            try:
                with tracing.generation(
                    "generate.image",
                    model=self.model,
                    input={"prompt": prompt, "size": size},
                    metadata={"provider": "openai", "attempt": attempt + 1},
                ) as generation:
                    try:
                        response = self._client.images.generate(
                            model=self.model,
                            prompt=prompt,
                            size=size,
                            n=1,
                        )
                        usage = _image_usage(response)
                        latency_ms = int((time.monotonic() - started) * 1000)
                        self.last_call = _last_call(
                            usage=usage,
                            latency_ms=latency_ms,
                            request_id=_request_id(response),
                        )
                        data = _image_bytes(response, timeout_s=self._timeout_s)
                    except Exception as exc:
                        if self.last_call is None:
                            self.last_call = _last_call(
                                usage=None,
                                latency_ms=int((time.monotonic() - started) * 1000),
                                request_id=None,
                            )
                        if _is_moderation(exc):
                            generation.update(level="WARNING", status_message=str(exc))
                        else:
                            generation.update(level="ERROR", status_message=str(exc))
                        raise
                    update: dict[str, Any] = {
                        "output": {
                            "status": "generated",
                            "bytes": len(data),
                            "latency_ms": latency_ms,
                            "request_id": self.last_call.get("request_id")
                            if self.last_call
                            else None,
                        }
                    }
                    usage_details = _usage_details(usage)
                    if usage_details:
                        update["usage_details"] = usage_details
                    generation.update(**update)
                    return data
            except Exception as exc:
                if _is_moderation(exc):
                    log.warning(
                        "Image generation moderation refusal (attempt %s/%s): %s",
                        attempt + 1,
                        self._max_retries + 1,
                        exc,
                    )
                    raise ModerationRefusal(
                        f"Image generation was refused by moderation: {exc}"
                    ) from exc
                if _is_transient(exc):
                    last_error = exc
                    if attempt >= self._max_retries:
                        break
                    delay = min(2.0 * (2**attempt), 30.0) + random.uniform(0.0, 0.25)
                    log.warning(
                        "Image generation failed (attempt %s/%s); retrying in %.1fs: %s",
                        attempt + 1,
                        self._max_retries + 1,
                        delay,
                        exc,
                    )
                    self._sleep(delay)
                    continue
                raise ImageProviderError(
                    "image_failed",
                    f"Image generation failed: {exc}",
                ) from exc

        raise ImageProviderError(
            "image_failed",
            f"Image generation failed after {self._max_retries + 1} attempt(s).",
            retryable=True,
        ) from last_error


def _is_moderation(exc: Exception) -> bool:
    text = f"{getattr(exc, 'code', '')} {exc}".casefold()
    tokens = (
        "moderation",
        "content_policy",
        "content policy",
        "safety system",
        "safety_violation",
        "safety violation",
        "moderation_blocked",
    )
    return any(token in text for token in tokens)


def _is_transient(exc: Exception) -> bool:
    try:
        import openai
    except ImportError:
        return False
    if isinstance(
        exc,
        (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError),
    ):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500
    return False


def _image_usage(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        raw = dict(usage)
    elif hasattr(usage, "model_dump"):
        raw = usage.model_dump(mode="json", exclude_none=True)
    else:
        raw = {
            name: getattr(usage, name)
            for name in (
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "prompt_tokens",
                "completion_tokens",
            )
            if getattr(usage, name, None) is not None
        }
    return raw or None


def _usage_details(usage: dict[str, Any] | None) -> dict[str, int] | None:
    if not usage:
        return None
    details: dict[str, int] = {}
    input_tokens = usage.get("input_tokens")
    if input_tokens is None:
        input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("output_tokens")
    if output_tokens is None:
        output_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    for key, value in (
        ("input", input_tokens),
        ("output", output_tokens),
        ("total", total_tokens),
    ):
        if value is None:
            continue
        try:
            details[key] = int(value)
        except (TypeError, ValueError):
            continue
    if "total" not in details and details:
        details["total"] = sum(details.values())
    return details or None


def _request_id(response: Any) -> str | None:
    for attribute in ("request_id", "_request_id"):
        value = getattr(response, attribute, None)
        if value:
            return str(value)
    return None


def _last_call(
    *,
    usage: dict[str, Any] | None,
    latency_ms: int,
    request_id: str | None,
) -> dict[str, Any]:
    return {
        "usage": usage,
        "latency_ms": latency_ms,
        "request_id": request_id,
    }


def _image_bytes(response: Any, *, timeout_s: float) -> bytes:
    data = getattr(response, "data", None) or []
    if not data:
        raise ImageProviderError("image_failed", "Image provider returned no image data.")
    item = data[0]
    encoded = getattr(item, "b64_json", None)
    if encoded:
        try:
            return base64.b64decode(encoded)
        except Exception as exc:
            raise ImageProviderError(
                "image_failed", "Image provider returned invalid base64."
            ) from exc
    url = getattr(item, "url", None)
    if url:
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as handle:
                return handle.read()
        except Exception as exc:
            raise ImageProviderError(
                "image_failed", f"Could not download generated image: {exc}"
            ) from exc
    raise ImageProviderError("image_failed", "Image provider returned no image bytes.")
