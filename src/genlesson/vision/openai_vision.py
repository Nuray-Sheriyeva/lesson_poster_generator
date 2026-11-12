"""OpenAI-backed vision transcription for scanned PDF pages."""

from __future__ import annotations

import base64
import logging
import random
import time
from pathlib import Path
from typing import Any

from .. import tracing
from ..errors import GenLessonError
from .base import Transcriber

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the OCR stage of a lesson-poster pipeline.
Transcribe every piece of text in the image exactly as written.
- Keep the original language, script, spelling, and punctuation. Never translate.
- Preserve reading order, paragraph breaks, and table structure.
- Write table rows as cells separated by " | ".
- Write mathematical notation as readable plain text.
- Do not describe pictures, summarize, explain, or add commentary.
- Return only the transcription. If the page contains no text, return an empty string."""

_MAX_IMAGE_BYTES = 20 * 1024 * 1024


class OpenAIVisionTranscriber(Transcriber):
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        client: Any | None = None,
        sleep: Any = time.sleep,
        call_name: str = "ingest.transcribe_page",
    ) -> None:
        if not api_key and client is None:
            raise GenLessonError(
                "vision_required",
                "Transcribing a scanned PDF needs an OpenAI API key.",
                hint="Set OPENAI_API_KEY, or run with --vision none to skip OCR.",
            )
        self.model = model
        self._sleep = sleep
        self.max_retries = max_retries
        self._call_name = call_name
        self._sdk = None
        self._auto_traced = False
        if client is not None:
            self._client = client
        else:
            try:
                import openai

                client_cls = tracing.openai_client_factory()
            except ImportError as exc:
                raise GenLessonError(
                    "vision_unavailable",
                    "The openai package is not installed, so scanned pages cannot be transcribed.",
                    hint="Install it with: uv pip install openai",
                ) from exc
            self._sdk = openai
            self._client = client_cls(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout_s,
                max_retries=0,  # Retries are handled here so they are observable/testable.
            )
            self._auto_traced = tracing.uses_openai_wrapper()

    def transcribe(self, image_path: Path, page_number: int) -> str:
        try:
            image_bytes = image_path.read_bytes()
        except OSError as exc:
            raise GenLessonError(
                "vision_failed",
                f"Could not read rendered page image: {image_path}",
                details={"reason": str(exc)},
            ) from exc
        if len(image_bytes) > _MAX_IMAGE_BYTES:
            raise GenLessonError(
                "vision_failed",
                f"Rendered page {page_number} is too large to send to the vision model.",
                hint="Lower --dpi and resume the run.",
                details={"bytes": len(image_bytes), "limit": _MAX_IMAGE_BYTES},
            )

        data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Transcribe page {page_number}."},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "high"},
                    },
                ],
            },
        ]

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._call_vision(messages, data_url, page_number)
            except Exception as exc:  # narrowed below against the SDK hierarchy
                if self._sdk is None:
                    raise GenLessonError(
                        "vision_failed",
                        f"Vision transcription failed on page {page_number}: {exc}",
                    ) from exc
                if not self._sdk_should_retry(exc):
                    raise self._translate_error(exc, page_number) from exc
                last_error = exc
                if attempt >= self.max_retries:
                    break
                delay = min(2.0 * (2**attempt), 30.0) + random.uniform(0.0, 0.25)
                log.warning(
                    "Vision call failed on page %s (attempt %s/%s); retrying in %.1fs: %s",
                    page_number,
                    attempt + 1,
                    self.max_retries + 1,
                    delay,
                    exc,
                )
                self._sleep(delay)

        code = "vision_failed"
        hint = "Check API status, then resume the run."
        if self._sdk is not None and isinstance(last_error, self._sdk.RateLimitError):
            code = "vision_rate_limited"
            hint = "Wait for the rate limit or check quota, then resume the run."
        raise GenLessonError(
            code,
            f"Vision transcription failed on page {page_number} after "
            f"{self.max_retries + 1} attempt(s).",
            hint=hint,
            details={"last_error": str(last_error)},
        ) from last_error

    def _call_vision(self, messages: list[dict], data_url: str, page_number: int) -> str:
        kwargs = tracing.call_kwargs(name=self._call_name)
        if tracing.is_enabled() and not self._auto_traced:
            with tracing.generation(
                self._call_name,
                model=self.model,
                input={"page": page_number, "image": data_url},
                metadata={"provider": "openai", "page": page_number},
            ) as generation:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kwargs,
                )
                text = _response_text(response)
                usage = _usage_from_response(response)
                update = {"output": text}
                if usage:
                    update["usage_details"] = usage
                generation.update(**update)
                return text
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        return _response_text(response)

    def _sdk_should_retry(self, exc: Exception) -> bool:
        sdk = self._sdk
        if isinstance(exc, (sdk.RateLimitError, sdk.APITimeoutError, sdk.APIConnectionError)):
            return True
        if isinstance(exc, sdk.APIStatusError):
            return exc.status_code >= 500
        return False

    def _translate_error(self, exc: Exception, page_number: int) -> GenLessonError:
        sdk = self._sdk
        if isinstance(exc, sdk.AuthenticationError):
            return GenLessonError(
                "vision_auth_failed",
                "OpenAI rejected the API key while transcribing a scanned page.",
                hint="Check OPENAI_API_KEY.",
            )
        if isinstance(exc, sdk.PermissionDeniedError):
            return GenLessonError(
                "vision_auth_failed",
                "This API key does not have permission to use the vision model.",
            )
        if isinstance(exc, sdk.APITimeoutError):
            return GenLessonError(
                "vision_failed",
                f"Vision transcription timed out on page {page_number}.",
                hint="Try again later or raise --vision-timeout.",
            )
        if isinstance(exc, sdk.APIStatusError):
            return GenLessonError(
                "vision_failed",
                f"Vision transcription failed on page {page_number} "
                f"(HTTP {exc.status_code}).",
                details={"reason": str(exc)},
            )
        return GenLessonError(
            "vision_failed",
            f"Vision transcription failed on page {page_number}: {exc}",
        )


def _usage_value(usage: Any, name: str) -> Any:
    if isinstance(usage, dict):
        return usage.get(name)
    return getattr(usage, name, None)


def _usage_from_response(response: Any) -> dict[str, int] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    input_tokens = _usage_value(usage, "input_tokens")
    if input_tokens is None:
        input_tokens = _usage_value(usage, "prompt_tokens")
    output_tokens = _usage_value(usage, "output_tokens")
    if output_tokens is None:
        output_tokens = _usage_value(usage, "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens")
    details: dict[str, int] = {}
    if input_tokens is not None:
        details["input"] = int(input_tokens)
    if output_tokens is not None:
        details["output"] = int(output_tokens)
    if total_tokens is not None:
        details["total"] = int(total_tokens)
    elif details:
        details["total"] = sum(details.values())
    return details or None


def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise GenLessonError("vision_failed", "Vision model returned no choices.")
    message = choices[0].message
    refusal = getattr(message, "refusal", None)
    if refusal:
        raise GenLessonError(
            "vision_failed",
            "Vision model refused to transcribe the page.",
            details={"refusal": str(refusal)},
        )
    content = message.content
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                value = part.get("text")
                if value:
                    parts.append(str(value))
            else:
                value = getattr(part, "text", None)
                if value:
                    parts.append(str(value))
        text = "\n".join(parts)
    else:
        text = "" if content is None else str(content)

    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return text
