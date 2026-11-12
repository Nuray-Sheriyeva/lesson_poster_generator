"""OpenAI-backed planner for content classification and language detection."""

from __future__ import annotations

import base64
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any

from .. import tracing
from ..errors import GenLessonError
from .prompts import (
    CANDIDATE_SYSTEM,
    CLASSIFY_SYSTEM,
    LANGUAGE_SYSTEM,
    PLAN_SYSTEM,
    system_prompt,
)
from .base import (
    CandidateRequest,
    CandidateSet,
    ClassificationDraft,
    ClassificationRequest,
    LanguageDecision,
    LanguageRequest,
    Planner,
    PlannerCapabilities,
    PlanRequest,
    PosterSpecDraft,
)
from .strict_schema import strict_json_schema

log = logging.getLogger(__name__)

_MAX_IMAGE_BYTES = 20 * 1024 * 1024

class OpenAIPlanner(Planner):
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        json_retries: int = 1,
        client: Any | None = None,
        sleep: Any = time.sleep,
        structured_output: bool = True,
        vision: bool = True,
        max_context: int = 200000,
        provider_name: str = "openai",
    ) -> None:
        if not api_key and client is None:
            raise GenLessonError(
                "planner_required",
                "Stage 2 needs an OpenAI API key for the Luna planner.",
                hint="Set OPENAI_API_KEY and run classify again.",
            )
        self.model = model
        self.provider_name = provider_name
        self.capabilities = PlannerCapabilities(
            structured_output=structured_output,
            vision=vision,
            max_context=max_context,
        )
        self._sleep = sleep
        self._max_retries = max_retries
        self._json_retries = json_retries
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
                    "planner_unavailable",
                    "The openai package is not installed, so Luna cannot be called.",
                    hint="Install it with: uv pip install openai",
                ) from exc
            self._sdk = openai
            self._client = client_cls(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout_s,
                max_retries=0,
            )
            self._auto_traced = tracing.uses_openai_wrapper()

    def classify(self, request: ClassificationRequest) -> ClassificationDraft:
        messages = self._classification_messages(request)
        return self._request_json(
            messages,
            ClassificationDraft,
            "classification",
            call_name="classify.segments",
        )

    def detect_language(self, request: LanguageRequest) -> LanguageDecision:
        user = (
            "Identify the poster language of this lesson-content body text.\n\n"
            "LESSON BODY TEXT:\n"
            f"{request.text}"
        )
        messages = [
            {"role": "system", "content": system_prompt("language")},
            {"role": "user", "content": user},
        ]
        return self._request_json(
            messages,
            LanguageDecision,
            "language detection",
            call_name="classify.language",
        )

    def plan(self, request: PlanRequest) -> PosterSpecDraft:
        messages = self._plan_messages(request)
        return self._request_json(
            messages,
            PosterSpecDraft,
            "poster plan",
            call_name="plan.spec",
            response_format=(
                _strict_format("poster_spec", PosterSpecDraft)
                if self.capabilities.structured_output
                else {"type": "json_object"}
            ),
        )

    def extract_candidates(self, request: CandidateRequest) -> CandidateSet:
        messages = self._candidate_messages(request)
        return self._request_json(
            messages,
            CandidateSet,
            "candidate extraction",
            call_name="plan.candidates",
            response_format=(
                _strict_format("candidate_ideas", CandidateSet)
                if self.capabilities.structured_output
                else {"type": "json_object"}
            ),
        )

    def _plan_messages(self, request: PlanRequest) -> list[dict]:
        header = (
            f"Stage 2 poster language: {request.language}\n"
            f"Lesson title: {request.title or '(untitled)'}\n"
            f"Dominant topic: {request.dominant_topic or '(not provided)'}\n"
            f"Maximum details: {request.max_details}\n"
        )
        if request.candidates is not None:
            body = (
                "MODE: merged candidate ideas extracted from the full lesson.\n"
                "Select and synthesize the final poster spec from these candidates only.\n\n"
                "CANDIDATES (JSON):\n"
                f"{json.dumps(request.candidates.model_dump(mode='json'), ensure_ascii=False)}\n"
            )
        else:
            body = (
                "MODE: complete lesson body text.\n\n"
                "LESSON BODY TEXT:\n"
                f"{request.text or ''}\n"
            )
        content: list[dict] = [{"type": "text", "text": header + "\n" + body}]
        for image in request.images:
            content.append(
                {"type": "text", "text": f"Source page {image.page} image (context only):"}
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(image.path), "detail": "high"},
                }
            )
        return [
            {"role": "system", "content": system_prompt("plan")},
            {"role": "user", "content": content},
        ]

    def _candidate_messages(self, request: CandidateRequest) -> list[dict]:
        pages = ", ".join(str(page) for page in request.source_pages) or "(unknown)"
        text = (
            f"Stage 2 poster language: {request.language}\n"
            f"Lesson title: {request.title or '(untitled)'}\n"
            f"Dominant topic: {request.dominant_topic or '(not provided)'}\n"
            f"Chunk: {request.chunk_index} of {request.chunk_count}\n"
            f"Source pages in this chunk: {pages}\n"
            f"Maximum candidate details: {request.max_candidates}\n\n"
            "CHUNK TEXT:\n"
            f"{request.text}"
        )
        return [
            {"role": "system", "content": system_prompt("candidate")},
            {"role": "user", "content": text},
        ]

    def _classification_messages(self, request: ClassificationRequest) -> list[dict]:
        segment_payload = [
            {
                "id": segment.id,
                "page": segment.page,
                "source": segment.text_source,
                "text": segment.text,
            }
            for segment in request.segments
        ]
        preamble = (
            f"Lesson title: {request.title or '(untitled)'}\n"
            f"Document is scanned/OCR-only: {'yes' if request.scanned_document else 'no'}\n"
            f"Segment count: {len(request.segments)}\n\n"
            "SEGMENTS (JSON):\n"
            f"{json.dumps(segment_payload, ensure_ascii=False)}\n\n"
            "Classify every segment and return the JSON object."
        )
        content: list[dict] = [{"type": "text", "text": preamble}]
        for image in request.images if self.capabilities.vision else ():
            content.append(
                {"type": "text", "text": f"Page {image.page} image (context only):"}
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(image.path), "detail": "high"},
                }
            )
        return [
            {"role": "system", "content": system_prompt("classify")},
            {"role": "user", "content": content},
        ]

    def _request_json(
        self,
        messages: list[dict],
        model_cls: type,
        label: str,
        *,
        call_name: str | None = None,
        response_format: dict | None = None,
    ) -> Any:
        last_problem: str | None = None
        working = list(messages)
        for attempt in range(self._json_retries + 1):
            raw = self._complete(working, response_format=response_format, call_name=call_name)
            try:
                payload = _parse_json_object(raw)
                return model_cls.model_validate(payload)
            except Exception as exc:
                last_problem = str(exc)
                log.warning(
                    "Invalid %s JSON from %s (attempt %s): %s",
                    label,
                    self.provider_name,
                    attempt + 1,
                    exc,
                )
                if attempt >= self._json_retries:
                    break
                working = working + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            f"That response was invalid: {last_problem}. "
                            "Return only one valid JSON object with exactly the requested schema. "
                            "No markdown fences, no commentary."
                        ),
                    },
                ]
        raise GenLessonError(
            "planner_response_invalid",
            f"{self.provider_name.title()} returned invalid JSON for {label} "
            f"after {self._json_retries + 1} attempt(s).",
            hint="Retry the stage, or switch --planner-model.",
            details={"reason": last_problem},
        )

    def _complete(
        self,
        messages: list[dict],
        *,
        response_format: dict | None = None,
        call_name: str | None = None,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._call_completion(
                    messages,
                    response_format=response_format,
                    call_name=call_name or "planner.call",
                )
            except Exception as exc:
                if self._sdk is None or not self._should_retry(exc):
                    raise self._translate_error(exc) from exc
                last_error = exc
                if attempt >= self._max_retries:
                    break
                delay = min(2.0 * (2**attempt), 30.0) + random.uniform(0.0, 0.25)
                log.warning(
                    "%s call failed (attempt %s/%s); retrying in %.1fs: %s",
                    self.provider_name,
                    attempt + 1,
                    self._max_retries + 1,
                    delay,
                    exc,
                )
                self._sleep(delay)

        code = "planner_failed"
        hint = "Check API status, then resume the run."
        if self._sdk is not None and isinstance(last_error, self._sdk.RateLimitError):
            code = "planner_rate_limited"
            hint = "Wait for the rate limit or check quota, then resume the run."
        raise GenLessonError(
            code,
            f"{self.provider_name.title()} planner call failed after "
            f"{self._max_retries + 1} attempt(s).",
            hint=hint,
            details={"last_error": str(last_error)},
        ) from last_error

    def _call_completion(
        self,
        messages: list[dict],
        *,
        response_format: dict | None,
        call_name: str,
    ) -> str:
        kwargs = tracing.call_kwargs(name=call_name)
        if tracing.is_enabled() and not self._auto_traced:
            with tracing.generation(
                call_name,
                model=self.model,
                input=messages,
                metadata={"provider": self.provider_name},
            ) as generation:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format=response_format or {"type": "json_object"},
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
            response_format=response_format or {"type": "json_object"},
            **kwargs,
        )
        return _response_text(response)

    def _should_retry(self, exc: Exception) -> bool:
        sdk = self._sdk
        if isinstance(exc, (sdk.RateLimitError, sdk.APITimeoutError, sdk.APIConnectionError)):
            return True
        if isinstance(exc, sdk.APIStatusError):
            return exc.status_code >= 500
        return False

    def _translate_error(self, exc: Exception) -> GenLessonError:
        sdk = self._sdk
        if sdk is None:
            return GenLessonError(
            "planner_failed", f"The {self.provider_name} planner failed: {exc}"
        )
        if isinstance(exc, sdk.AuthenticationError):
            return GenLessonError(
                "planner_auth_failed",
                f"{self.provider_name.title()} rejected the API key for the planner.",
                hint=f"Check {self.provider_name.upper()}_API_KEY.",
            )
        if isinstance(exc, sdk.PermissionDeniedError):
            return GenLessonError(
                "planner_auth_failed",
                f"This API key does not have permission to use the {self.provider_name} planner.",
            )
        if isinstance(exc, sdk.APITimeoutError):
            return GenLessonError(
                "planner_failed",
                f"The {self.provider_name} planner timed out.",
                hint="Try again later or raise --planner-timeout.",
            )
        if isinstance(exc, sdk.APIStatusError):
            return GenLessonError(
                "planner_failed",
                f"The {self.provider_name} planner request failed "
                f"(HTTP {exc.status_code}).",
                details={"reason": str(exc)},
            )
        return GenLessonError(
            "planner_failed", f"The {self.provider_name} planner failed: {exc}"
        )


def _strict_format(name: str, model: type) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": strict_json_schema(model),
        },
    }


def _image_data_url(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise GenLessonError(
            "artifact_missing", f"Could not read page image: {path}", details={"reason": str(exc)}
        ) from exc
    if len(data) > _MAX_IMAGE_BYTES:
        raise GenLessonError(
            "planner_failed",
            f"Page image is too large for the planner: {path.name}",
            hint="Lower the ingest --dpi and start a new run.",
            details={"bytes": len(data), "limit": _MAX_IMAGE_BYTES},
        )
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


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
        raise GenLessonError("planner_failed", "Luna returned no choices.")
    message = choices[0].message
    refusal = getattr(message, "refusal", None)
    if refusal:
        raise GenLessonError(
            "planner_failed",
            "Luna refused the Stage 2 request.",
            details={"refusal": str(refusal)},
        )
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
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
        return "\n".join(parts)
    return "" if content is None else str(content)


def _parse_json_object(text: str) -> dict:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response did not contain a JSON object")
    parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("response JSON was not an object")
    return parsed


class DeepSeekPlanner(OpenAIPlanner):
    """DeepSeek chat-completions adapter.

    DeepSeek is OpenAI-compatible but does not provide strict JSON Schema or
    vision on the configured chat model. It therefore reports those capabilities
    false and uses JSON-object mode plus Pydantic validation.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        json_retries: int = 1,
        max_context: int = 128000,
        client: Any | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or "https://api.deepseek.com",
            timeout_s=timeout_s,
            max_retries=max_retries,
            json_retries=json_retries,
            client=client,
            sleep=sleep,
            structured_output=False,
            vision=False,
            max_context=max_context,
            provider_name="deepseek",
        )
