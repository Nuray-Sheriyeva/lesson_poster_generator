"""Anthropic Messages adapter for the planner boundary."""

from __future__ import annotations

import base64
import json
import logging
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import tracing
from ..errors import GenLessonError
from .base import (
    CandidateRequest,
    CandidateSet,
    ClassificationDraft,
    ClassificationRequest,
    LanguageDecision,
    LanguageRequest,
    PageImage,
    Planner,
    PlannerCapabilities,
    PlanRequest,
    PosterSpecDraft,
)
from .openai_planner import _parse_json_object
from .prompts import system_prompt
from .strict_schema import strict_json_schema

log = logging.getLogger(__name__)

_MAX_IMAGE_BYTES = 20 * 1024 * 1024


class AnthropicPlanner(Planner):
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        max_context: int = 200000,
        post_json: Callable[[str, dict, dict, float], dict] | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        if not api_key and post_json is None:
            raise GenLessonError(
                "planner_required",
                "The Anthropic planner needs an API key.",
                hint="Set ANTHROPIC_API_KEY and run classify again.",
            )
        self.model = model
        self.capabilities = PlannerCapabilities(
            structured_output=True,
            vision=True,
            max_context=max_context,
        )
        self._api_key = api_key
        self._base_url = (base_url or "https://api.anthropic.com").rstrip("/")
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._sleep = sleep
        self._post_json = post_json or _default_post_json

    def close(self) -> None:
        pass

    def __enter__(self) -> AnthropicPlanner:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def classify(self, request: ClassificationRequest) -> ClassificationDraft:
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
        content = self._content(preamble, request.images)
        payload = self._tool_call(
            system_prompt("classify"),
            content,
            ClassificationDraft,
            "emit_classification",
            call_name="classify.segments",
        )
        return ClassificationDraft.model_validate(payload)

    def detect_language(self, request: LanguageRequest) -> LanguageDecision:
        user = (
            "Identify the poster language of this lesson-content body text.\n\n"
            "LESSON BODY TEXT:\n"
            f"{request.text}"
        )
        payload = self._tool_call(
            system_prompt("language"),
            [{"type": "text", "text": user}],
            LanguageDecision,
            "emit_language",
            call_name="classify.language",
        )
        return LanguageDecision.model_validate(payload)

    def plan(self, request: PlanRequest) -> PosterSpecDraft:
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
        payload = self._tool_call(
            system_prompt("plan"),
            self._content(header + "\n" + body, request.images),
            PosterSpecDraft,
            "emit_poster_spec",
            call_name="plan.spec",
        )
        return PosterSpecDraft.model_validate(payload)

    def extract_candidates(self, request: CandidateRequest) -> CandidateSet:
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
        payload = self._tool_call(
            system_prompt("candidate"),
            [{"type": "text", "text": text}],
            CandidateSet,
            "emit_candidates",
            call_name="plan.candidates",
        )
        return CandidateSet.model_validate(payload)

    def _content(self, text: str, images: tuple[PageImage, ...]) -> list[dict]:
        content: list[dict] = [{"type": "text", "text": text}]
        for image in images:
            content.append({"type": "text", "text": f"Page {image.page} image (context only):"})
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": _image_base64(image.path),
                    },
                }
            )
        return content

    def _tool_call(
        self,
        system: str,
        content: list[dict],
        model_cls: type,
        tool_name: str,
        *,
        call_name: str = "planner.call",
    ) -> dict:
        payload = {
            "model": self.model,
            "max_tokens": 8192,
            "system": system.replace("Luna", "the planner"),
            "messages": [{"role": "user", "content": content}],
            "tools": [
                {
                    "name": tool_name,
                    "description": "Return the requested structured JSON object.",
                    "input_schema": strict_json_schema(model_cls),
                }
            ],
            "tool_choice": {"type": "tool", "name": tool_name},
        }
        data = self._post(payload, call_name=call_name)
        if data.get("stop_reason") == "refusal":
            raise GenLessonError(
                "planner_refused",
                f"The Anthropic planner refused the {tool_name} request.",
            )
        for block in data.get("content") or []:
            if block.get("type") == "tool_use" and block.get("name") == tool_name:
                value = block.get("input")
                if isinstance(value, dict):
                    return value
        for block in data.get("content") or []:
            if block.get("type") == "text":
                try:
                    return _parse_json_object(block.get("text") or "")
                except Exception:
                    continue
        raise GenLessonError(
            "planner_response_invalid",
            f"The Anthropic planner returned no structured {tool_name} tool call.",
        )

    def _post(self, payload: dict, *, call_name: str = "planner.call") -> dict:
        last_error: Exception | None = None
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        for attempt in range(self._max_retries + 1):
            try:
                with tracing.generation(
                    call_name,
                    model=self.model,
                    input=payload,
                    metadata={"provider": "anthropic"},
                ) as generation:
                    data = self._post_json(
                        f"{self._base_url}/v1/messages",
                        headers,
                        payload,
                        self._timeout_s,
                    )
                    usage = _usage_from_response(data)
                    update = {"output": data}
                    if usage:
                        update["usage_details"] = usage
                    generation.update(**update)
                return data
            except GenLessonError as exc:
                if exc.code not in {"planner_rate_limited", "planner_failed"}:
                    raise
                last_error = exc
            except Exception as exc:
                last_error = exc
            if attempt >= self._max_retries:
                break
            delay = min(2.0 * (2**attempt), 30.0) + random.uniform(0.0, 0.25)
            log.warning(
                "Anthropic planner call failed (attempt %s/%s); retrying in %.1fs: %s",
                attempt + 1,
                self._max_retries + 1,
                delay,
                last_error,
            )
            self._sleep(delay)
        raise GenLessonError(
            "planner_failed",
            f"The Anthropic planner failed after {self._max_retries + 1} attempt(s).",
            details={"last_error": str(last_error)},
        ) from last_error


def _usage_from_response(data: dict) -> dict[str, int] | None:
    usage = data.get("usage") or {}
    if not isinstance(usage, dict):
        return None
    details: dict[str, int] = {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if input_tokens is not None:
        details["input"] = int(input_tokens)
    if output_tokens is not None:
        details["output"] = int(output_tokens)
    if details:
        details["total"] = sum(details.values())
    return details or None


def _default_post_json(
    url: str, headers: dict, payload: dict, timeout: float
) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise GenLessonError(
                "planner_auth_failed",
                "Anthropic rejected the API key for the planner.",
                hint="Check ANTHROPIC_API_KEY.",
            ) from exc
        if exc.code == 429:
            raise GenLessonError(
                "planner_rate_limited", "Anthropic rate-limited the planner request."
            ) from exc
        raise GenLessonError(
            "planner_failed",
            f"Anthropic request failed (HTTP {exc.code}).",
            details={"reason": str(exc)},
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise GenLessonError(
            "planner_failed", f"Anthropic request failed: {exc}"
        ) from exc


def _image_base64(path: Path) -> str:
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
            details={"bytes": len(data), "limit": _MAX_IMAGE_BYTES},
        )
    return base64.b64encode(data).decode("ascii")
