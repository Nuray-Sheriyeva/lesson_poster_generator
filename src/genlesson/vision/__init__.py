"""Vision provider selection."""

from __future__ import annotations

from ..config import IngestOptions
from ..errors import GenLessonError
from .base import Transcriber
from .openai_vision import OpenAIVisionTranscriber

__all__ = ["Transcriber", "OpenAIVisionTranscriber", "build_transcriber"]


def build_transcriber(options: IngestOptions) -> Transcriber:
    if options.vision_mode == "none":
        raise GenLessonError(
            "vision_required",
            "This PDF has pages without a text layer, but vision transcription is disabled.",
            hint="Set OPENAI_API_KEY and omit --vision none, then resume.",
        )
    if not options.openai_api_key:
        raise GenLessonError(
            "vision_required",
            "This PDF has pages without a text layer and no OPENAI_API_KEY is set.",
            hint="Set OPENAI_API_KEY, or run with --vision none if the empty pages are acceptable.",
        )
    return OpenAIVisionTranscriber(
        model=options.vision_model,
        api_key=options.openai_api_key,
        base_url=options.openai_base_url,
        timeout_s=options.vision_timeout_s,
        max_retries=options.vision_max_retries,
    )
