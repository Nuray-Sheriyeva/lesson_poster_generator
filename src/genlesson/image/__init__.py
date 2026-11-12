"""Image provider selection."""

from __future__ import annotations

from ..config import GenerationOptions
from ..errors import GenLessonError
from .base import ImageProvider, ImageProviderError, ModerationRefusal
from .openai_image import OpenAIImageProvider

__all__ = [
    "ImageProvider",
    "ImageProviderError",
    "ModerationRefusal",
    "OpenAIImageProvider",
    "build_image_provider",
]


def build_image_provider(options: GenerationOptions, *, model: str) -> ImageProvider:
    if options.provider != "openai":
        raise GenLessonError(
            "image_provider_unavailable",
            f"Unsupported image provider: {options.provider}",
        )
    if not options.api_key:
        raise GenLessonError(
            "image_provider_required",
            "Image generation needs an OpenAI API key.",
            hint="Set OPENAI_API_KEY and run generate again.",
        )
    return OpenAIImageProvider(
        model=model,
        api_key=options.api_key,
        base_url=options.base_url,
        timeout_s=options.timeout_s,
        max_retries=options.max_retries,
    )
