"""Image provider boundary and typed failures for Stage 6."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class ImageProviderError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        moderation: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.moderation = moderation

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "moderation": self.moderation,
            "retryable": self.retryable,
        }


class ModerationRefusal(ImageProviderError):
    def __init__(self, message: str) -> None:
        super().__init__(
            "moderation_refused",
            message,
            retryable=False,
            moderation=True,
        )


@runtime_checkable
class ImageProvider(Protocol):
    name: str
    model: str

    def generate(self, prompt: str, size: str) -> bytes:
        """Return the generated image bytes for one format."""
