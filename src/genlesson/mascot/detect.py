"""Compatibility import for text detection backends."""

from .zones import (
    NullTextDetector,
    RapidOCRDetector,
    TextDetector,
    get_text_detector,
    reset_text_detector,
)

__all__ = [
    "NullTextDetector",
    "RapidOCRDetector",
    "TextDetector",
    "get_text_detector",
    "reset_text_detector",
]
