"""Mascot assets, deterministic layout, text-aware placement, and compositing."""

from .assets import MascotAsset, load_mascot_assets
from .compose import (
    Composition,
    compose_overlay,
    compose_png,
    compose_png_result,
    render_preview,
)
from .config import MascotConfig
from .layout import MascotLayout, Rect, layout_mascot

__all__ = [
    "Composition",
    "MascotAsset",
    "MascotConfig",
    "MascotLayout",
    "Rect",
    "compose_overlay",
    "compose_png",
    "compose_png_result",
    "layout_mascot",
    "load_mascot_assets",
    "render_preview",
]
