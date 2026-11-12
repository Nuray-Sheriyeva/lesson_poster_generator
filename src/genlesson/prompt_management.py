"""Langfuse prompt management with the in-code text as the offline fallback."""

from __future__ import annotations

import os

from . import tracing

PROMPT_LABEL_ENV = "GENLESSON_PROMPT_LABEL"
PROMPT_VERSION_ENV = "GENLESSON_PROMPT_VERSION"
DEFAULT_LABEL = "production"


def prompt_label() -> str:
    return os.environ.get(PROMPT_LABEL_ENV, "").strip() or DEFAULT_LABEL


def prompt_version() -> int | None:
    raw = os.environ.get(PROMPT_VERSION_ENV, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def get_prompt(
    name: str,
    fallback: str,
    *,
    label: str | None = None,
    version: int | None = None,
) -> str:
    if not tracing.is_enabled():
        return fallback
    if label is None and version is None:
        version = prompt_version()
        if version is None:
            label = prompt_label()
    return tracing.get_prompt_text(
        name,
        fallback,
        label=label,
        version=version,
    )


def clear_prompt_cache() -> None:
    """Compatibility hook; prompt fetching is cached by the Langfuse SDK."""


def sync_prompt(
    name: str,
    text: str,
    *,
    labels: tuple[str, ...] = (DEFAULT_LABEL,),
    commit_message: str | None = None,
) -> bool:
    return tracing.create_text_prompt(
        name,
        text,
        labels=labels,
        commit_message=commit_message,
    )


def sync_prompts(
    prompts: dict[str, str],
    *,
    labels: tuple[str, ...] = (DEFAULT_LABEL,),
) -> dict[str, bool]:
    return {
        name: sync_prompt(name, text, labels=labels)
        for name, text in prompts.items()
    }
