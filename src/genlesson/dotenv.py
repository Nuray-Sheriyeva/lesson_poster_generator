"""Minimal .env loader for provider credentials.

Only provider, Langfuse, and POSTER_* variables are loaded so a local .env cannot
silently override config.toml model choices through legacy GENLESSON_* settings.
"""

from __future__ import annotations

import os
from pathlib import Path

_ALLOWED_PREFIXES = ("OPENAI_", "ANTHROPIC_", "DEEPSEEK_", "LANGFUSE_", "POSTER_")


def load_dotenv(path: Path | None = None) -> None:
    if os.environ.get("POSTER_DISABLE_DOTENV", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    env_path = path or (Path.cwd() / ".env")
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not key.startswith(_ALLOWED_PREFIXES):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)
