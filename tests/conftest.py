from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_live_langfuse(monkeypatch):
    """Never send test runs to a real Langfuse project from the developer .env."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    monkeypatch.setenv("LANGFUSE_HOST", "")
