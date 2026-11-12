from __future__ import annotations

import pytest

from genlesson import prompt_management, tracing
from genlesson import prompts as poster_prompts
from genlesson.planner import prompts as planner_prompts
from genlesson.planner.openai_planner import OpenAIPlanner

from .test_planner import _FakeClient as _PlannerClient
from .test_planner_providers import _request
from .test_tracing import _enable


@pytest.fixture(autouse=True)
def _clear_tracing():
    yield
    tracing.shutdown()


def test_prompt_falls_back_when_tracing_is_disabled() -> None:
    tracing.shutdown()

    assert prompt_management.get_prompt("missing", "fallback") == "fallback"


def test_prompt_uses_the_configured_label(monkeypatch) -> None:
    client = _enable(monkeypatch)
    monkeypatch.setenv(prompt_management.PROMPT_LABEL_ENV, "staging")
    client.prompt_texts[planner_prompts.SYSTEM_PROMPTS["classify"][0]] = "REMOTE"

    assert planner_prompts.system_prompt("classify") == "REMOTE"
    assert client.prompt_requests[-1]["label"] == "staging"
    assert client.prompt_requests[-1]["type"] == "text"


def test_prompt_fetch_errors_keep_the_fallback(monkeypatch) -> None:
    client = _enable(monkeypatch)
    client.prompt_error = RuntimeError("langfuse unavailable")

    assert planner_prompts.system_prompt("classify") == planner_prompts.CLASSIFY_SYSTEM
    assert isinstance(tracing.last_error(), RuntimeError)


def test_sync_prompts_creates_versions(monkeypatch) -> None:
    client = _enable(monkeypatch)

    statuses = planner_prompts.sync_prompts(labels=("staging",))

    assert all(statuses.values())
    assert client.created_prompts
    assert {item["labels"][0] for item in client.created_prompts} == {"staging"}
    assert {item["type"] for item in client.created_prompts} == {"text"}


def test_poster_layouts_use_prompt_management(monkeypatch) -> None:
    client = _enable(monkeypatch)
    client.prompt_texts[poster_prompts.LAYOUT_PROMPT_NAMES["portrait"]] = "REMOTE LAYOUT"

    assert poster_prompts.layout_for("portrait") == "REMOTE LAYOUT"


def test_openai_planner_messages_use_the_remote_prompt(monkeypatch, tmp_path) -> None:
    client = _enable(monkeypatch)
    client.prompt_texts[planner_prompts.SYSTEM_PROMPTS["classify"][0]] = "REMOTE CLASSIFY"
    planner = OpenAIPlanner(
        model="gpt-6-luna",
        api_key="",
        client=_PlannerClient([]),
        max_retries=0,
        json_retries=0,
    )

    messages = planner._classification_messages(_request(tmp_path))

    assert messages[0] == {"role": "system", "content": "REMOTE CLASSIFY"}
