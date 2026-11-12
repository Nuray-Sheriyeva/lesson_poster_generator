from __future__ import annotations

import json
from pathlib import Path

from genlesson.config import PlannerOptions
from genlesson.planner import (
    ClassificationRequest,
    PageImage,
    SegmentInput,
    build_planner,
)
from genlesson.planner.anthropic_planner import AnthropicPlanner
from genlesson.planner.openai_planner import DeepSeekPlanner, OpenAIPlanner

from .test_planner import _FakeClient, _classification_json, _png, _response


def _request(tmp_path: Path) -> ClassificationRequest:
    return ClassificationRequest(
        title="Photosynthesis",
        segments=(
            SegmentInput(
                id="s0001",
                page=1,
                page_label="1",
                order=1,
                text="Photosynthesis turns light into sugar.",
                text_source="text_layer",
            ),
        ),
        images=(PageImage(page=1, path=_png(tmp_path)),),
    )


def test_openai_capabilities() -> None:
    planner = OpenAIPlanner(
        model="gpt-6-luna", api_key="x", max_context=12345, max_retries=0
    )

    assert planner.capabilities.structured_output is True
    assert planner.capabilities.vision is True
    assert planner.capabilities.max_context == 12345


def test_deepseek_is_text_only_and_uses_json_mode(tmp_path: Path) -> None:
    client = _FakeClient([_response(_classification_json())])
    planner = DeepSeekPlanner(
        model="deepseek-chat",
        api_key="",
        client=client,
        max_retries=0,
        json_retries=0,
    )

    planner.classify(_request(tmp_path))

    assert planner.capabilities.structured_output is False
    assert planner.capabilities.vision is False
    assert client.calls[0]["response_format"] == {"type": "json_object"}
    user_content = client.calls[0]["messages"][1]["content"]
    assert all(block.get("type") != "image_url" for block in user_content)


def test_anthropic_uses_tool_schema_and_vision(tmp_path: Path) -> None:
    calls: list[dict] = []

    def post_json(url, headers, payload, timeout):
        calls.append(payload)
        tool_name = payload["tools"][0]["name"]
        return {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "name": tool_name,
                    "input": json.loads(_classification_json()),
                }
            ],
        }

    planner = AnthropicPlanner(
        model="claude-test",
        api_key="x",
        max_context=200000,
        post_json=post_json,
        max_retries=0,
    )

    result = planner.classify(_request(tmp_path))

    assert result.segments[0].id == "s0001"
    assert planner.capabilities.structured_output is True
    assert planner.capabilities.vision is True
    assert planner.capabilities.max_context == 200000
    content = calls[0]["messages"][0]["content"]
    assert any(block.get("type") == "image" for block in content)


def test_build_planner_dispatches_all_adapters() -> None:
    openai = build_planner(
        PlannerOptions(provider="openai", model="gpt-6-luna", api_key="x")
    )
    anthropic = build_planner(
        PlannerOptions(provider="anthropic", model="claude-test", api_key="x")
    )
    deepseek = build_planner(
        PlannerOptions(provider="deepseek", model="deepseek-chat", api_key="x")
    )

    assert isinstance(openai, OpenAIPlanner)
    assert isinstance(anthropic, AnthropicPlanner)
    assert isinstance(deepseek, DeepSeekPlanner)
    assert deepseek.capabilities.vision is False
    anthropic.close()


def test_provider_specific_env_key_and_model(monkeypatch) -> None:
    monkeypatch.setenv("POSTER_PLANNER_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("POSTER_ANTHROPIC_MODEL", "claude-test")

    options = PlannerOptions.from_env()

    assert options.provider == "anthropic"
    assert options.api_key == "anthropic-key"
    assert options.model == "claude-test"
