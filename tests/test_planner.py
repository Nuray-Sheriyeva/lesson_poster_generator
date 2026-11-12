from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from genlesson.config import PlannerOptions
from genlesson.errors import GenLessonError
from genlesson.planner import (
    CandidateRequest,
    ClassificationRequest,
    LanguageRequest,
    PageImage,
    PlanRequest,
    SegmentInput,
    build_planner,
)
from genlesson.planner.openai_planner import OpenAIPlanner


def _png(tmp_path: Path) -> Path:
    import pymupdf

    path = tmp_path / "page.png"
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 2, 2), 0)
    pixmap.clear_with(255)
    pixmap.save(path)
    return path


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, refusal=None))]
    )


class _FakeClient:
    def __init__(self, responses: list[SimpleNamespace]):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _classification_json() -> str:
    return json.dumps(
        {
            "segments": [
                {
                    "id": "s0001",
                    "label": "lesson_content",
                    "category": "lesson_content",
                    "reason": "body",
                }
            ],
            "topics": [{"title": "Photosynthesis", "segment_ids": ["s0001"]}],
        }
    )


def test_openai_planner_classifies_with_page_image(tmp_path: Path) -> None:
    client = _FakeClient([_response(_classification_json())])
    planner = OpenAIPlanner(
        model="gpt-6-luna",
        api_key="",
        client=client,
        max_retries=0,
        json_retries=0,
    )
    request = ClassificationRequest(
        title="Photosynthesis",
        segments=(
            SegmentInput(
                id="s0001",
                page=1,
                page_label="Page 1",
                order=1,
                text="Plants make sugar from light.",
                text_source="text_layer",
            ),
        ),
        images=(PageImage(page=1, path=_png(tmp_path)),),
    )

    draft = planner.classify(request)

    assert draft.segments[0].label == "lesson_content"
    assert draft.topics[0].title == "Photosynthesis"
    call = client.calls[0]
    assert call["model"] == "gpt-6-luna"
    assert call["response_format"] == {"type": "json_object"}
    content = call["messages"][1]["content"]
    assert "Plants make sugar" in content[0]["text"]
    assert content[-1]["type"] == "image_url"
    assert content[-1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_openai_planner_detects_language(tmp_path: Path) -> None:
    payload = json.dumps(
        {
            "poster_language": "kk",
            "confidence": 0.95,
            "evidence": "Фотосинтез үдерісі",
        }
    )
    client = _FakeClient([_response(payload)])
    planner = OpenAIPlanner(
        model="gpt-6-luna",
        api_key="",
        client=client,
        max_retries=0,
        json_retries=0,
    )

    decision = planner.detect_language(LanguageRequest(text="Фотосинтез үдерісі"))

    assert decision.poster_language == "kk"
    assert decision.confidence == pytest.approx(0.95)
    assert "Фотосинтез" in client.calls[0]["messages"][1]["content"]


def test_openai_planner_repairs_invalid_json(tmp_path: Path) -> None:
    client = _FakeClient([_response("not json"), _response(_classification_json())])
    planner = OpenAIPlanner(
        model="gpt-6-luna", api_key="", client=client, max_retries=0, json_retries=1
    )
    request = ClassificationRequest(
        title=None,
        segments=(
            SegmentInput(
                id="s0001",
                page=1,
                page_label="Page 1",
                order=1,
                text="Lesson body.",
                text_source="text_layer",
            ),
        ),
    )

    draft = planner.classify(request)

    assert draft.segments[0].id == "s0001"
    assert len(client.calls) == 2
    assert "invalid" in client.calls[1]["messages"][-1]["content"]


def test_planner_requires_key() -> None:
    with pytest.raises(GenLessonError) as excinfo:
        build_planner(PlannerOptions(api_key=None))
    assert excinfo.value.code == "planner_required"


def test_openai_planner_accepts_fenced_json(tmp_path: Path) -> None:
    fenced = "```json\n" + _classification_json() + "\n```"
    client = _FakeClient([_response(fenced)])
    planner = OpenAIPlanner(
        model="gpt-6-luna",
        api_key="",
        client=client,
        max_retries=0,
        json_retries=0,
    )
    request = ClassificationRequest(
        title=None,
        segments=(
            SegmentInput(
                id="s0001",
                page=1,
                page_label="Page 1",
                order=1,
                text="Lesson body.",
                text_source="text_layer",
            ),
        ),
    )

    draft = planner.classify(request)

    assert draft.segments[0].label == "lesson_content"


def _poster_json() -> str:
    return json.dumps(
        {
            "language": "kk",
            "age_band": "middle",
            "central_idea": {
                "title": "Байланыс түрлері",
                "definition": "Байланыс — ақпарат алмасудың түрлері.",
            },
            "details": [
                {
                    "label": "Ауызша байланыс",
                    "explanation": "Сөйлеу арқылы ақпарат беру.",
                    "example": None,
                    "visual": "Two people talking.",
                    "source_page": 1,
                }
            ],
            "composition": {
                "concept": "hero scene with callouts",
                "description": "A hero scene with one callout per type.",
            },
            "style": {
                "palette": ["deep teal", "warm ochre", "cream"],
                "mood": "bright",
                "illustration_style": "friendly flat vector",
            },
            "strings_to_render": ["Байланыс түрлері", "Ауызша байланыс"],
            "source_conflicts": [],
            "file_slug": "communication-types",
        }
    )


def test_openai_planner_uses_strict_poster_schema(tmp_path: Path) -> None:
    client = _FakeClient([_response(_poster_json())])
    planner = OpenAIPlanner(
        model="gpt-6-luna",
        api_key="",
        client=client,
        max_retries=0,
        json_retries=0,
    )
    request = PlanRequest(
        language="kk",
        title="Байланыс",
        dominant_topic="Байланыс түрлері",
        max_details=3,
        text="Байланыс түрлері туралы мәтін.",
        images=(PageImage(page=1, path=_png(tmp_path)),),
    )

    draft = planner.plan(request)

    assert draft.language == "kk"
    assert draft.details[0].label == "Ауызша байланыс"
    call = client.calls[0]
    response_format = call["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "poster_spec"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert "$defs" not in json.dumps(schema)
    content = call["messages"][1]["content"]
    assert "Байланыс түрлері туралы мәтін" in content[0]["text"]
    assert content[-1]["type"] == "image_url"


def test_openai_planner_extracts_candidates_with_strict_schema() -> None:
    payload = json.dumps(
        {
            "central_ideas": [
                {"title": "Байланыс", "definition": "Ақпарат алмасу."}
            ],
            "details": [
                {
                    "label": "Жазбаша байланыс",
                    "explanation": "Мәтін арқылы ақпарат беру.",
                    "example": None,
                    "visual": "A letter and a pen.",
                    "source_page": 2,
                }
            ],
            "key_terms": ["байланыс"],
            "source_conflicts": [],
        }
    )
    client = _FakeClient([_response(payload)])
    planner = OpenAIPlanner(
        model="gpt-6-luna",
        api_key="",
        client=client,
        max_retries=0,
        json_retries=0,
    )
    request = CandidateRequest(
        language="kk",
        title="Байланыс",
        dominant_topic=None,
        text="Жазбаша байланыс туралы мәтін.",
        chunk_index=1,
        chunk_count=2,
        source_pages=(2,),
        max_candidates=10,
    )

    candidates = planner.extract_candidates(request)

    assert candidates.details[0].label == "Жазбаша байланыс"
    response_format = client.calls[0]["response_format"]
    assert response_format["json_schema"]["name"] == "candidate_ideas"
    assert response_format["json_schema"]["strict"] is True


def test_strict_schema_removes_unsupported_keywords() -> None:
    from genlesson.planner.strict_schema import strict_json_schema
    from genlesson.planner import PosterSpecDraft

    schema = strict_json_schema(PosterSpecDraft)

    assert "$defs" not in schema
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)
