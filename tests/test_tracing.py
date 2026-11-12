from __future__ import annotations

import contextlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from genlesson import tracing
from genlesson.app_config import TracingConfig
from genlesson.cli import build_parser, main
from genlesson.planner import CandidateRequest, PlanRequest
from genlesson.planner.anthropic_planner import AnthropicPlanner
from genlesson.planner.openai_planner import OpenAIPlanner
from genlesson.runs import RunStore
from genlesson.vision.openai_vision import OpenAIVisionTranscriber

from .helpers import (
    FakePlanner,
    FakePosterPlanner,
    FakeVisionTranscriber,
    make_scanned_pdf,
    make_text_pdf,
)
from .test_planner import (
    _classification_json,
    _png,
    _poster_json,
    _response,
)
from .test_planner import (
    _FakeClient as _PlannerClient,
)
from .test_planner_providers import _request
from .test_vision import _FakeClient as _VisionClient


@pytest.fixture(autouse=True)
def _reset_tracing(monkeypatch):
    for key in (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
    ):
        monkeypatch.delenv(key, raising=False)
    tracing.shutdown()
    yield
    tracing.shutdown()


class _FakeObservation:
    def __init__(self, name: str, observation_id: str) -> None:
        self.name = name
        self.id = observation_id
        self.updates: list[dict] = []
        self.ended = False

    def update(self, **kwargs) -> None:
        self.updates.append(kwargs)

    def end(self) -> None:
        self.ended = True


class _FakeContext:
    def __init__(self, client: _FakeClient, observation: _FakeObservation) -> None:
        self.client = client
        self.observation = observation

    def __enter__(self) -> _FakeObservation:
        self.client.stack.append(self.observation.id)
        return self.observation

    def __exit__(self, *exc_info) -> bool:
        if self.client.stack and self.client.stack[-1] == self.observation.id:
            self.client.stack.pop()
        return False


class _FakeClient:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.created: list[dict] = []
        self.observations: list[_FakeObservation] = []
        self.stack: list[str] = []
        self.scores: list[dict] = []
        self.events: list[dict] = []
        self.propagations: list[dict] = []
        self.prompt_texts: dict[str, str] = {}
        self.prompt_requests: list[dict] = []
        self.created_prompts: list[dict] = []
        self.prompt_error: Exception | None = None
        self.shutdown_called = False
        self.flush_called = False

    def start_as_current_observation(self, **kwargs):
        parent = self.stack[-1] if self.stack else None
        self.created.append({**kwargs, "_parent": parent})
        observation = _FakeObservation(
            kwargs.get("name", ""), f"obs-{len(self.observations) + 1}"
        )
        self.observations.append(observation)
        return _FakeContext(self, observation)

    def create_score(self, **kwargs) -> None:
        self.scores.append(kwargs)

    def create_event(self, **kwargs) -> None:
        self.events.append(kwargs)

    def get_prompt(self, name, **kwargs):
        self.prompt_requests.append({"name": name, **kwargs})
        if self.prompt_error is not None:
            raise self.prompt_error
        return SimpleNamespace(
            prompt=self.prompt_texts.get(name, kwargs.get("fallback")),
            version=kwargs.get("version") or 1,
        )

    def create_prompt(self, **kwargs) -> None:
        self.created_prompts.append(kwargs)

    def shutdown(self) -> None:
        self.shutdown_called = True

    def flush(self) -> None:
        self.flush_called = True


def _install_fake_langfuse(monkeypatch, client: _FakeClient, *, openai: bool = True):
    module = types.ModuleType("langfuse")
    module.__path__ = []  # type: ignore[attr-defined]

    def _make_client(**kwargs):
        client.kwargs = kwargs
        return client

    def _propagate(**kwargs):
        client.propagations.append(kwargs)
        return contextlib.nullcontext()

    module.Langfuse = _make_client  # type: ignore[attr-defined]
    module.propagate_attributes = _propagate  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", module)
    if openai:
        openai_module = types.ModuleType("langfuse.openai")
        openai_module.OpenAI = object()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "langfuse.openai", openai_module)
    return client


def _enable(monkeypatch, *, max_input_chars: int = 8000) -> _FakeClient:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    client = _FakeClient()
    _install_fake_langfuse(monkeypatch, client)
    tracing.init(
        config=TracingConfig(
            enabled="on",
            environment="test",
            max_input_chars=max_input_chars,
            mask_paths=True,
        )
    )
    return client


def test_disabled_without_keys_is_a_noop() -> None:
    tracing.init()

    assert tracing.is_enabled() is False
    assert tracing.enabled() is False
    assert tracing.call_kwargs() == {}
    assert tracing.call_kwargs(name="classify.segments") == {}
    with tracing.stage_span("ingest", run_id="run-1") as span:
        assert span.trace_id == tracing.trace_id_for_run("run-1")
        assert span.observation_id is None
        span.update(input="ignored")
        span.score("quality", 1.0)
        tracing.annotate("note", metadata={"ignored": True})
    tracing.shutdown()


def test_trace_id_is_deterministic_and_otel_compatible() -> None:
    first = tracing.trace_id_for_run("lesson-run")
    second = tracing.trace_id_for_run("lesson-run")
    other = tracing.trace_id_for_run("other-run")

    assert first == second
    assert first != other
    assert len(first) == 32
    assert all(character in "0123456789abcdef" for character in first)
    assert tracing.session_id_for_run("lesson-run") == "lesson-run"


def test_enabled_span_records_attributes_and_scores(monkeypatch) -> None:
    client = _enable(monkeypatch)

    assert tracing.is_enabled() is True
    assert client.kwargs["public_key"] == "pk-test"
    assert client.kwargs["environment"] == "test"

    with tracing.stage_span("ingest", run_id="run-1") as span:
        span.update(output={"pages": 2})
        span.score("grounded", 0.9)
        with tracing.generation("plan", model="fake-model") as generation:
            generation.update(output="planned")
            assert generation.trace_id == tracing.trace_id_for_run("run-1")
        tracing.annotate("reviewed", metadata={"ok": True})

    assert len(client.observations) == 2
    assert client.observations[0].name == "ingest"
    assert client.observations[0].updates[-1]["output"] == {"pages": 2}
    assert client.observations[1].name == "plan"
    assert client.scores[0]["trace_id"] == tracing.trace_id_for_run("run-1")
    assert client.scores[0]["observation_id"] == "obs-1"
    assert "trace_context" not in client.events[0]
    assert client.shutdown_called is False
    tracing.shutdown()
    assert client.shutdown_called is True


def test_tags_and_experiment_propagate(monkeypatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    client = _FakeClient()
    _install_fake_langfuse(monkeypatch, client)
    tracing.init(
        tags=("cli", "e2e", "cli"),
        experiment="exp-1",
        config=TracingConfig(enabled="on", environment="staging"),
    )

    with tracing.stage_span("ingest", run_id="run-1"):
        pass

    propagation = client.propagations[0]
    assert propagation["session_id"] == "run-1"
    assert propagation["tags"] == ["cli", "e2e"]
    assert propagation["metadata"] == {"experiment": "exp-1"}
    assert propagation["trace_name"] == "ingest"
    assert propagation["environment"] == "staging"


def test_call_kwargs_are_empty_until_the_openai_wrapper_is_available(monkeypatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    client = _FakeClient()
    _install_fake_langfuse(monkeypatch, client, openai=False)
    tracing.init(config=TracingConfig(enabled="on"))

    assert tracing.is_enabled() is True
    assert tracing.call_kwargs() == {}


def test_call_kwargs_and_openai_factory_when_enabled(monkeypatch) -> None:
    _enable(monkeypatch)

    kwargs = tracing.call_kwargs(name="classify.segments", run_id="run-1")
    assert kwargs == {
        "trace_id": tracing.trace_id_for_run("run-1"),
        "name": "classify.segments",
    }
    assert tracing.uses_openai_wrapper() is True

    with tracing.stage_span("ingest", run_id="run-1") as span:
        assert span.observation_id == "obs-1"
        nested = tracing.call_kwargs()
        assert nested["parent_observation_id"] == "obs-1"

    assert tracing.openai_client_factory() is sys.modules["langfuse.openai"].OpenAI


def test_payloads_are_masked_and_truncated(monkeypatch) -> None:
    client = _enable(monkeypatch, max_input_chars=20)

    with tracing.stage_span(
        "ingest",
        run_id="run-1",
        input={"path": "/Users/example/private/lesson.pdf", "text": "x" * 100},
    ):
        pass

    payload = client.created[0]
    truncated = payload["input"]["text"]
    assert "…[truncated sha256=" in truncated
    assert "length=100]" in truncated
    assert payload["input"]["path"] == "<path>"


def test_image_data_urls_are_replaced_with_digest(monkeypatch) -> None:
    client = _enable(monkeypatch, max_input_chars=1000)
    data_url = "data:image/png;base64," + ("A" * 400)

    with tracing.stage_span(
        "ingest",
        run_id="run-1",
        input={"image": data_url, "raw": "QUJD" * 200},
    ):
        pass

    value = client.created[0]["input"]["image"]
    assert value.startswith("<data-image sha256=")
    assert "length=" in value
    assert "base64" not in value
    assert len(value) < len(data_url)
    raw = client.created[0]["input"]["raw"]
    assert raw.startswith("<base64 sha256=")
    assert "length=800" in raw


def test_sdk_construction_error_is_isolated(monkeypatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    module = types.ModuleType("langfuse")

    def _explode(**kwargs):
        raise RuntimeError("no langfuse today")

    module.Langfuse = _explode  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", module)

    tracing.init(config=TracingConfig(enabled="on"))

    assert tracing.is_enabled() is False
    assert isinstance(tracing.last_error(), RuntimeError)
    with tracing.stage_span("ingest", run_id="run-1"):
        pass
    tracing.shutdown()


def test_observation_errors_do_not_escape(monkeypatch) -> None:
    client = _enable(monkeypatch)

    def _explode(**kwargs):
        raise RuntimeError("span failed")

    client.start_as_current_observation = _explode  # type: ignore[method-assign]
    with tracing.stage_span("ingest", run_id="run-1") as span:
        span.update(output="still safe")

    assert tracing.is_enabled() is True
    assert isinstance(tracing.last_error(), RuntimeError)


def test_init_without_keys_never_constructs_a_client(monkeypatch) -> None:
    constructed: list[dict] = []
    module = types.ModuleType("langfuse")

    def _record(**kwargs):
        constructed.append(kwargs)
        return _FakeClient()

    module.Langfuse = _record  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", module)

    tracing.init(enabled=True)
    tracing.init(enabled=False)

    assert constructed == []
    assert tracing.is_enabled() is False


def test_cli_global_tracing_flags() -> None:
    parser = build_parser()

    enabled = parser.parse_args(
        [
            "--tracing",
            "--trace-tag",
            "one",
            "--trace-tag",
            "two",
            "--experiment",
            "exp",
            "status",
            "r",
        ]
    )
    assert enabled.tracing is True
    assert enabled.trace_tag == ["one", "two"]
    assert enabled.experiment == "exp"

    disabled = parser.parse_args(["--no-tracing", "status", "r"])
    assert disabled.tracing is False
    assert disabled.trace_tag is None
    assert disabled.experiment is None


def test_cli_ingest_persists_tracing_identity(tmp_path: Path, capsys) -> None:
    pdf = make_text_pdf(tmp_path / "lesson.pdf", ["A lesson about the water cycle."])
    runs = tmp_path / "runs"
    args = [
        "ingest",
        str(pdf),
        "--run-id",
        "trace-run",
        "--runs-dir",
        str(runs),
        "--vision",
        "none",
        "--dpi",
        "72",
    ]
    assert main(args) == 0
    capsys.readouterr()

    store = RunStore(runs)
    _, manifest = store.load("trace-run")
    assert manifest.config["tracing"] == {
        "trace_id": tracing.trace_id_for_run("trace-run"),
        "session_id": "trace-run",
    }
    ingest_config = manifest.config["ingest"]

    assert main(args + ["--resume"]) == 0
    capsys.readouterr()
    _, resumed = store.load("trace-run")
    assert resumed.config["tracing"] == manifest.config["tracing"]
    assert resumed.config["ingest"] == ingest_config


def test_cli_passes_flags_to_tracing_and_shuts_down(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import genlesson.cli as cli_module

    calls: dict[str, object] = {}

    def _init(**kwargs) -> None:
        calls.update(kwargs)

    def _shutdown() -> None:
        calls["shutdown"] = True

    monkeypatch.setattr(cli_module.tracing, "init", _init)
    monkeypatch.setattr(cli_module.tracing, "shutdown", _shutdown)

    code = main(
        [
            "--no-tracing",
            "--trace-tag",
            "cli",
            "--experiment",
            "exp-1",
            "status",
            "missing",
            "--runs-dir",
            str(tmp_path / "runs"),
        ]
    )
    capsys.readouterr()

    assert code == 2
    assert calls["enabled"] is False
    assert calls["tags"] == ("cli",)
    assert calls["experiment"] == "exp-1"
    assert calls["shutdown"] is True


def test_openai_planner_attaches_name_usage_and_parent(monkeypatch, tmp_path) -> None:
    backend = _enable(monkeypatch)
    response = _response(_classification_json())
    response.usage = SimpleNamespace(
        prompt_tokens=11, completion_tokens=7, total_tokens=18
    )
    client = _PlannerClient([response])
    planner = OpenAIPlanner(
        model="gpt-6-luna",
        api_key="",
        client=client,
        max_retries=0,
        json_retries=0,
    )

    with tracing.stage_span("classify", run_id="run-1") as parent:
        planner.classify(_request(tmp_path))

    call = client.calls[0]
    assert call["name"] == "classify.segments"
    assert call["trace_id"] == tracing.trace_id_for_run("run-1")
    assert "session_id" not in call
    assert call["parent_observation_id"] == parent.observation_id

    generation = next(
        item for item in backend.observations if item.name == "classify.segments"
    )
    assert generation.updates[-1]["usage_details"] == {
        "input": 11,
        "output": 7,
        "total": 18,
    }
    assert generation.updates[-1]["output"]


def test_openai_planner_candidate_call_uses_plan_candidates_name(
    monkeypatch, tmp_path
) -> None:
    _enable(monkeypatch)
    payload = json.dumps(
        {
            "central_ideas": [{"title": "Topic", "definition": "Definition."}],
            "details": [
                {
                    "label": "Label",
                    "explanation": "Explanation.",
                    "example": None,
                    "visual": "A visual.",
                    "source_page": 1,
                }
            ],
            "key_terms": ["term"],
            "source_conflicts": [],
        }
    )
    client = _PlannerClient([_response(payload)])
    planner = OpenAIPlanner(
        model="gpt-6-luna",
        api_key="",
        client=client,
        max_retries=0,
        json_retries=0,
    )
    request = CandidateRequest(
        language="kk",
        title="Lesson",
        dominant_topic=None,
        text="Chunk text.",
        chunk_index=1,
        chunk_count=2,
        source_pages=(1,),
        max_candidates=5,
    )

    with tracing.stage_span("plan", run_id="run-1"):
        planner.extract_candidates(request)

    assert client.calls[0]["name"] == "plan.candidates"


def test_openai_planner_plan_uses_plan_spec_name_and_usage(monkeypatch, tmp_path) -> None:
    backend = _enable(monkeypatch)
    response = _response(_poster_json())
    response.usage = SimpleNamespace(
        prompt_tokens=21, completion_tokens=9, total_tokens=30
    )
    client = _PlannerClient([response])
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
    )

    with tracing.stage_span("plan", run_id="run-1"):
        planner.plan(request)

    assert client.calls[0]["name"] == "plan.spec"
    generation = next(item for item in backend.observations if item.name == "plan.spec")
    assert generation.updates[-1]["usage_details"] == {
        "input": 21,
        "output": 9,
        "total": 30,
    }


def test_vision_transcriber_uses_stage_specific_call_name(monkeypatch, tmp_path) -> None:
    backend = _enable(monkeypatch)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Page text", refusal=None))],
        usage=SimpleNamespace(input_tokens=9, output_tokens=4, total_tokens=13),
    )
    client = _VisionClient(response)
    transcriber = OpenAIVisionTranscriber(
        model="gpt-6-luna",
        api_key="",
        client=client,
        max_retries=0,
    )

    with tracing.stage_span("ingest", run_id="run-1") as parent:
        assert transcriber.transcribe(_png(tmp_path), 2) == "Page text"

    call = client.calls[0]
    assert call["name"] == "ingest.transcribe_page"
    assert call["parent_observation_id"] == parent.observation_id
    generation = next(
        item for item in backend.observations if item.name == "ingest.transcribe_page"
    )
    assert generation.updates[-1]["usage_details"] == {
        "input": 9,
        "output": 4,
        "total": 13,
    }


def test_verify_transcriber_uses_its_own_call_name(monkeypatch, tmp_path) -> None:
    _enable(monkeypatch)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Page text", refusal=None))]
    )
    client = _VisionClient(response)
    transcriber = OpenAIVisionTranscriber(
        model="gpt-6-luna",
        api_key="",
        client=client,
        max_retries=0,
        call_name="verify.transcribe",
    )

    with tracing.stage_span("verify", run_id="run-1"):
        transcriber.transcribe(_png(tmp_path), 1)

    assert client.calls[0]["name"] == "verify.transcribe"


def test_anthropic_generation_records_token_usage(monkeypatch, tmp_path) -> None:
    backend = _enable(monkeypatch)

    def post_json(url, headers, payload, timeout):
        return {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "name": payload["tools"][0]["name"],
                    "input": json.loads(_classification_json()),
                }
            ],
            "usage": {"input_tokens": 8, "output_tokens": 3},
        }

    planner = AnthropicPlanner(
        model="claude-test",
        api_key="x",
        post_json=post_json,
        max_retries=0,
    )

    with tracing.stage_span("classify", run_id="run-1") as parent:
        planner.classify(_request(tmp_path))

    generation = next(
        item for item in backend.observations if item.name == "classify.segments"
    )
    assert backend.created[1]["_parent"] == parent.observation_id
    assert generation.updates[-1]["usage_details"] == {
        "input": 8,
        "output": 3,
        "total": 11,
    }


def test_dispatch_wraps_stages_and_annotates(monkeypatch, tmp_path, capsys) -> None:
    backend = _FakeClient()
    _install_fake_langfuse(monkeypatch, backend)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    pdf = make_text_pdf(tmp_path / "lesson.pdf", ["A lesson about the water cycle."])
    runs = tmp_path / "runs"
    assert (
        main(
            [
                "ingest",
                str(pdf),
                "--run-id",
                "trace-run",
                "--runs-dir",
                str(runs),
                "--vision",
                "none",
                "--dpi",
                "72",
            ]
        )
        == 0
    )
    capsys.readouterr()

    import genlesson.classify as classify_module

    monkeypatch.setattr(
        classify_module,
        "build_planner",
        lambda options: FakePlanner(model=options.model),
    )
    assert (
        main(["classify", "trace-run", "--runs-dir", str(runs)])
        == 0
    )
    capsys.readouterr()

    spans = [item for item in backend.created if item["name"] in {"ingest", "classify"}]
    assert [item["name"] for item in spans] == ["ingest", "classify"]
    trace_id = tracing.trace_id_for_run("trace-run")
    assert all(item["trace_context"]["trace_id"] == trace_id for item in spans)
    assert all(item["_parent"] is None for item in spans)
    events = {item["name"]: item for item in backend.events}
    assert events["classify"]["metadata"]["language"] == "kk"
    assert events["classify"]["metadata"]["segments"] >= 1
    assert events["ingest"]["metadata"]["pages"] >= 1
    scores = {item["name"]: item for item in backend.scores}
    assert scores["language_confidence"]["value"] == 0.92


def test_cli_plan_and_review_scores(monkeypatch, tmp_path, capsys) -> None:
    backend = _FakeClient()
    _install_fake_langfuse(monkeypatch, backend)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    pdf = make_text_pdf(tmp_path / "lesson.pdf", ["A lesson about the water cycle."])
    runs = tmp_path / "runs"
    assert (
        main(
            [
                "ingest",
                str(pdf),
                "--run-id",
                "score-run",
                "--runs-dir",
                str(runs),
                "--vision",
                "none",
                "--dpi",
                "72",
            ]
        )
        == 0
    )
    capsys.readouterr()

    import genlesson.classify as classify_module
    import genlesson.plan as plan_module

    monkeypatch.setattr(
        classify_module,
        "build_planner",
        lambda options: FakePlanner(model=options.model),
    )
    assert main(["classify", "score-run", "--runs-dir", str(runs)]) == 0
    capsys.readouterr()

    monkeypatch.setattr(
        plan_module,
        "build_planner",
        lambda options: FakePosterPlanner(detail_count=2),
    )
    assert main(["plan", "score-run", "--runs-dir", str(runs)]) == 0
    capsys.readouterr()

    assert (
        main(["review", "score-run", "--runs-dir", str(runs), "--review", "on"])
        == 0
    )
    capsys.readouterr()

    scores = {item["name"]: item for item in backend.scores}
    assert scores["details"]["value"] == 2.0
    assert scores["conflicts"]["value"] == 0.0
    assert scores["review_edited"]["value"] == 0.0
    assert scores["review_edited"]["data_type"] == "BOOLEAN"
    assert scores["human_wait_s"]["value"] == 0.0


def test_ingest_stage_span_with_fake_vision_transcriber(
    monkeypatch, tmp_path, capsys
) -> None:
    import genlesson.ingest.pdf as pdf_module

    backend = _FakeClient()
    _install_fake_langfuse(monkeypatch, backend)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    transcriber = FakeVisionTranscriber(["Scanned lesson body."])
    monkeypatch.setattr(pdf_module, "build_transcriber", lambda options: transcriber)

    pdf = make_scanned_pdf(tmp_path / "scan.pdf", ["Scanned lesson body."])
    assert (
        main(
            [
                "ingest",
                str(pdf),
                "--run-id",
                "scan-run",
                "--runs-dir",
                str(tmp_path / "runs"),
                "--vision",
                "openai",
                "--dpi",
                "72",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert transcriber.calls
    span = next(item for item in backend.created if item["name"] == "ingest")
    assert span["trace_context"]["trace_id"] == tracing.trace_id_for_run("scan-run")
    event = next(item for item in backend.events if item["name"] == "ingest")
    assert event["metadata"]["vision_pages"] >= 1
