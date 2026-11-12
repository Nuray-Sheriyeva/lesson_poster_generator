"""Optional Langfuse tracing.

This module is the only place that imports :mod:`langfuse`.  It is deliberately
small and defensive: tracing is an optional extra, so a missing SDK, missing
credentials, an unreachable Langfuse instance, or an invalid payload must never
change or interrupt the pipeline.

When tracing is disabled every public helper is a no-op.  In particular,
``call_kwargs()`` returns ``{}`` so provider calls are byte-for-byte identical
to a run without tracing.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import os
import re
import sys
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# Langfuse limits propagated metadata values to 200 characters.  Inputs are
# clipped to the configured budget before they leave the process.
_DEFAULT_MAX_INPUT_CHARS = 8000
_MAX_METADATA_VALUE = 200

_ABS_PATH_RE = re.compile(
    r"(?<![\w:/])(?:/(?:Users|home|private|tmp|var|opt|mnt|srv|root|workspace|app|data|etc)/"
    r"[^\s\"']*|[A-Za-z]:\\[^\s\"']*)"
)
_DATA_URL_RE = re.compile(r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=_-]+")
_BASE64_RE = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{512,}={0,2}(?![A-Za-z0-9+/])")

_current_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "genlesson_trace_id", default=None
)
_current_observation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "genlesson_observation_id", default=None
)
_current_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "genlesson_session_id", default=None
)


@dataclass(frozen=True)
class _FallbackConfig:
    enabled: str = "auto"
    environment: str = "local"
    max_input_chars: int = _DEFAULT_MAX_INPUT_CHARS
    mask_paths: bool = True


@dataclass
class _State:
    client: Any = None
    enabled: bool = False
    trace_id: str | None = None
    session_id: str | None = None
    tags: tuple[str, ...] = ()
    experiment: str | None = None
    environment: str = "local"
    max_input_chars: int = _DEFAULT_MAX_INPUT_CHARS
    mask_paths: bool = True
    openai_wrapped: bool = False
    error: BaseException | None = None


_state = _State()


def trace_id_for_run(run_id: str) -> str:
    """Return the stable Langfuse trace id for a run.

    A UUIDv5 keeps the id deterministic across processes and restarts while
    staying within the 32-hex-character shape used by OpenTelemetry.
    """
    return uuid.uuid5(uuid.NAMESPACE_URL, f"genlesson-run:{run_id}").hex


def session_id_for_run(run_id: str) -> str:
    """Return the Langfuse session id for a run (the run id itself)."""
    return run_id


def is_enabled() -> bool:
    return bool(_state.enabled and _state.client is not None)


def enabled() -> bool:
    """Alias for :func:`is_enabled`."""
    return is_enabled()


def uses_openai_wrapper() -> bool:
    """Whether the Langfuse OpenAI drop-in is available and active."""
    return bool(is_enabled() and _state.openai_wrapped)


def current_observation_id() -> str | None:
    """Return the active observation id, if a stage/generation span is open."""
    return _current_observation_id.get()


def current_trace_id() -> str | None:
    """Return the active trace id, if a stage/generation span is open."""
    return _current_trace_id.get()


def last_error() -> BaseException | None:
    """Return the last error swallowed by the tracing layer (for diagnostics)."""
    return _state.error


def _record_error(exc: BaseException) -> None:
    _state.error = exc


def _load_tracing_config() -> Any:
    try:
        from .app_config import load_app_config

        return load_app_config().tracing
    except Exception as exc:  # tracing must not make config errors fatal
        _record_error(exc)
        return _FallbackConfig()


def _clean_tags(tags: Sequence[str] | None) -> tuple[str, ...]:
    cleaned: list[str] = []
    for raw in tags or ():
        tag = _mask_string(str(raw).strip())[:200]
        if tag and tag not in cleaned:
            cleaned.append(tag)
    return tuple(cleaned)


def _can_wrap_openai() -> bool:
    try:
        from langfuse.openai import OpenAI  # noqa: F401
    except Exception as exc:
        _record_error(exc)
        return False
    return True


def init(
    *,
    enabled: bool | None = None,
    tags: Sequence[str] | None = None,
    experiment: str | None = None,
    run_id: str | None = None,
    config: Any = None,
) -> None:
    """Initialise the optional Langfuse client.

    ``enabled=None`` means "auto": trace when both credentials are present and
    the ``[tracing].enabled`` setting allows it.  ``enabled=True`` forces
    tracing on but still requires credentials.  ``enabled=False`` always wins.

    This function never raises.  If the SDK is missing, credentials are absent,
    or client construction fails, tracing stays disabled.
    """
    global _state

    if _state.client is not None:
        shutdown()

    _state = _State()
    public_key = (os.environ.get("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret_key = (os.environ.get("LANGFUSE_SECRET_KEY") or "").strip()

    if enabled is False or not (public_key and secret_key):
        return

    tracing_config = config if config is not None else _load_tracing_config()
    mode = getattr(tracing_config, "enabled", "auto")
    if enabled is None and mode == "off":
        return

    try:
        _state.environment = str(getattr(tracing_config, "environment", "local"))
        _state.max_input_chars = int(
            getattr(tracing_config, "max_input_chars", _DEFAULT_MAX_INPUT_CHARS)
        )
        _state.mask_paths = bool(getattr(tracing_config, "mask_paths", True))
    except Exception as exc:  # tracing must not fail on a malformed config object
        _record_error(exc)
        _state = _State(error=exc)
        return
    _state.tags = _clean_tags(tags)
    if experiment is not None:
        experiment_name = str(experiment).strip()
        _state.experiment = experiment_name or None
    if run_id is not None:
        _state.trace_id = trace_id_for_run(run_id)
        _state.session_id = session_id_for_run(run_id)

    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=(os.environ.get("LANGFUSE_HOST") or "").strip() or None,
            environment=_state.environment,
            tracing_enabled=True,
        )
    except Exception as exc:
        _record_error(exc)
        _state = _State(error=exc)
        return

    _state.client = client
    _state.enabled = True
    _state.openai_wrapped = _can_wrap_openai()


def shutdown() -> None:
    """Flush and stop the Langfuse client.  Never raises."""
    global _state

    client = _state.client
    _state = _State()
    if client is None:
        return
    try:
        client.shutdown()
    except Exception as exc:
        _record_error(exc)
        try:
            client.flush()
        except Exception as flush_exc:  # pragma: no cover - best effort
            _record_error(flush_exc)


# ---------------------------------------------------------------------------
# Payload preparation


def _clip(value: Any, limit: int) -> Any:
    """Clip strings while sharing one character budget across the payload."""
    budget = [max(limit, 0)]
    return _clip_value(value, budget)


def _clip_value(value: Any, budget: list[int]) -> Any:
    if isinstance(value, str):
        remaining = budget[0]
        if remaining <= 0:
            return ""
        if len(value) <= remaining:
            budget[0] -= len(value)
            return value
        budget[0] = 0
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return f"{value[:remaining]}…[truncated sha256={digest} length={len(value)}]"
    if isinstance(value, Mapping):
        return {str(key): _clip_value(item, budget) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clip_value(item, budget) for item in value]
    return value


def _mask_string(value: str) -> str:
    def _blob_placeholder(match: re.Match[str]) -> str:
        raw = match.group(0)
        return f"<data-image sha256={hashlib.sha256(raw.encode()).hexdigest()} length={len(raw)}>"

    def _base64_placeholder(match: re.Match[str]) -> str:
        raw = match.group(0)
        return f"<base64 sha256={hashlib.sha256(raw.encode()).hexdigest()} length={len(raw)}>"

    value = _DATA_URL_RE.sub(_blob_placeholder, value)
    value = _BASE64_RE.sub(_base64_placeholder, value)
    if _state.mask_paths:
        value = _ABS_PATH_RE.sub("<path>", value)
    return value


def _mask(value: Any) -> Any:
    if isinstance(value, str):
        return _mask_string(value)
    if isinstance(value, Mapping):
        return {str(key): _mask(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_mask(item) for item in value]
    return value


def _prepare(value: Any, *, limit: int | None = None) -> Any:
    if value is None:
        return None
    return _clip(_mask(value), _state.max_input_chars if limit is None else limit)


def _prepare_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if metadata is None:
        return None
    prepared: dict[str, Any] = {}
    for key, value in metadata.items():
        masked = _mask(value)
        if isinstance(masked, str) and len(masked) > _MAX_METADATA_VALUE:
            digest = hashlib.sha256(masked.encode("utf-8")).hexdigest()
            masked = f"…[truncated sha256={digest} length={len(masked)}]"
        prepared[str(key)] = masked
    return prepared


def _base_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    if _state.experiment:
        merged["experiment"] = _state.experiment
    if metadata:
        merged.update(metadata)
    return _prepare_metadata(merged) or None


# ---------------------------------------------------------------------------
# Observation handles


class Span:
    """A safe wrapper around a Langfuse observation (or a no-op)."""

    __slots__ = ("_observation", "session_id", "trace_id")

    def __init__(
        self,
        observation: Any = None,
        *,
        trace_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self._observation = observation
        self.trace_id = trace_id
        self.session_id = session_id

    @property
    def observation_id(self) -> str | None:
        return getattr(self._observation, "id", None) if self._observation else None

    @property
    def span_id(self) -> str | None:
        return self.observation_id

    def update(self, **kwargs: Any) -> Span:
        if self._observation is None:
            return self
        if "input" in kwargs:
            kwargs["input"] = _prepare(kwargs["input"])
        if "output" in kwargs:
            kwargs["output"] = _prepare(kwargs["output"])
        if "metadata" in kwargs:
            kwargs["metadata"] = _prepare_metadata(kwargs["metadata"])
        try:
            self._observation.update(**kwargs)
        except Exception as exc:
            _record_error(exc)
        return self

    def score(
        self,
        name: str,
        value: float | str,
        *,
        comment: str | None = None,
        data_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        score(
            name,
            value,
            trace_id=self.trace_id,
            observation_id=self.observation_id,
            comment=comment,
            data_type=data_type,
            metadata=metadata,
        )

    def annotate(
        self,
        name: str,
        *,
        input: Any = None,
        output: Any = None,
        metadata: Mapping[str, Any] | None = None,
        level: str | None = None,
    ) -> None:
        annotate(
            name,
            trace_id=self.trace_id,
            input=input,
            output=output,
            metadata=metadata,
            level=level,
        )

    def end(self) -> None:
        if self._observation is None:
            return
        try:
            self._observation.end()
        except Exception as exc:
            _record_error(exc)


@contextlib.contextmanager
def _attributes(
    *,
    session_id: str | None,
    tags: Sequence[str] | None,
    metadata: Mapping[str, Any] | None,
    trace_name: str | None,
) -> Iterator[None]:
    prepared_metadata = _base_metadata(metadata)
    prepared_tags = _clean_tags(tags)
    if session_id is None and not prepared_tags and not prepared_metadata:
        yield
        return
    try:
        from langfuse import propagate_attributes

        context = propagate_attributes(
            session_id=session_id,
            tags=list(prepared_tags) or None,
            metadata=prepared_metadata,
            trace_name=trace_name,
            environment=_state.environment,
        )
        context.__enter__()
    except Exception as exc:
        _record_error(exc)
        yield
        return
    try:
        yield
    finally:
        try:
            context.__exit__(None, None, None)
        except Exception as exc:  # pragma: no cover - Langfuse failure
            _record_error(exc)


@contextlib.contextmanager
def _observation(
    as_type: str,
    name: str,
    *,
    trace_id: str | None,
    session_id: str | None,
    input: Any = None,
    output: Any = None,
    metadata: Mapping[str, Any] | None = None,
    model: str | None = None,
    usage: Mapping[str, int] | None = None,
    level: str | None = None,
    status_message: str | None = None,
    tags: Sequence[str] | None = None,
) -> Iterator[Span]:
    resolved_trace = trace_id or _current_trace_id.get() or _state.trace_id
    resolved_session = (
        session_id or _current_session_id.get() or _state.session_id
    )
    parent_observation = _current_observation_id.get()
    if not is_enabled():
        yield Span(trace_id=resolved_trace, session_id=resolved_session)
        return

    kwargs: dict[str, Any] = {
        "name": name,
        "as_type": as_type,
        "input": _prepare(input),
        "output": _prepare(output),
        "metadata": _base_metadata(metadata),
    }
    if model is not None:
        kwargs["model"] = model
    if usage is not None:
        kwargs["usage_details"] = dict(usage)
    if level is not None:
        kwargs["level"] = level
    if status_message is not None:
        kwargs["status_message"] = status_message
    if resolved_trace and (
        parent_observation is None or resolved_trace != _current_trace_id.get()
    ):
        kwargs["trace_context"] = {"trace_id": resolved_trace}

    try:
        context = _state.client.start_as_current_observation(**kwargs)
        observation = context.__enter__()
    except Exception as exc:
        _record_error(exc)
        yield Span(trace_id=resolved_trace, session_id=resolved_session)
        return

    handle = Span(observation, trace_id=resolved_trace, session_id=resolved_session)
    trace_token = _current_trace_id.set(resolved_trace)
    session_token = _current_session_id.set(resolved_session)
    observation_token = _current_observation_id.set(handle.observation_id)
    exc_info: tuple[Any, Any, Any] = (None, None, None)
    try:
        with _attributes(
            session_id=resolved_session,
            tags=tags,
            metadata=metadata,
            trace_name=name if parent_observation is None else None,
        ):
            yield handle
    except BaseException:
        exc_info = sys.exc_info()
        raise
    finally:
        _current_observation_id.reset(observation_token)
        _current_session_id.reset(session_token)
        _current_trace_id.reset(trace_token)
        try:
            context.__exit__(*exc_info)
        except Exception as exc:  # pragma: no cover - Langfuse failure
            _record_error(exc)


@contextlib.contextmanager
def stage_span(
    name: str,
    *,
    run_id: str | None = None,
    trace_id: str | None = None,
    input: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[Span]:
    """Trace one pipeline stage.  Yields a no-op :class:`Span` when disabled."""
    resolved_trace = trace_id or (trace_id_for_run(run_id) if run_id else None)
    with _observation(
        "span",
        name,
        trace_id=resolved_trace,
        session_id=session_id_for_run(run_id) if run_id else None,
        input=input,
        metadata=metadata,
        tags=_state.tags,
    ) as span:
        yield span


@contextlib.contextmanager
def generation(
    name: str,
    *,
    model: str | None = None,
    input: Any = None,
    output: Any = None,
    metadata: Mapping[str, Any] | None = None,
    usage: Mapping[str, int] | None = None,
    run_id: str | None = None,
    trace_id: str | None = None,
    tags: Sequence[str] | None = None,
) -> Iterator[Span]:
    """Trace one model call.  Yields a no-op :class:`Span` when disabled."""
    resolved_trace = trace_id or (trace_id_for_run(run_id) if run_id else None)
    with _observation(
        "generation",
        name,
        trace_id=resolved_trace,
        session_id=session_id_for_run(run_id) if run_id else None,
        input=input,
        output=output,
        metadata=metadata,
        model=model,
        usage=usage,
        tags=tags or _state.tags,
    ) as span:
        yield span


# ---------------------------------------------------------------------------
# Scores, annotations, provider helpers


def score(
    name: str,
    value: float | str,
    *,
    trace_id: str | None = None,
    observation_id: str | None = None,
    comment: str | None = None,
    data_type: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Attach a Langfuse score.  No-op when tracing is disabled."""
    if not is_enabled():
        return
    resolved_trace = trace_id or _current_trace_id.get() or _state.trace_id
    resolved_observation = observation_id or _current_observation_id.get()
    kwargs: dict[str, Any] = {"name": name, "value": value}
    if resolved_trace:
        kwargs["trace_id"] = resolved_trace
    if resolved_observation:
        kwargs["observation_id"] = resolved_observation
    if comment:
        kwargs["comment"] = comment
    if data_type:
        kwargs["data_type"] = data_type
    prepared_metadata = _prepare_metadata(metadata)
    if prepared_metadata:
        kwargs["metadata"] = prepared_metadata
    try:
        _state.client.create_score(**kwargs)
    except Exception as exc:
        _record_error(exc)


def annotate(
    name: str,
    *,
    input: Any = None,
    output: Any = None,
    metadata: Mapping[str, Any] | None = None,
    level: str | None = None,
    trace_id: str | None = None,
) -> None:
    """Record a point-in-time event on the current trace.  Never raises."""
    if not is_enabled():
        return
    resolved_trace = trace_id or _current_trace_id.get() or _state.trace_id
    kwargs: dict[str, Any] = {"name": name}
    if resolved_trace and (
        _current_observation_id.get() is None or resolved_trace != _current_trace_id.get()
    ):
        kwargs["trace_context"] = {"trace_id": resolved_trace}
    if input is not None:
        kwargs["input"] = _prepare(input)
    if output is not None:
        kwargs["output"] = _prepare(output)
    prepared_metadata = _base_metadata(metadata)
    if prepared_metadata:
        kwargs["metadata"] = prepared_metadata
    if level:
        kwargs["level"] = level
    try:
        _state.client.create_event(**kwargs)
    except Exception as exc:
        _record_error(exc)


def call_kwargs(
    *,
    name: str | None = None,
    run_id: str | None = None,
    trace_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Return extra kwargs for the Langfuse/OpenAI drop-in client.

    These are the ``name``/``trace_id``/``parent_observation_id``/``metadata``
    kwargs understood by ``langfuse.openai.OpenAI``.  Session id and tags are
    deliberately not included: they belong to the stage span's propagated
    attributes, and Langfuse 5.2's OpenAI wrapper would forward unknown
    ``session_id``/``tags`` kwargs to OpenAI itself.
    """
    if not is_enabled() or not _state.openai_wrapped:
        return {}
    resolved_trace = (
        trace_id
        or _current_trace_id.get()
        or (trace_id_for_run(run_id) if run_id else None)
        or _state.trace_id
    )
    if not resolved_trace:
        return {"name": name} if name else {}
    kwargs: dict[str, Any] = {"trace_id": resolved_trace}
    if name:
        kwargs["name"] = name
    parent = _current_observation_id.get()
    if parent:
        kwargs["parent_observation_id"] = parent
    if _state.experiment:
        kwargs["metadata"] = {"experiment": _mask_string(_state.experiment)[:200]}
    return kwargs


def get_client() -> Any:
    """Return the active Langfuse client, if any."""
    return _state.client if is_enabled() else None


def get_prompt_text(
    name: str,
    fallback: str,
    *,
    label: str | None = None,
    version: int | None = None,
) -> str:
    """Fetch a text prompt, falling back to the in-code template."""
    if not is_enabled():
        return fallback
    try:
        prompt = _state.client.get_prompt(
            name,
            type="text",
            label=label,
            version=version,
            fallback=fallback,
        )
        return str(prompt.prompt)
    except Exception as exc:
        _record_error(exc)
        return fallback


def create_text_prompt(
    name: str,
    text: str,
    *,
    labels: Sequence[str] = ("production",),
    commit_message: str | None = None,
) -> bool:
    """Create a Langfuse text prompt version.  Never raises."""
    if not is_enabled():
        return False
    try:
        _state.client.create_prompt(
            name=name,
            prompt=text,
            labels=list(labels),
            type="text",
            commit_message=commit_message,
        )
        return True
    except Exception as exc:
        _record_error(exc)
        return False


def openai_client_factory() -> Any:
    """Return the OpenAI class to use.

    Tracing enabled: Langfuse's drop-in client, which records each model call
    automatically.  Otherwise: the regular ``openai.OpenAI`` class.
    """
    if is_enabled():
        try:
            from langfuse.openai import OpenAI

            _state.openai_wrapped = True
            return OpenAI
        except Exception as exc:
            _record_error(exc)
            _state.openai_wrapped = False
    from openai import OpenAI

    return OpenAI
