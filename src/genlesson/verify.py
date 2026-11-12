"""Stage 7: verify rendered text per format and retry mismatches."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image

from . import tracing
from .app_config import load_app_config
from .config import VerificationOptions, default_planner_model, default_vision_model
from .errors import GenLessonError
from .generate import decode_image, save_image, soften_prompt
from .image import ImageProvider, ImageProviderError, ModerationRefusal, build_image_provider
from .mascot.assets import load_mascot_assets
from .mascot.select import Placement, PlacementFailure, resolve_placement
from .models import (
    ExtraString,
    FormatVerification,
    GeneratedFormat,
    GenerationDocument,
    GenerationErrorInfo,
    MascotLayoutDocument,
    PosterSpecDocument,
    PromptBundle,
    PromptFormatSpec,
    ReviewDocument,
    RunManifest,
    StringCheck,
    VerificationAttempt,
    VerificationConfig,
    VerificationDocument,
    VerificationStats,
    utc_now,
)
from .runs import RunPaths, RunStore
from .storage import read_model, sha256_bytes, sha256_file, write_json_atomic
from .vision import Transcriber
from .vision.openai_vision import OpenAIVisionTranscriber

log = logging.getLogger(__name__)

_PUNCT_TRANSLATION = str.maketrans(
    {
        "«": '"',
        "»": '"',
        "“": '"',
        "”": '"',
        "„": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
    }
)


@dataclass
class VerifyResult:
    document: VerificationDocument
    output_path: Path
    best_paths: dict[str, Path]
    resumed: bool
    needs_attention: int


@dataclass
class _Expected:
    kind: str
    text: str
    normalized: str
    threshold: float | None


@dataclass
class _Match:
    check: StringCheck
    consume: tuple[int, ...] = ()


def normalize_compare(text: str) -> str:
    value = unicodedata.normalize("NFC", text).translate(_PUNCT_TRANSLATION)
    value = value.casefold()
    value = re.sub(r"\s+", " ", value).strip()
    while value and (unicodedata.category(value[0]).startswith("P") or value[0].isspace()):
        value = value[1:]
    while value and (unicodedata.category(value[-1]).startswith("P") or value[-1].isspace()):
        value = value[:-1]
    return value.strip()


def _contains(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    pattern = r"(?<!\w)" + re.escape(needle) + r"(?!\w)"
    return re.search(pattern, haystack) is not None


def _expected_checks(spec: PosterSpecDocument, threshold: float) -> list[_Expected]:
    expected: list[_Expected] = [
        _Expected(
            "title",
            spec.central_idea.title,
            normalize_compare(spec.central_idea.title),
            None,
        ),
        _Expected(
            "sentence",
            spec.central_idea.definition,
            normalize_compare(spec.central_idea.definition),
            threshold,
        ),
    ]
    canonical = {item.normalized for item in expected}
    for detail in spec.details:
        for kind, value, is_sentence in (
            ("label", detail.label, False),
            ("sentence", detail.explanation, True),
            ("sentence", detail.example, True),
        ):
            if not value:
                continue
            normalized = normalize_compare(value)
            if not normalized:
                continue
            expected.append(
                _Expected(
                    kind,
                    value,
                    normalized,
                    threshold if is_sentence else None,
                )
            )
            canonical.add(normalized)
    for value in spec.strings_to_render:
        normalized = normalize_compare(value)
        if not normalized or normalized in canonical:
            continue
        expected.append(_Expected("term", value, normalized, None))
        canonical.add(normalized)
    return expected


_LIST_PREFIX = re.compile(r"^\s*(?:\d{1,3}[.)]|[-*•])\s*")


def _strip_list_prefix(line: str) -> str:
    return _LIST_PREFIX.sub("", line.strip()).strip()


def _line_candidates(lines: list[str]) -> list[tuple[tuple[int, ...], str]]:
    candidates: list[tuple[tuple[int, ...], str]] = []
    for index, line in enumerate(lines):
        if line:
            candidates.append(((index,), line))
    for width in (2, 3):
        for start in range(len(lines) - width + 1):
            joined = " ".join(line for line in lines[start : start + width] if line)
            if joined:
                candidates.append((tuple(range(start, start + width)), joined))
    return candidates


def _compare(spec: PosterSpecDocument, transcript: str, threshold: float) -> tuple:
    expected = _expected_checks(spec, threshold)
    raw_lines = [_strip_list_prefix(line) for line in transcript.splitlines()]
    normalized_lines = [normalize_compare(line) for line in raw_lines]
    lines = [line for line in normalized_lines if line]
    original_lines = [line for line, normalized in zip(raw_lines, normalized_lines) if normalized]
    full = normalize_compare(transcript)
    candidates = _line_candidates(lines)

    matches: list[_Match] = []
    explained: set[str] = set()
    for item in expected:
        if item.kind == "sentence":
            if _contains(full, item.normalized):
                match = _Match(
                    StringCheck(
                        kind="sentence",
                        expected=item.text,
                        normalized=item.normalized,
                        best_match=item.text,
                        similarity=1.0,
                        status="similar",
                        threshold=threshold,
                    )
                )
                explained.add(item.normalized)
            else:
                best_indices: tuple[int, ...] = ()
                best_text = ""
                best_ratio = 0.0
                for indices, candidate in candidates:
                    ratio = SequenceMatcher(None, item.normalized, candidate).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_text = candidate
                        best_indices = indices
                passed = best_ratio >= threshold
                match = _Match(
                    StringCheck(
                        kind="sentence",
                        expected=item.text,
                        normalized=item.normalized,
                        best_match=best_text or None,
                        similarity=best_ratio,
                        status="similar" if passed else "missing",
                        threshold=threshold,
                    ),
                    consume=best_indices if passed else (),
                )
                if passed:
                    explained.add(best_text)
                    for index in best_indices:
                        if index < len(lines):
                            explained.add(lines[index])
        else:
            if _contains(full, item.normalized):
                match = _Match(
                    StringCheck(
                        kind=item.kind,  # type: ignore[arg-type]
                        expected=item.text,
                        normalized=item.normalized,
                        best_match=item.text,
                        similarity=1.0,
                        status="exact",
                    )
                )
                explained.add(item.normalized)
            else:
                match = _Match(
                    StringCheck(
                        kind=item.kind,  # type: ignore[arg-type]
                        expected=item.text,
                        normalized=item.normalized,
                        best_match=None,
                        similarity=0.0,
                        status="missing",
                    )
                )
        matches.append(match)

    extra: list[ExtraString] = []
    seen_extra: set[str] = set()
    expected_values = [item.normalized for item in expected if item.normalized]
    for index, value in enumerate(lines):
        normalized = normalize_compare(value)
        if len(normalized) < 2 or normalized in seen_extra:
            continue
        if normalized in explained or any(
            normalized in expected_value for expected_value in expected_values
        ):
            continue
        seen_extra.add(normalized)
        original = original_lines[index] if index < len(original_lines) else value
        extra.append(ExtraString(text=original, normalized=normalized))

    checks = [match.check for match in matches]
    exact_mismatches = [
        check.expected for check in checks if check.kind != "sentence" and check.status == "missing"
    ]
    below_threshold = [
        check.expected for check in checks if check.kind == "sentence" and check.status == "missing"
    ]
    score = sum(check.similarity for check in checks) / len(checks) if checks else 0.0
    score = max(0.0, score - 0.05 * len(extra))
    passed = bool(checks) and not exact_mismatches and not below_threshold and not extra
    return checks, extra, exact_mismatches, below_threshold, score, passed


def _vision_error(exc: Exception) -> GenerationErrorInfo:
    return GenerationErrorInfo(
        code="vision_failed",
        message=f"Vision transcription failed: {exc}",
        moderation=False,
        retryable=True,
    )


def _verify_image(
    transcriber: Transcriber,
    image_path: Path,
    *,
    spec: PosterSpecDocument,
    attempt: int,
    similarity_threshold: float,
    paths: RunPaths,
) -> VerificationAttempt:
    image_sha = sha256_file(image_path)
    try:
        transcript = transcriber.transcribe(image_path, attempt)
    except Exception as exc:  # noqa: BLE001 - transcribed failure is recorded
        return VerificationAttempt(
            attempt=attempt,
            image_path=paths.relative(image_path),
            image_sha256=image_sha,
            transcript="",
            checks=[],
            extra_strings=[],
            exact_mismatches=[],
            below_threshold=[],
            score=0.0,
            passed=False,
            text_passed=False,
            error=_vision_error(exc),
            verified_at=utc_now(),
        )
    checks, extra, exact_mismatches, below, score, passed = _compare(
        spec, transcript, similarity_threshold
    )
    return VerificationAttempt(
        attempt=attempt,
        image_path=paths.relative(image_path),
        image_sha256=image_sha,
        transcript=transcript,
        checks=checks,
        extra_strings=extra,
        exact_mismatches=exact_mismatches,
        below_threshold=below,
        score=score,
        passed=passed,
        text_passed=passed,
        error=None,
        verified_at=utc_now(),
    )


def _correction_note(attempt: VerificationAttempt) -> str:
    mascot_corner = attempt.mascot_required_corner
    mascot_failed = bool(mascot_corner and attempt.text_passed and not attempt.passed)
    wrong = [
        check for check in attempt.checks if check.kind != "sentence" and check.status == "missing"
    ] + [
        check for check in attempt.checks if check.kind == "sentence" and check.status == "missing"
    ]
    if not wrong and not attempt.extra_strings and not mascot_failed:
        return ""
    lines = [
        "CORRECTION NOTE FOR THIS RETRY",
        "The previous poster had text-rendering errors. Correct exactly these strings:",
    ]
    for check in wrong[:20]:
        if check.kind == "sentence":
            lines.append(
                f"- Render this sentence exactly: {json.dumps(check.expected, ensure_ascii=False)} "
                f"(previous similarity {check.similarity:.2f})."
            )
        else:
            previous = (
                f" Previous nearest text: {json.dumps(check.best_match, ensure_ascii=False)}."
                if check.best_match and check.best_match != check.expected
                else ""
            )
            lines.append(
                f"- Render this {check.kind} exactly: "
                f"{json.dumps(check.expected, ensure_ascii=False)}.{previous}"
            )
    for extra in attempt.extra_strings[:10]:
        lines.append(f"- Remove this extra text: {json.dumps(extra.text, ensure_ascii=False)}.")
    if mascot_failed:
        corner = str(mascot_corner).replace("_", " ")
        lines.append(
            f"- Keep the {corner} corner completely empty: no text, labels, numbers, "
            "or key illustrations. A mascot and its speech bubble are placed there."
        )
    lines.append(
        "Keep every other string, the composition, and the style unchanged. "
        "Render the corrected strings exactly once."
    )
    return "\n".join(lines)


def _retry_image(
    provider: ImageProvider,
    *,
    original_prompt: str,
    note: str,
    size: str,
    format: str,
    attempt: int,
    paths: RunPaths,
    stage_dir: Path,
) -> tuple[Path | None, GenerationErrorInfo | None]:
    prompt = original_prompt if not note else f"{original_prompt}\n\n{note}"
    try:
        data = provider.generate(prompt, size)
        image = decode_image(data)
    except ModerationRefusal:
        try:
            data = provider.generate(soften_prompt(prompt), size)
            image = decode_image(data)
        except ModerationRefusal as exc:
            return None, GenerationErrorInfo(**exc.to_dict())
        except ImageProviderError as exc:
            return None, GenerationErrorInfo(**exc.to_dict())
        except Exception as exc:  # noqa: BLE001 - provider boundary
            return (
                None,
                GenerationErrorInfo(
                    code="image_failed",
                    message=f"Correction retry failed: {exc}",
                    retryable=True,
                ),
            )
    except ImageProviderError as exc:
        return None, GenerationErrorInfo(**exc.to_dict())
    except Exception as exc:  # noqa: BLE001 - provider boundary
        return (
            None,
            GenerationErrorInfo(
                code="image_failed",
                message=f"Correction retry failed: {exc}",
                retryable=True,
            ),
        )
    try:
        image_path = save_image(
            paths,
            stage_dir,
            base_stem=f"{format}-attempt-{attempt}",
            image=image,
            subdir=f"attempts/{format}",
        )
    except Exception as exc:  # noqa: BLE001 - save failure is recorded
        return (
            None,
            GenerationErrorInfo(
                code="image_save_failed",
                message=f"Could not save correction attempt: {exc}",
                retryable=False,
            ),
        )
    return image_path, None


def _best_attempt(attempts: list[VerificationAttempt]) -> VerificationAttempt | None:
    valid = [attempt for attempt in attempts if attempt.image_path is not None]
    if not valid:
        return None
    return max(valid, key=lambda attempt: (attempt.score, -attempt.attempt))


def _save_best(
    best: VerificationAttempt,
    *,
    format_name: str,
    spec: PosterSpecDocument,
    paths: RunPaths,
    stage_dir: Path,
) -> tuple[str | None, str | None, list[str]]:
    if not best.image_path:
        return None, None, []
    try:
        source = paths.resolve(best.image_path)
        data = source.read_bytes()
        image = decode_image(data)
        path = save_image(
            paths,
            stage_dir,
            base_stem=f"{spec.file_slug}-{format_name}-attempt-{best.attempt}",
            image=image,
            subdir="best",
        )
        return paths.relative(path), sha256_bytes(data), []
    except Exception as exc:  # noqa: BLE001 - best-effort copy
        return best.image_path, best.image_sha256, [f"Could not copy best attempt: {exc}"]


def _resolve_verifier(
    manifest: RunManifest,
    options: VerificationOptions,
) -> tuple[str, str, str, bool]:
    planner = manifest.config.get("planner") or {}
    planner_provider = str(planner.get("provider") or "openai")
    planner_model = str(planner.get("model") or default_planner_model())
    capabilities = planner.get("capabilities") or {}
    supports_vision = bool(capabilities.get("vision", planner_provider == "openai"))
    # The verifier adapter is OpenAI-backed. An Anthropic planner with vision
    # still drops to the configured OpenAI vision provider because its model ID
    # cannot be sent to the OpenAI vision endpoint.
    openai_verifier_available = planner_provider == "openai" and supports_vision
    if options.verifier_model:
        verifier_model = options.verifier_model
    elif openai_verifier_available:
        verifier_model = planner_model
    else:
        verifier_model = default_vision_model()
    fallback_used = verifier_model != planner_model or not openai_verifier_available
    return planner_provider, planner_model, verifier_model, fallback_used


def _attempt_output(attempt: VerificationAttempt) -> dict:
    if attempt.error is not None:
        status = "error"
    elif attempt.passed:
        status = "passed"
    else:
        status = "needs_attention"
    return {
        "attempt": attempt.attempt,
        "status": status,
        "passed": attempt.passed,
        "text_passed": attempt.text_passed,
        "score": attempt.score,
        "mascot": attempt.mascot,
        "error": attempt.error.model_dump(mode="json") if attempt.error else None,
    }


def _score_format(span, result: FormatVerification) -> None:
    best = next(
        (attempt for attempt in result.attempts if attempt.attempt == result.best_attempt),
        None,
    )
    span.score("text_match", float(result.best_score or 0.0))
    span.score(
        "exact_mismatch_count",
        float(len(best.exact_mismatches) if best is not None else 0),
    )
    span.score(
        "extra_strings_count",
        float(len(best.extra_strings) if best is not None else 0),
    )
    span.score("retries_used", float(max(0, len(result.attempts) - 1)))
    span.score("verify_status", result.status, data_type="CATEGORICAL")


_MIN_MASCOT_VERIFY_CANVAS = 256


def _resolve_attempt_mascot(
    attempt: VerificationAttempt,
    *,
    image_path: Path,
    entry,
    config,
    assets: dict | None,
    run_id: str,
    format_name: str,
    warnings: list[str],
) -> VerificationAttempt:
    """Attach an accepted mascot placement to a text-passing attempt."""
    if entry is None or assets is None or not attempt.passed or not attempt.text_passed:
        return attempt
    try:
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
    except OSError as exc:
        warnings.append(f"{format_name}: mascot placement could not read the image: {exc}")
        return attempt
    if min(image.size) < _MIN_MASCOT_VERIFY_CANVAS:
        return attempt

    preferred = getattr(entry, "preferred_corner", None) or getattr(entry, "corner", None)
    try:
        with tracing.stage_span(
            "mascot.place",
            run_id=run_id,
            input={
                "format": format_name,
                "attempt": attempt.attempt,
                "preferred_corner": preferred,
            },
            metadata={"stage": "verify", "format": format_name},
        ) as span:
            result = resolve_placement(
                image,
                entry.question,
                config=config,
                entry=entry,
                assets=assets,
            )
            if isinstance(result, Placement):
                record = {
                    "corner": result.corner,
                    "rect": result.layout.block.as_dict(),
                    "method": result.method,
                    "text_boxes_hit": result.score.text_boxes_hit,
                }
                attempt.mascot = record
                attempt.mascot_required_corner = result.preferred_corner
                span.update(output={"status": "accepted", **record})
                span.score("font_size", float(entry.font_size))
                span.score(
                    "corner_changed",
                    0.0 if result.corner == result.preferred_corner else 1.0,
                )
                span.score("text_boxes_hit", float(result.score.text_boxes_hit))
                span.score("busyness", float(result.score.busyness))
            elif isinstance(result, PlacementFailure):
                record = {
                    "corner": None,
                    "rect": None,
                    "method": "none",
                    "text_boxes_hit": result.text_boxes_hit,
                }
                attempt.mascot = record
                attempt.mascot_required_corner = result.preferred_corner
                attempt.passed = False
                span.update(output={"status": "no_clear_corner", **record})
                span.score("font_size", float(entry.font_size))
                span.score("corner_changed", 0.0)
                span.score("text_boxes_hit", float(result.text_boxes_hit))
                span.score("busyness", 1.0)
    except GenLessonError as exc:
        warnings.append(f"{format_name}: mascot placement skipped: {exc.message}")
    return attempt


def _verify_one_format(
    prompt: PromptFormatSpec,
    generated: GeneratedFormat | None,
    *,
    transcriber: Transcriber,
    spec: PosterSpecDocument,
    options: VerificationOptions,
    paths: RunPaths,
    stage_dir: Path,
    image_model: str,
    image_provider: ImageProvider | None,
    image_provider_error: GenLessonError | None,
    warnings: list[str],
    mascot_entry=None,
    mascot_config=None,
    mascot_assets: dict | None = None,
) -> tuple[FormatVerification, int, ImageProvider | None, GenLessonError | None]:
    if generated is None or generated.status != "generated" or not generated.image_path:
        warnings.append(f"{prompt.format}: generation failed; verification skipped.")
        return (
            FormatVerification(
                format=prompt.format,
                status="generation_failed",
                attempts=[],
                warnings=["Generation did not produce an image for this format."],
            ),
            0,
            image_provider,
            image_provider_error,
        )
    image_path = paths.resolve(generated.image_path)
    if not image_path.is_file():
        warnings.append(f"{prompt.format}: generated image file is missing.")
        return (
            FormatVerification(
                format=prompt.format,
                status="generation_failed",
                attempts=[],
                warnings=["Generated image file is missing."],
            ),
            0,
            image_provider,
            image_provider_error,
        )

    attempts: list[VerificationAttempt] = []
    with tracing.stage_span(
        "verify.attempt:1",
        run_id=paths.run_id,
        input={"format": prompt.format, "attempt": 1, "image": generated.image_path},
        metadata={"stage": "verify", "format": prompt.format, "attempt": 1},
    ) as attempt_span:
        attempt = _verify_image(
            transcriber,
            image_path,
            spec=spec,
            attempt=1,
            similarity_threshold=options.similarity_threshold,
            paths=paths,
        )
        attempt = _resolve_attempt_mascot(
            attempt,
            image_path=image_path,
            entry=mascot_entry,
            config=mascot_config,
            assets=mascot_assets,
            run_id=paths.run_id,
            format_name=prompt.format,
            warnings=warnings,
        )
        attempts.append(attempt)
        attempt_span.update(output=_attempt_output(attempt))

    retries = 0
    while (
        not attempts[-1].passed
        and attempts[-1].error is None
        and len(attempts) <= options.max_retries
    ):
        if image_provider is None and image_provider_error is None:
            try:
                image_provider = build_image_provider(options.image_options(), model=image_model)
            except GenLessonError as exc:
                image_provider_error = exc
                warnings.append(f"{prompt.format}: cannot retry generation: {exc.message}")
                break
        if image_provider is None:
            break
        note = _correction_note(attempts[-1])
        attempt_no = len(attempts) + 1
        with (
            tracing.stage_span(
                f"verify.attempt:{attempt_no}",
                run_id=paths.run_id,
                input={"format": prompt.format, "attempt": attempt_no},
                metadata={"stage": "verify", "format": prompt.format, "attempt": attempt_no},
            ) as attempt_span,
            tracing.stage_span(
                "verify.retry_image",
                run_id=paths.run_id,
                input={"format": prompt.format, "attempt": attempt_no},
                metadata={"stage": "verify", "format": prompt.format, "attempt": attempt_no},
            ) as retry_span,
        ):
            retry_path, error = _retry_image(
                image_provider,
                original_prompt=prompt.prompt,
                note=note,
                size=f"{prompt.width}x{prompt.height}",
                format=prompt.format,
                attempt=attempt_no,
                paths=paths,
                stage_dir=stage_dir,
            )
            retry_span.update(
                output={
                    "status": "generated" if retry_path is not None else "failed",
                    "error": error.model_dump(mode="json") if error else None,
                }
            )
            retries += 1
            if error is not None or retry_path is None:
                attempt = VerificationAttempt(
                    attempt=attempt_no,
                    image_path=None,
                    transcript="",
                    checks=[],
                    extra_strings=[],
                    exact_mismatches=[],
                    below_threshold=[],
                    score=0.0,
                    passed=False,
                    error=error,
                    verified_at=utc_now(),
                )
                attempts.append(attempt)
                attempt_span.update(output=_attempt_output(attempt))
                continue
            attempt = _verify_image(
                transcriber,
                retry_path,
                spec=spec,
                attempt=attempt_no,
                similarity_threshold=options.similarity_threshold,
                paths=paths,
            )
            attempt = _resolve_attempt_mascot(
                attempt,
                image_path=retry_path,
                entry=mascot_entry,
                config=mascot_config,
                assets=mascot_assets,
                run_id=paths.run_id,
                format_name=prompt.format,
                warnings=warnings,
            )
            attempts.append(attempt)
            attempt_span.update(output=_attempt_output(attempt))
        if attempts[-1].error is not None:
            break

    best = _best_attempt(attempts)
    passed_attempt = next((attempt for attempt in attempts if attempt.passed), None)
    mascot_record = passed_attempt.mascot if passed_attempt else None
    mascot_reason: str | None = None
    if (
        passed_attempt is None
        and best is not None
        and best.text_passed
        and best.mascot_required_corner is not None
    ):
        mascot_reason = "mascot_no_clear_corner"
        mascot_record = best.mascot
    if passed_attempt is not None:
        status = "passed"
    elif best is not None:
        status = "needs_attention"
    else:
        status = "generation_failed"
    best_path = None
    best_sha = None
    best_warnings: list[str] = []
    if best is not None:
        best_path, best_sha, copy_warnings = _save_best(
            best,
            format_name=prompt.format,
            spec=spec,
            paths=paths,
            stage_dir=stage_dir,
        )
        best_warnings.extend(copy_warnings)
    format_warnings = list(best_warnings)
    for attempt in attempts:
        if attempt.error is not None:
            format_warnings.append(f"attempt {attempt.attempt}: {attempt.error.message}")
            warnings.append(
                f"{prompt.format}: attempt {attempt.attempt} failed: {attempt.error.message}"
            )
    if mascot_reason == "mascot_no_clear_corner":
        format_warnings.append(
            "Retries exhausted; no clear corner for the mascot (mascot_no_clear_corner)."
        )
        warnings.append(
            f"{prompt.format}: no clear corner for the mascot; the poster will be saved without it."
        )
    elif status == "needs_attention":
        format_warnings.append("Retries exhausted; best attempt needs attention.")
        warnings.append(f"{prompt.format}: needs attention after verification retries.")
    return (
        FormatVerification(
            format=prompt.format,
            status=status,  # type: ignore[arg-type]
            mascot=mascot_record,
            mascot_reason=mascot_reason,
            reason=mascot_reason,
            attempts=attempts,
            best_attempt=best.attempt if best else None,
            best_image_path=best_path,
            best_image_sha256=best_sha,
            best_score=best.score if best else None,
            warnings=format_warnings,
        ),
        retries,
        image_provider,
        image_provider_error,
    )


def run_verify(
    paths: RunPaths,
    manifest: RunManifest,
    store: RunStore,
    options: VerificationOptions,
    *,
    transcriber: Transcriber | None = None,
    image_provider: ImageProvider | None = None,
) -> VerifyResult:
    options.validate()
    source = manifest.source
    if source is None:
        raise GenLessonError("manifest_invalid", "Run manifest is missing source metadata.")
    generation_stage = manifest.stages.get("generate")
    if generation_stage is None:
        raise GenLessonError("manifest_invalid", "Run manifest is missing the generate stage.")
    if generation_stage.status != "complete":
        raise GenLessonError(
            "stage_incomplete",
            "Generation must complete before verification can run.",
            hint=f"Run `genlesson generate {paths.run_id}` first.",
        )

    prompt_path = paths.stage_dir("prompts") / "prompts.json"
    generation_path = paths.stage_dir("generate") / "generation.json"
    review_path = paths.stage_dir("review") / "review.json"
    prompts = read_model(prompt_path, PromptBundle)
    generation = read_model(generation_path, GenerationDocument)
    review = read_model(review_path, ReviewDocument)
    mascot_config = load_app_config().mascot
    mascot_layout_path = paths.stage_dir("prompts") / "mascot_layout.json"
    mascot_layout = (
        read_model(mascot_layout_path, MascotLayoutDocument)
        if mascot_layout_path.is_file()
        else None
    )
    if mascot_layout is not None and (
        mascot_layout.source_sha256 != source.sha256
        or (
            review.approved_spec_sha256
            and mascot_layout.approved_spec_sha256 != review.approved_spec_sha256
        )
    ):
        raise GenLessonError("artifact_invalid", "mascot_layout.json does not match the run spec.")
    mascot_by_format = (
        {entry.format: entry for entry in mascot_layout.formats}
        if mascot_layout is not None
        else {}
    )
    mascot_assets: dict | None = None
    if mascot_config.enabled and mascot_by_format:
        try:
            mascot_assets = load_mascot_assets(mascot_config.assets_dir)
        except GenLessonError as exc:
            log.warning("Mascot assets unavailable during verification: %s", exc)
            mascot_assets = None
    if prompts.source_sha256 != source.sha256 or generation.source_sha256 != source.sha256:
        raise GenLessonError("artifact_invalid", "Prompts or generation do not match the source.")
    if (
        generation.prompts_sha256 != sha256_file(prompt_path)
        or prompts.review_sha256 != sha256_file(review_path)
        or prompts.approved_spec_sha256 != review.approved_spec_sha256
    ):
        raise GenLessonError(
            "artifact_invalid", "Generation does not match the prompt/review chain."
        )
    if not review.approved_spec_path or not review.approved_spec_sha256:
        raise GenLessonError(
            "artifact_invalid", "Review artifact does not point to an approved spec."
        )
    approved_path = paths.resolve(review.approved_spec_path)
    if sha256_file(approved_path) != review.approved_spec_sha256:
        raise GenLessonError("artifact_invalid", "approved_spec.json does not match review.json.")
    spec = read_model(approved_path, PosterSpecDocument)

    planner_provider, planner_model, verifier_model, fallback_used = _resolve_verifier(
        manifest, options
    )
    config = VerificationConfig(
        verifier_provider=options.provider,
        verifier_model=verifier_model,
        planner_provider=planner_provider,
        planner_model=planner_model,
        fallback_used=fallback_used,
        similarity_threshold=options.similarity_threshold,
        max_retries=options.max_retries,
        image_model=prompts.image_model,
    )

    stage = manifest.stages.get("verify")
    if stage is None:
        raise GenLessonError("manifest_invalid", "Run manifest is missing the verify stage.")
    stage_dir = paths.stage_dir("verify")
    stage_dir.mkdir(parents=True, exist_ok=True)
    output_path = stage_dir / "verification.json"
    if stage.status == "complete":
        document = read_model(output_path, VerificationDocument)
        if document.generation_sha256 == sha256_file(
            generation_path
        ) and document.prompts_sha256 == sha256_file(prompt_path):
            return VerifyResult(
                document=document,
                output_path=output_path,
                best_paths={
                    item.format: paths.resolve(item.best_image_path)
                    for item in document.formats
                    if item.best_image_path
                },
                resumed=True,
                needs_attention=document.stats.needs_attention,
            )
        # The generation or prompts changed; fall through and verify the new set.

    if transcriber is None:
        if not options.api_key:
            raise GenLessonError(
                "vision_provider_required",
                "Stage 7 needs an OpenAI API key for vision verification.",
                hint="Set OPENAI_API_KEY and run verify again.",
            )
        transcriber = OpenAIVisionTranscriber(
            model=verifier_model,
            api_key=options.api_key,
            base_url=options.base_url,
            timeout_s=options.vision_timeout_s,
            max_retries=options.vision_max_retries,
            call_name="verify.transcribe",
        )

    generation_by_format: dict[str, GeneratedFormat] = {
        item.format: item for item in generation.formats
    }
    formats: list[FormatVerification] = []
    warnings: list[str] = []
    retries = 0
    image_provider_error: GenLessonError | None = None

    store.set_stage(paths, manifest, "verify", "running")
    try:
        for prompt in prompts.formats:
            generated = generation_by_format.get(prompt.format)
            with tracing.stage_span(
                f"verify:{prompt.format}",
                run_id=paths.run_id,
                input={
                    "format": prompt.format,
                    "image": generated.image_path if generated else None,
                },
                metadata={"stage": "verify", "format": prompt.format},
            ) as format_span:
                result, format_retries, image_provider, image_provider_error = _verify_one_format(
                    prompt,
                    generated,
                    transcriber=transcriber,
                    spec=spec,
                    options=options,
                    paths=paths,
                    stage_dir=stage_dir,
                    image_model=prompts.image_model,
                    image_provider=image_provider,
                    image_provider_error=image_provider_error,
                    warnings=warnings,
                    mascot_entry=mascot_by_format.get(prompt.format),
                    mascot_config=mascot_config,
                    mascot_assets=mascot_assets,
                )
                retries += format_retries
                formats.append(result)
                format_span.update(
                    output={
                        "status": result.status,
                        "attempts": len(result.attempts),
                        "best_score": result.best_score,
                        "mascot": result.mascot,
                        "mascot_reason": result.mascot_reason,
                    }
                )
                _score_format(format_span, result)

    except GenLessonError as exc:
        store.set_stage(paths, manifest, "verify", "failed", error=exc.to_dict())
        raise
    except Exception as exc:
        wrapped = GenLessonError(
            "internal_error",
            f"Verification failed unexpectedly: {exc}",
            details={"exception": type(exc).__name__},
        )
        store.set_stage(paths, manifest, "verify", "failed", error=wrapped.to_dict())
        raise wrapped from exc

    stats = VerificationStats(
        formats=len(formats),
        passed=sum(1 for item in formats if item.status == "passed"),
        needs_attention=sum(1 for item in formats if item.status == "needs_attention"),
        generation_failed=sum(1 for item in formats if item.status == "generation_failed"),
        retries=retries,
    )
    document = VerificationDocument(
        source_sha256=source.sha256,
        approved_spec_sha256=review.approved_spec_sha256,
        review_sha256=sha256_file(review_path),
        prompts_sha256=sha256_file(prompt_path),
        generation_sha256=sha256_file(generation_path),
        formats=formats,
        warnings=warnings,
        stats=stats,
        config=config,
        verified_at=utc_now(),
    )
    write_json_atomic(output_path, document.model_dump(mode="json"))
    manifest.config["verification"] = {
        "verifier_model": verifier_model,
        "planner_model": planner_model,
        "fallback_used": fallback_used,
        "passed": stats.passed,
        "needs_attention": stats.needs_attention,
        "generation_failed": stats.generation_failed,
    }
    store.set_stage(paths, manifest, "verify", "complete", output=paths.relative(output_path))
    return VerifyResult(
        document=document,
        output_path=output_path,
        best_paths={
            item.format: paths.resolve(item.best_image_path)
            for item in document.formats
            if item.best_image_path
        },
        resumed=False,
        needs_attention=stats.needs_attention,
    )
