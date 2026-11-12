from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw

from genlesson.app_config import clear_config_cache, load_app_config, set_active_config_path
from genlesson.config import VerificationOptions
from genlesson.errors import GenLessonError
from genlesson.mascot.assets import load_mascot_assets
from genlesson.mascot.compose import check_text_inside_bubble, render_preview
from genlesson.mascot.layout import layout_mascot
from genlesson.mascot.select import (
    Placement,
    PlacementFailure,
    choose_variant,
    plan_question,
    resolve_placement,
)
from genlesson.mascot.zones import NullTextDetector, Rect
from genlesson.models import (
    GenerationDocument,
    MascotLayoutDocument,
    PosterSpecDocument,
    RunManifest,
)
from genlesson.save import run_save
from genlesson.storage import read_model
from genlesson.verify import run_verify

from .helpers import FakeImageProvider, FakeVisionTranscriber
from .test_verify import _exact_transcript, _generated_run

REPO_ROOT = Path(__file__).resolve().parents[1]
FONT = REPO_ROOT / "assets" / "fonts" / "NotoSans-Regular.ttf"


@pytest.fixture(autouse=True)
def _clear_config():
    clear_config_cache()
    set_active_config_path(None)
    yield
    clear_config_cache()
    set_active_config_path(None)


def _config():
    return load_app_config().mascot


def _busy_png(width: int = 1024, height: int = 1536) -> bytes:
    image = Image.new("RGB", (width, height), (70, 70, 70))
    draw = ImageDraw.Draw(image)
    for x in range(0, width, 8):
        for y in range(0, height, 8):
            draw.rectangle((x, y, x + 4, y + 4), fill=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _blank_png(width: int = 1024, height: int = 1536) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (245, 242, 235)).save(buffer, format="PNG")
    return buffer.getvalue()


def _replace_generated_images(paths, png: bytes) -> None:
    generation = read_model(paths.stage_dir("generate") / "generation.json", GenerationDocument)
    for item in generation.formats:
        if item.image_path:
            paths.resolve(item.image_path).write_bytes(png)


def test_font_covers_all_nine_kazakh_letters() -> None:
    font = TTFont(str(FONT))
    cmap: set[int] = set()
    for table in font["cmap"].tables:
        cmap.update(table.cmap)
    for letter in "әғқңөұүһі":
        assert ord(letter) in cmap, f"font is missing {letter!r}"
    for letter in "ӘҒҚҢӨҰҮҺІ":
        assert ord(letter) in cmap, f"font is missing {letter!r}"


def test_long_question_stays_inside_bubble_padding() -> None:
    question = (
        "Бұл сабақта қарастырылған негізгі идеяны өз сөзіңмен қалай "
        "түсіндірер едің және ол күнделікті өмірде қалай көрінеді?"
    )
    composition = render_preview(
        1024,
        1536,
        question,
        _config(),
        corner="bottom_right",
        variant="Vector",
        detector=NullTextDetector(),
    )
    result = check_text_inside_bubble(composition.image, composition.layout, _config())
    assert result["checked"] is True
    assert len(composition.layout.lines) >= 3
    text_bbox = result["text_bbox"]
    inner = result["inner"]
    assert text_bbox[0] >= inner["x"]
    assert text_bbox[1] >= inner["y"]
    assert text_bbox[2] < inner["x"] + inner["width"]
    assert text_bbox[3] < inner["y"] + inner["height"]
    top_gap = text_bbox[1] - inner["y"]
    bottom_gap = inner["y"] + inner["height"] - text_bbox[3]
    assert abs(top_gap - bottom_gap) <= 3


def test_overflow_hint_points_at_review() -> None:
    with pytest.raises(GenLessonError) as excinfo:
        layout_mascot(
            1024,
            1536,
            "А" * 300,
            corner="bottom_right",
            config=_config(),
            mascot_size=(213, 188),
        )
    assert excinfo.value.code == "mascot_text_overflow"
    assert "review" in (excinfo.value.hint or "").lower()


def test_seeded_question_draws_cover_all_five_icon_variants() -> None:
    variants = sorted(load_mascot_assets(_config().assets_dir))
    assert len(variants) == 5
    drawn = {
        choose_variant(f"Сұрақ нөмірі {index}?", variants, salt=str(index)) for index in range(200)
    }
    assert drawn == set(variants)


def test_overlap_moves_placement_and_busyness_rejects() -> None:
    config = _config()
    assets = load_mascot_assets(config.assets_dir)
    variant = min(assets)
    entry = type(
        "Entry",
        (),
        {
            "question": "«Байланыс түрлері» туралы не білесің?",
            "preferred_corner": "bottom_right",
            "corner": "bottom_right",
            "variant": variant,
            "font_size": 29,
        },
    )()
    image = Image.new("RGB", (1024, 1536), "white")
    layout = layout_mascot(
        image.width,
        image.height,
        entry.question,
        "bottom_right",
        config,
        mascot_size=assets[variant].size,
    )

    class BoxDetector:
        name = "synthetic-boxes"

        def detect(self, crop: Image.Image):
            import numpy as np

            array = np.asarray(crop.convert("L"))
            ys, xs = (array < 40).nonzero()
            if len(xs) == 0:
                return ()
            return (
                Rect(
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max() - xs.min() + 1),
                    int(ys.max() - ys.min() + 1),
                ),
            )

    draw = ImageDraw.Draw(image)
    draw.rectangle(layout.block.box, fill=(0, 0, 0))
    placement = resolve_placement(
        image,
        entry.question,
        config=config,
        entry=entry,
        detector=BoxDetector(),
        assets=assets,
    )
    assert isinstance(placement, Placement)
    assert placement.corner != "bottom_right"
    assert placement.method == "detected"

    busy = Image.open(io.BytesIO(_busy_png()))
    failure = resolve_placement(
        busy,
        entry.question,
        config=config,
        entry=entry,
        detector=BoxDetector(),
        assets=assets,
    )
    assert isinstance(failure, PlacementFailure)
    assert failure.preferred_corner == "bottom_right"


def test_question_plan_language_length_and_source() -> None:
    from types import SimpleNamespace

    config = _config()
    planner = SimpleNamespace(
        language="kk",
        mascot_question="Бұл сабақтағы басты ойды қалай түсінесің?",
        central_idea=SimpleNamespace(title="Байланыс түрлері"),
    )
    planned = plan_question(planner, config)
    assert planned.source == "planner"
    assert planned.text == planner.mascot_question

    derived = SimpleNamespace(
        language="kk",
        mascot_question=None,
        central_idea=SimpleNamespace(title="Байланыс түрлері"),
    )
    planned = plan_question(derived, config)
    assert planned.source == "central_idea"
    assert "Байланыс" in planned.text
    assert planned.text.endswith("?")

    fallback = SimpleNamespace(
        language="ru",
        mascot_question=None,
        central_idea=SimpleNamespace(title=""),
    )
    planned = plan_question(fallback, config)
    assert planned.source == "fallback"
    assert "Что ты знаешь" in planned.text

    too_long = SimpleNamespace(
        language="kk",
        mascot_question="Сұрақ " * 80,
        central_idea=SimpleNamespace(title="Байланыс"),
    )
    planned = plan_question(too_long, config)
    assert len(planned.text) <= config.question_max_chars


def test_stage3_question_is_not_in_strings_to_render(tmp_path: Path) -> None:
    _store, paths, _manifest, spec = _generated_run(tmp_path, formats=("portrait",))
    assert spec.mascot_question
    assert spec.mascot_question not in spec.strings_to_render
    layout = read_model(paths.stage_dir("prompts") / "mascot_layout.json", MascotLayoutDocument)
    assert layout.formats[0].question_source == "planner"


def test_golden_spec_without_mascot_question_is_back_compatible() -> None:
    spec_path = REPO_ROOT / "fixtures" / "expected" / "relationtype_approved_spec.json"
    spec = PosterSpecDocument.model_validate_json(spec_path.read_text(encoding="utf-8"))
    assert spec.mascot_question is None
    planned = plan_question(spec, _config())
    assert planned.source == "central_idea"
    assert planned.text.endswith("?")


class _BusyProvider(FakeImageProvider):
    def generate(self, prompt: str, size: str) -> bytes:
        self.calls.append(("portrait", prompt, size))
        return _busy_png()


def _verify_with_images(
    paths,
    manifest,
    store,
    spec,
    *,
    png: bytes,
    provider=None,
    max_retries: int = 0,
):
    _replace_generated_images(paths, png)
    return run_verify(
        paths,
        manifest,
        store,
        VerificationOptions(
            api_key="test-key",
            similarity_threshold=0.9,
            max_retries=max_retries,
        ),
        transcriber=FakeVisionTranscriber(_exact_transcript(spec)),
        image_provider=provider,
    )


def test_stage7_no_clear_corner_retries_and_marks_reason(tmp_path: Path, monkeypatch) -> None:
    from genlesson.mascot import zones

    monkeypatch.setattr(zones, "_DETECTOR", NullTextDetector())
    store, paths, manifest, spec = _generated_run(tmp_path, formats=("portrait",))
    provider = _BusyProvider()

    result = _verify_with_images(
        paths, manifest, store, spec, png=_busy_png(), provider=provider, max_retries=1
    )

    item = result.document.formats[0]
    assert item.status == "needs_attention"
    assert item.mascot_reason == "mascot_no_clear_corner"
    assert item.reason == "mascot_no_clear_corner"
    assert item.mascot is not None
    assert item.mascot["corner"] is None
    assert item.mascot["method"] == "none"
    assert len(item.attempts) == 2
    assert len(provider.calls) == 1
    assert "corner completely empty" in provider.calls[0][1]


def test_stage7_accepts_non_preferred_corner_without_regeneration(
    tmp_path: Path, monkeypatch
) -> None:
    from genlesson.mascot import zones

    monkeypatch.setattr(zones, "_DETECTOR", NullTextDetector())
    store, paths, manifest, spec = _generated_run(tmp_path, formats=("portrait",))
    provider = _BusyProvider()
    # Blank except for synthetic text in the preferred bottom-right block.  The
    # null detector does not see the text, so use busyness by filling only the
    # preferred block with dense lines.
    image = Image.new("RGB", (1024, 1536), (245, 242, 235))
    layout = layout_mascot(
        1024,
        1536,
        "Байланыс түрлері туралы не білесің?",
        "bottom_right",
        _config(),
        mascot_size=(213, 188),
    )
    draw = ImageDraw.Draw(image)
    for y in range(layout.block.y, layout.block.bottom, 4):
        draw.line((layout.block.x, y, layout.block.right, y), fill=(0, 0, 0), width=2)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    result = _verify_with_images(
        paths, manifest, store, spec, png=buffer.getvalue(), provider=provider
    )

    item = result.document.formats[0]
    assert item.status == "passed"
    assert item.mascot is not None
    assert item.mascot["corner"] != "bottom_right"
    assert item.mascot["method"] == "detected"
    assert provider.calls == []


def test_stage8_sidecar_has_mascot_block_and_raw_final_hashes(tmp_path: Path, monkeypatch) -> None:
    from genlesson.mascot import zones

    monkeypatch.setattr(zones, "_DETECTOR", NullTextDetector())
    store, paths, manifest, spec = _generated_run(tmp_path, formats=("portrait",))
    result = _verify_with_images(paths, manifest, store, spec, png=_blank_png(), provider=None)
    assert result.document.formats[0].status == "passed"
    saved_manifest = read_model(paths.manifest_path, RunManifest)
    saved = run_save(paths, saved_manifest, store, out_dir=tmp_path / "out")
    record = saved.document.files[0]
    sidecar = json.loads(Path(record.sidecar_path).read_text(encoding="utf-8"))
    mascot = sidecar["mascot"]
    assert mascot["applied"] is True
    assert mascot["variant"]
    assert mascot["preferred_corner"] == mascot["final_corner"]
    assert mascot["question"] and mascot["question_source"]
    assert mascot["font_size"] > 0
    assert mascot["rectangle"]["width"] > 0
    assert mascot["raw_sha256"] != mascot["final_sha256"]
    assert mascot["self_check"]["checked"] is True


def test_stage8_saves_without_mascot_after_no_clear_corner(tmp_path: Path, monkeypatch) -> None:
    from genlesson.mascot import zones

    monkeypatch.setattr(zones, "_DETECTOR", NullTextDetector())
    store, paths, manifest, spec = _generated_run(tmp_path, formats=("portrait",))
    provider = _BusyProvider()
    _verify_with_images(
        paths, manifest, store, spec, png=_busy_png(), provider=provider, max_retries=0
    )
    saved_manifest = read_model(paths.manifest_path, RunManifest)
    saved = run_save(paths, saved_manifest, store, out_dir=tmp_path / "out")
    record = saved.document.files[0]
    sidecar = json.loads(Path(record.sidecar_path).read_text(encoding="utf-8"))
    mascot = sidecar["mascot"]
    assert mascot["applied"] is False
    assert mascot["reason"] == "mascot_no_clear_corner"
    assert mascot["raw_sha256"] == mascot["final_sha256"]
    assert any("without it" in warning for warning in saved.document.warnings)


def test_self_check_rejects_text_outside_padding() -> None:
    composition = render_preview(
        1024,
        1536,
        "Бұл сұрақ нені білдіреді?",
        _config(),
        corner="bottom_right",
        variant="Vector",
        detector=NullTextDetector(),
    )
    image = composition.image.copy()
    layout = composition.layout
    # A dark text-coloured pixel just above the padded text area.
    x = layout.bubble.x + layout.bubble.width // 2
    y = layout.bubble.y + max(1, layout.padding[1] - 2)
    image.putpixel((x, y), (59, 7, 100))
    with pytest.raises(GenLessonError) as excinfo:
        check_text_inside_bubble(image, layout, _config())
    assert excinfo.value.code == "mascot_text_outside_bubble"


def test_preview_cli_overlays_an_existing_poster_without_credits(tmp_path: Path) -> None:
    source = tmp_path / "poster.png"
    Image.new("RGB", (1024, 1536), (245, 242, 235)).save(source)
    output = tmp_path / "overlay.png"
    from genlesson.cli import main

    code = main(
        [
            "mascot-preview",
            "--image",
            str(source),
            "--variant",
            "Vector",
            "--question",
            "Бұл сұрақ нені білдіреді?",
            "--out",
            str(output),
            "--detector",
            "none",
            "--json",
        ]
    )
    assert code == 0
    with Image.open(output) as image:
        assert image.size == (1024, 1536)


def test_low_confidence_detections_are_ignored() -> None:
    from types import SimpleNamespace

    from genlesson.mascot.zones import RapidOCRDetector

    class FakeEngine:
        def __call__(self, array, **kwargs):
            return SimpleNamespace(
                boxes=[
                    [[10.0, 10.0], [110.0, 10.0], [110.0, 40.0], [10.0, 40.0]],
                    [[10.0, 60.0], [40.0, 60.0], [40.0, 70.0], [10.0, 70.0]],
                ],
                scores=[0.91, 0.27],
            )

    detector = RapidOCRDetector(min_score=0.5)
    detector._engine = FakeEngine()
    boxes = detector.detect(Image.new("RGB", (200, 100), "white"))
    assert len(boxes) == 1
    assert boxes[0].width == 100
