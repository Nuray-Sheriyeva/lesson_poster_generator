from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from genlesson.app_config import (
    MascotConfig,
    clear_config_cache,
    load_app_config,
    set_active_config_path,
)
from genlesson.cli import main
from genlesson.errors import GenLessonError
from genlesson.mascot.assets import load_mascot_assets
from genlesson.mascot.compose import compose_overlay, render_preview
from genlesson.mascot.layout import layout_mascot, measure_text
from genlesson.mascot.select import Placement, resolve_placement
from genlesson.mascot.zones import NullTextDetector
from genlesson.models import MascotLayoutDocument, PromptBundle
from genlesson.save import _apply_mascot
from genlesson.storage import read_model

from .test_verify import _generated_run

REPO_ROOT = Path(__file__).resolve().parents[1]
FONT = REPO_ROOT / "assets" / "fonts" / "NotoSans-Regular.ttf"


@pytest.fixture(autouse=True)
def _clear_config():
    clear_config_cache()
    set_active_config_path(None)
    yield
    clear_config_cache()
    set_active_config_path(None)


def _make_asset(path: Path, size: int = 512, mode: str = "RGBA") -> Path:
    image = Image.new(mode, (size, size), (46, 196, 182, 255) if mode == "RGBA" else (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((size // 4, size // 4, size * 3 // 4, size * 3 // 4), fill=(255, 255, 255, 255))
    image.save(path)
    return path


def _five_assets(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(5):
        _make_asset(directory / f"variant-{index}.png")
    return directory


def _config(tmp_path: Path, **overrides) -> MascotConfig:
    values = {
        "assets_dir": _five_assets(tmp_path / "mascot"),
        "font_path": FONT,
    }
    values.update(overrides)
    return MascotConfig(**values)


def test_mascot_config_defaults_and_custom_table(tmp_path: Path) -> None:
    default = load_app_config().mascot
    assert default.enabled is True
    assert default.size_ratio == 0.09
    assert default.bubble_max_width_ratio == 0.42
    assert default.question_max_chars == 140

    path = tmp_path / "custom.toml"
    path.write_text(
        f"""
[planner]
provider = "openai"
model = "planner-custom"
[image]
provider = "openai"
model = "gpt-image-2"
[image.sizes.gpt-image-2]
portrait = "1024x1536"
[vision]
provider = "openai"
model = "vision-custom"
[defaults]
formats = ["portrait"]
review = "off"
out_dir = "outputs"
[mascot]
enabled = false
assets_dir = "{tmp_path / "assets"}"
size_ratio = 0.11
fill = "#FAFAFA"
outline = "#123456"
outline_width = 7
font_path = "{FONT}"
min_font_ratio = 0.02
max_font_ratio = 0.05
bubble_max_width_ratio = 0.5
edge_margin = 0.03
question_max_chars = 99
avoid_repeat = true
""",
        encoding="utf-8",
    )
    config = load_app_config(path).mascot
    assert config.enabled is False
    assert config.size_ratio == 0.11
    assert config.fill == "#FAFAFA"
    assert config.outline == "#123456"
    assert config.outline_width == 7
    assert config.question_max_chars == 99
    assert config.avoid_repeat is True


def test_asset_loader_accepts_non_square_icons_and_requires_five(tmp_path: Path) -> None:
    directory = _five_assets(tmp_path / "good")
    assets = load_mascot_assets(directory)
    assert set(assets) == {f"variant-{index}" for index in range(5)}
    assert assets["variant-0"].size == (512, 512)

    # The bundled icons are wider than tall; the loader must preserve that.
    icon_dir = _five_assets(tmp_path / "icons")
    Image.new("RGBA", (213, 188), (0, 0, 0, 255)).save(icon_dir / "variant-0.png")
    icons = load_mascot_assets(icon_dir)
    assert icons["variant-0"].size == (213, 188)
    assert icons["variant-0"].target_size(92) == (104, 92)

    missing = _five_assets(tmp_path / "four")
    (missing / "variant-4.png").unlink()
    with pytest.raises(GenLessonError) as excinfo:
        load_mascot_assets(missing)
    assert excinfo.value.code == "mascot_assets_invalid"

    extra = _five_assets(tmp_path / "six")
    _make_asset(extra / "variant-extra.png")
    with pytest.raises(GenLessonError):
        load_mascot_assets(extra)

    opaque = _five_assets(tmp_path / "opaque")
    Image.new("RGB", (512, 512), (0, 0, 0)).save(opaque / "variant-0.png")
    with pytest.raises(GenLessonError) as excinfo:
        load_mascot_assets(opaque)
    assert "alpha" in excinfo.value.message

    small = _five_assets(tmp_path / "small")
    Image.new("RGBA", (64, 64), (0, 0, 0, 255)).save(small / "variant-0.png")
    with pytest.raises(GenLessonError) as excinfo:
        load_mascot_assets(small)
    assert "short side" in excinfo.value.message


def test_layout_matches_documented_size_ratios(tmp_path: Path) -> None:
    config = _config(tmp_path)
    big = layout_mascot(2000, 3000, "Сұрақ?", config=config)
    small = layout_mascot(1024, 1536, "Сұрақ?", config=config)
    assert big.mascot.width == 180
    assert small.mascot.width == 92
    assert big.mascot.width == round(2000 * 0.09)


def test_layout_is_inward_and_stays_inside_the_canvas(tmp_path: Path) -> None:
    config = _config(tmp_path)
    for corner, bubble_on_left in (
        ("top_left", False),
        ("bottom_left", False),
        ("top_right", True),
        ("bottom_right", True),
    ):
        layout = layout_mascot(1024, 1536, "Бұл сұрақ нені білдіреді?", corner, config)
        assert layout.corner == corner
        assert layout.block.left >= 0
        assert layout.block.top >= 0
        assert layout.block.right <= 1024
        assert layout.block.bottom <= 1536
        if bubble_on_left:
            assert layout.bubble.right <= layout.mascot.left
        else:
            assert layout.bubble.left >= layout.mascot.right
        assert layout.bubble.width <= round(1024 * config.bubble_max_width_ratio)


def test_layout_font_fits_real_rendered_lines_including_accents(tmp_path: Path) -> None:
    config = _config(tmp_path)
    question = "«Әке» дегеніміз не? Қазақстан туралы не білесің?"
    layout = layout_mascot(1024, 1536, question, "bottom_right", config)
    assert layout.font_size >= round(1024 * config.min_font_ratio)
    assert layout.font_size <= round(1024 * config.max_font_ratio)
    assert "".join(layout.lines).replace(" ", "") == question.replace(" ", "")

    from genlesson.mascot.layout import load_font

    font = load_font(layout.font_path, layout.font_size)
    max_line = max(measure_text(line, font)[0] for line in layout.lines)
    assert max_line <= layout.bubble.width - layout.padding[0] * 2
    assert layout.bubble.height >= layout.line_height * len(layout.lines)


def test_layout_returns_compatibility_aliases_and_dict(tmp_path: Path) -> None:
    layout = layout_mascot(1024, 1536, "Сұрақ?", config=_config(tmp_path))
    assert layout.mascot_rect == layout.mascot
    assert layout.bubble_rect == layout.bubble
    assert layout.block_rect == layout.block
    assert layout.combined == layout.block
    assert layout.text_lines == layout.lines
    payload = layout.as_dict()
    assert payload["font_size"] == layout.font_size
    assert payload["lines"] == list(layout.lines)
    assert layout["bubble"] == layout.bubble
    assert tuple(layout.mascot) == (
        layout.mascot.x,
        layout.mascot.y,
        layout.mascot.width,
        layout.mascot.height,
    )


def test_compose_overlay_draws_white_bubble_purple_outline_and_mascot(tmp_path: Path) -> None:
    config = _config(tmp_path)
    base = Image.new("RGB", (1024, 1536), (240, 240, 240))
    composition = compose_overlay(
        base,
        "Бұл суретте не көріп тұрсың?",
        config,
        variant="variant-0",
        corner="bottom_right",
        detector=NullTextDetector(),
    )
    assert composition.image.size == base.size
    bubble = composition.layout.bubble
    # The bubble interior is white and the border is purple.
    assert composition.image.getpixel((bubble.x + bubble.width // 2, bubble.y + 10)) == (
        255,
        255,
        255,
    )
    border_colors = {
        composition.image.getpixel((bubble.x, bubble.y + bubble.height // 2)),
        composition.image.getpixel((bubble.x + bubble.width // 2, bubble.y)),
    }
    assert (99, 96, 255) in border_colors
    # The mascot itself changed pixels near its center.
    mascot = composition.layout.mascot
    assert composition.image.getpixel(mascot.center) != (240, 240, 240)
    assert composition.metadata()["applied"] is True


def test_auto_corner_avoids_a_fake_text_box(tmp_path: Path) -> None:
    config = _config(tmp_path)
    base = Image.new("RGB", (1024, 1536), (255, 255, 255))
    draw = ImageDraw.Draw(base)
    # A red rectangle only in the bottom-right corner.  The fake detector treats
    # red pixels as text and returns the crop rectangle.
    draw.rectangle((800, 1200, 1000, 1500), fill=(255, 0, 0))

    class RedDetector:
        name = "fake-red"

        def detect(self, image: Image.Image):
            width, height = image.size
            if any(
                image.getpixel((x, y))[0] > 200
                and image.getpixel((x, y))[1] < 80
                and image.getpixel((x, y))[2] < 80
                for x, y in ((width // 2, height // 2),)
            ):
                from genlesson.mascot.layout import Rect

                return (Rect(0, 0, width, height),)
            return ()

    composition = compose_overlay(
        base,
        "Сұрақ?",
        config,
        variant="variant-0",
        corner="auto",
        detector=RedDetector(),
        preferred_corner="bottom_right",
    )
    assert composition.corner != "bottom_right"


def test_preview_is_a_blank_canvas_with_the_overlay(tmp_path: Path) -> None:
    config = _config(tmp_path)
    composition = render_preview(
        1024,
        1536,
        "Бұл сұрақ нені білдіреді?",
        config,
        variant="variant-1",
        corner="top_left",
        detector=NullTextDetector(),
    )
    assert composition.image.size == (1024, 1536)
    assert composition.variant == "variant-1"
    assert composition.corner == "top_left"


def test_stage5_records_mascot_plan_and_prompt(tmp_path: Path) -> None:
    _store, paths, _manifest, _spec = _generated_run(tmp_path, formats=("portrait",))
    bundle = read_model(paths.stage_dir("prompts") / "prompts.json", PromptBundle)
    item = bundle.formats[0]
    assert item.mascot is not None
    assert item.mascot.question
    assert item.mascot.question.endswith("?")
    assert item.mascot.variant
    assert item.mascot.font_size > 0
    assert min(item.mascot.mascot.width, item.mascot.mascot.height) == 92
    assert "MASCOT RESERVED AREA" in item.prompt
    assert "MASCOT RESERVED AREA" in item.prompt
    assert "% of the canvas width" in item.prompt
    assert "Do not draw a mascot" in item.prompt
    assert item.mascot.question_source == "planner"
    layout_doc = read_model(paths.stage_dir("prompts") / "mascot_layout.json", MascotLayoutDocument)
    assert layout_doc.formats[0].format == "portrait"
    assert layout_doc.formats[0].question_source == "planner"


def test_stage8_helper_composites_a_full_size_poster(tmp_path: Path, monkeypatch) -> None:
    from genlesson.mascot import zones

    zones._DETECTOR = NullTextDetector()
    _store, paths, _manifest, _spec = _generated_run(tmp_path, formats=("portrait",))
    layout_doc = read_model(paths.stage_dir("prompts") / "mascot_layout.json", MascotLayoutDocument)
    entry = layout_doc.formats[0]
    config = load_app_config().mascot
    base = Image.new("RGB", (1024, 1536), (250, 250, 250))
    placement = resolve_placement(
        base,
        entry.question,
        config=config,
        entry=entry,
        detector=NullTextDetector(),
    )
    assert isinstance(placement, Placement)
    buffer = io.BytesIO()
    base.save(buffer, format="PNG")

    png, metadata, warnings = _apply_mascot(
        buffer.getvalue(),
        width=1024,
        height=1536,
        fmt="portrait",
        config=config,
        entry=entry,
        placement=placement.as_dict(),
        mascot_reason=None,
        raw_image_path="raw.png",
    )

    assert metadata["applied"] is True
    assert metadata["question_source"] == "planner"
    assert metadata["preferred_corner"] == placement.preferred_corner
    assert metadata["final_corner"] == placement.corner
    assert metadata["font_size"] > 0
    assert metadata["raw_sha256"] != metadata["final_sha256"]
    assert metadata["self_check"]["checked"] is True
    assert metadata["rectangle"]["width"] > 0
    assert not warnings
    composed = Image.open(io.BytesIO(png))
    assert composed.size == (1024, 1536)
    zones._DETECTOR = None


def test_mascot_preview_cli_writes_png(tmp_path: Path) -> None:
    output = tmp_path / "preview.png"
    code = main(
        [
            "mascot-preview",
            "--size",
            "1024x1536",
            "--corner",
            "bottom_right",
            "--variant",
            "Vector",
            "--question",
            "Бұл суретте не көріп тұрсың?",
            "--out",
            str(output),
            "--detector",
            "none",
            "--json",
        ]
    )
    assert code == 0
    assert output.is_file()
    with Image.open(output) as image:
        assert image.size == (1024, 1536)


def test_mascot_preview_cli_rejects_unknown_variant(tmp_path: Path) -> None:
    code = main(
        [
            "mascot-preview",
            "--size",
            "1024x1536",
            "--variant",
            "does-not-exist",
            "--question",
            "Сұрақ?",
            "--out",
            str(tmp_path / "preview.png"),
            "--detector",
            "none",
        ]
    )
    assert code == 2
