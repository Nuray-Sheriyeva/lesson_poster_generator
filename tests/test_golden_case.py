from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from genlesson.models import PosterSpecDocument
from genlesson.prompts import build_prompt

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
INPUT = FIXTURES / "input" / "relationtype.pdf"
REFERENCE = FIXTURES / "expected" / "relationtype_reference.png"
SPEC = FIXTURES / "expected" / "relationtype_approved_spec.json"
META = FIXTURES / "expected" / "relationtype_golden.json"


def _require() -> None:
    missing = [str(path) for path in (INPUT, REFERENCE, SPEC, META) if not path.is_file()]
    if missing:
        pytest.skip(f"golden fixtures missing: {missing}")


def test_relationtype_input_and_reference_assets() -> None:
    _require()
    document = pymupdf.open(INPUT)
    try:
        assert document.page_count == 33
        text = "\n".join(page.get_text() for page in document)
    finally:
        document.close()
    assert "Тірі организмдер" in text
    assert "Жыртқыштық" in text

    reference = pymupdf.Pixmap(REFERENCE)
    assert (reference.width, reference.height) == (1536, 1024)

    meta = json.loads(META.read_text(encoding="utf-8"))
    assert meta["language"] == "kk"
    assert meta["format"] == "landscape"
    assert meta["reference_size"] == [1536, 1024]


def test_relationtype_golden_spec_and_prompt() -> None:
    _require()
    spec = PosterSpecDocument.model_validate_json(SPEC.read_text(encoding="utf-8"))
    assert spec.language == "kk"
    assert spec.file_slug == "biotic-interactions"
    assert len(spec.details) == 7
    assert spec.source_conflicts

    prompt = build_prompt(spec, "landscape", width=1536, height=1024)
    assert "1536 x 1024" in prompt
    assert "Never render 1., 2., 3." in prompt
    assert "no blood, gore, wounds" in prompt
    assert all(value in prompt for value in spec.strings_to_render)
    assert all(letter in prompt for letter in "әғқңөұүһі")
