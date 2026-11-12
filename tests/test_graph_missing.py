from __future__ import annotations

import json
import sys
from pathlib import Path

from genlesson.cli import main

from .helpers import make_lesson_docx


def test_graph_missing_dependency_is_clean(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    docx = make_lesson_docx(tmp_path / "lesson.docx", ["A short lesson."])
    monkeypatch.setitem(sys.modules, "langgraph", None)
    monkeypatch.setitem(sys.modules, "langgraph.graph", None)
    monkeypatch.delitem(sys.modules, "genlesson.graph", raising=False)

    code = main(
        [
            "run",
            str(docx),
            "--run-id",
            "no-graph",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    payload = json.loads(captured.out)
    assert payload["error"]["code"] == "graph_unavailable"
