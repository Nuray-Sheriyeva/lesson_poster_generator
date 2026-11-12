from __future__ import annotations

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langfuse")

from experiments import run as experiment_run


class _Dataset:
    def __init__(self, name: str, items=None) -> None:
        self.name = name
        self.items = list(items or [])


class _FakeDatasetClient:
    def __init__(self) -> None:
        self.datasets: dict[str, _Dataset] = {}
        self.items: list = []

    def get_dataset(self, name: str):
        if name not in self.datasets:
            raise KeyError(name)
        return self.datasets[name]

    def create_dataset(self, *, name: str, description: str = "", **kwargs):
        self.datasets[name] = _Dataset(name)
        return self.datasets[name]

    def create_dataset_item(self, *, dataset_name: str, **kwargs):
        item = _Item(kwargs["input"], kwargs.get("expected_output"), kwargs.get("metadata"))
        self.datasets[dataset_name].items.append(item)
        return item


class _Item:
    def __init__(self, input, expected_output, metadata) -> None:
        self.input = input
        self.expected_output = expected_output
        self.metadata = metadata


def test_dataset_has_fixtures_and_one_golden_item() -> None:
    client = _FakeDatasetClient()

    dataset = experiment_run._dataset_items(client, "fixtures")

    assert len(dataset.items) >= 5
    golden = [item for item in dataset.items if item.input["filename"] == "relationtype.pdf"]
    assert len(golden) == 1
    assert golden[0].expected_output == experiment_run.GOLDEN
    assert all(
        item.expected_output is None
        for item in dataset.items
        if item.input["filename"] != "relationtype.pdf"
    )


def test_golden_evaluator_scores_relationtype() -> None:
    output = {
        "status": "ok",
        "language": "kk",
        "details": 7,
        "file_slug": "biotic-interactions",
        "conflicts_present": True,
    }

    evaluations = experiment_run._evaluate(
        input={"filename": "relationtype.pdf"},
        output=output,
        expected_output=experiment_run.GOLDEN,
        metadata={"golden": True},
    )

    assert {item.name for item in evaluations} == {
        "language_match",
        "details_match",
        "slug_match",
        "conflicts_present",
    }
    assert all(item.value == 1.0 for item in evaluations)


def test_selected_items_can_be_filtered() -> None:
    items = [
        _Item({"filename": "relationtype.pdf", "stem": "relationtype"}, None, None),
        _Item({"filename": "government.pdf", "stem": "government"}, None, None),
    ]

    selected = experiment_run._selected_items(items, ["relationtype"])

    assert [item.input["filename"] for item in selected] == ["relationtype.pdf"]


def test_plan_graph_contains_only_the_planning_stages() -> None:
    topology = experiment_run._plan_graph().get_graph()
    assert {name for name in topology.nodes if not name.startswith("__")} == {
        "ingest",
        "classify",
        "plan",
    }
    assert ("__start__", "ingest") in {
        (edge.source, edge.target) for edge in topology.edges
    }
    assert ("plan", "__end__") in {
        (edge.source, edge.target) for edge in topology.edges
    }


def test_fixture_directory_contains_the_original_five_lessons() -> None:
    names = {path.name for path in experiment_run._fixture_files()}
    assert {
        "alashorda.pdf",
        "government.pdf",
        "relationtype.pdf",
        "russian_sample.docx",
        "scanned_sample.pdf",
    } <= names
    assert len(names) >= 5
