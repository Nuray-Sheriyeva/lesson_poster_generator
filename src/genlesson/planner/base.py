"""Planner boundary and structured drafts for Stage 2.

The classifier and language detector are separate model calls on purpose:
language is decided only from the segments that survived classification, never
from the whole document.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..models import (
    AgeBand,
    CentralIdea,
    Composition,
    PosterDetail,
    SegmentCategory,
    SegmentLabel,
    SourceConflict,
    StrictModel,
    Style,
    TextSource,
)


class SegmentDecision(StrictModel):
    id: str
    label: SegmentLabel
    category: SegmentCategory = "other"
    reason: str = ""


class TopicDraft(StrictModel):
    title: str
    segment_ids: list[str]


class ClassificationDraft(StrictModel):
    segments: list[SegmentDecision]
    topics: list[TopicDraft] = []


class LanguageDecision(StrictModel):
    poster_language: str
    confidence: float
    evidence: str


class CandidateCentralIdea(StrictModel):
    title: str
    definition: str


class CandidateDetail(StrictModel):
    label: str
    explanation: str
    example: str | None
    visual: str
    source_page: int


class CandidateSet(StrictModel):
    central_ideas: list[CandidateCentralIdea]
    details: list[CandidateDetail]
    key_terms: list[str]
    source_conflicts: list[SourceConflict]


class PosterSpecDraft(StrictModel):
    language: str
    age_band: AgeBand
    central_idea: CentralIdea
    details: list[PosterDetail]
    composition: Composition
    style: Style
    strings_to_render: list[str]
    mascot_question: str | None = None
    source_conflicts: list[SourceConflict]
    file_slug: str


@dataclass(frozen=True)
class SegmentInput:
    id: str
    page: int
    page_label: str
    order: int
    text: str
    text_source: TextSource


@dataclass(frozen=True)
class PageImage:
    page: int
    path: Path


@dataclass(frozen=True)
class ClassificationRequest:
    title: str | None
    segments: tuple[SegmentInput, ...]
    images: tuple[PageImage, ...] = ()
    scanned_document: bool = False


@dataclass(frozen=True)
class LanguageRequest:
    text: str
    title: str | None = None


@dataclass(frozen=True)
class PlannerCapabilities:
    structured_output: bool
    vision: bool
    max_context: int


@runtime_checkable
class Planner(Protocol):
    model: str
    capabilities: PlannerCapabilities

    def classify(self, request: ClassificationRequest) -> ClassificationDraft:
        """Split segments into lesson content and non-content."""

    def detect_language(self, request: LanguageRequest) -> LanguageDecision:
        """Return a BCP-47 language tag for already-filtered body text."""


@dataclass(frozen=True)
class PlanRequest:
    language: str
    title: str | None
    dominant_topic: str | None
    max_details: int
    text: str | None = None
    candidates: CandidateSet | None = None
    images: tuple[PageImage, ...] = ()


@dataclass(frozen=True)
class CandidateRequest:
    language: str
    title: str | None
    dominant_topic: str | None
    text: str
    chunk_index: int
    chunk_count: int
    source_pages: tuple[int, ...]
    max_candidates: int


@runtime_checkable
class PosterPlanner(Protocol):
    model: str
    capabilities: PlannerCapabilities

    def plan(self, request: PlanRequest) -> PosterSpecDraft:
        """Return a complete poster spec, optionally from chunk candidates."""

    def extract_candidates(self, request: CandidateRequest) -> CandidateSet:
        """Extract candidate ideas from a chunk that is too large for one call."""
