"""Planner provider selection."""

from __future__ import annotations

from ..config import PlannerOptions
from ..errors import GenLessonError
from .base import (
    CandidateCentralIdea,
    CandidateDetail,
    CandidateRequest,
    CandidateSet,
    ClassificationDraft,
    ClassificationRequest,
    LanguageDecision,
    LanguageRequest,
    PageImage,
    PlanRequest,
    Planner,
    PlannerCapabilities,
    PosterPlanner,
    PosterSpecDraft,
    SegmentDecision,
    SegmentInput,
    TopicDraft,
)
from .anthropic_planner import AnthropicPlanner
from .openai_planner import DeepSeekPlanner, OpenAIPlanner

__all__ = [
    "AnthropicPlanner",
    "CandidateCentralIdea",
    "CandidateDetail",
    "CandidateRequest",
    "CandidateSet",
    "ClassificationDraft",
    "ClassificationRequest",
    "DeepSeekPlanner",
    "LanguageDecision",
    "LanguageRequest",
    "OpenAIPlanner",
    "PageImage",
    "PlanRequest",
    "Planner",
    "PlannerCapabilities",
    "PosterPlanner",
    "PosterSpecDraft",
    "SegmentDecision",
    "SegmentInput",
    "TopicDraft",
    "build_planner",
]


def build_planner(options: PlannerOptions) -> Planner:
    if options.provider == "openai":
        if not options.api_key:
            raise GenLessonError(
                "planner_required",
                "Stage 2 needs an OpenAI API key for the Luna planner.",
                hint="Set OPENAI_API_KEY and run classify again.",
            )
        return OpenAIPlanner(
            model=options.model,
            api_key=options.api_key,
            base_url=options.base_url,
            timeout_s=options.timeout_s,
            max_retries=options.max_retries,
            max_context=options.max_context,
        )
    if options.provider == "anthropic":
        if not options.api_key:
            raise GenLessonError(
                "planner_required",
                "Stage 2 needs an Anthropic API key for the planner.",
                hint="Set ANTHROPIC_API_KEY and run classify again.",
            )
        return AnthropicPlanner(
            model=options.model,
            api_key=options.api_key,
            base_url=options.base_url,
            timeout_s=options.timeout_s,
            max_retries=options.max_retries,
            max_context=options.max_context,
        )
    if options.provider == "deepseek":
        if not options.api_key:
            raise GenLessonError(
                "planner_required",
                "Stage 2 needs a DeepSeek API key for the planner.",
                hint="Set DEEPSEEK_API_KEY and run classify again.",
            )
        return DeepSeekPlanner(
            model=options.model,
            api_key=options.api_key,
            base_url=options.base_url,
            timeout_s=options.timeout_s,
            max_retries=options.max_retries,
            max_context=options.max_context,
        )
    raise GenLessonError(
        "planner_unavailable",
        f"Unsupported planner provider: {options.provider}",
    )
