"""Concrete :class:`~DataCurator.pipeline.stage.Stage` implementations."""
from __future__ import annotations

from DataCurator.pipeline.stages.coalesce import CoalesceStage
from DataCurator.pipeline.stages.google_translate import GoogleTranslateStage
from DataCurator.pipeline.stages.llm import LLMStage

__all__ = ["CoalesceStage", "GoogleTranslateStage", "LLMStage"]
