"""Concrete :class:`~DataCurator.pipeline.stage.StageModifier` implementations."""
from __future__ import annotations

from DataCurator.pipeline.modifiers.candidates import CandidatesModifier
from DataCurator.pipeline.modifiers.conditional import SetWhenModifier
from DataCurator.pipeline.modifiers.constants import ConstantsModifier
from DataCurator.pipeline.modifiers.edits import ApplyEditsModifier
from DataCurator.pipeline.modifiers.google_translate import GoogleTranslateModifier
from DataCurator.pipeline.modifiers.languagetool import LanguageToolModifier
from DataCurator.pipeline.modifiers.lists import (
    ListDedupModifier,
    ListLimitModifier,
    ListPickModifier,
)
from DataCurator.pipeline.modifiers.metricx import MetricXScoreModifier
from DataCurator.pipeline.modifiers.rename import RenameModifier
from DataCurator.pipeline.modifiers.select_by_id import SelectByIdModifier
from DataCurator.pipeline.modifiers.transfer import TransferCheckModifier
from DataCurator.pipeline.modifiers.verdict import DeriveVerdictModifier

__all__ = [
    "ApplyEditsModifier",
    "CandidatesModifier",
    "ConstantsModifier",
    "DeriveVerdictModifier",
    "GoogleTranslateModifier",
    "LanguageToolModifier",
    "ListDedupModifier",
    "ListLimitModifier",
    "ListPickModifier",
    "MetricXScoreModifier",
    "RenameModifier",
    "SelectByIdModifier",
    "SetWhenModifier",
    "TransferCheckModifier",
]
