"""Concrete :class:`~DataCurator.pipeline.stage.StageModifier` implementations."""
from __future__ import annotations

from DataCurator.pipeline.modifiers.candidates import CandidatesModifier
from DataCurator.pipeline.modifiers.constants import ConstantsModifier
from DataCurator.pipeline.modifiers.google_translate import GoogleTranslateModifier
from DataCurator.pipeline.modifiers.languagetool import LanguageToolModifier
from DataCurator.pipeline.modifiers.rename import RenameModifier

__all__ = [
    "CandidatesModifier",
    "ConstantsModifier",
    "GoogleTranslateModifier",
    "LanguageToolModifier",
    "RenameModifier",
]
