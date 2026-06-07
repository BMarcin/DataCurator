"""Pydantic models used as structured-output schemas for LLM stages.

A :class:`~DataCurator.pipeline.stages.llm.LLMStage` is generic over its
output type — point its ``response_model`` at any ``BaseModel`` here (or
anywhere importable). ``TranslationReview`` matches the output contract of
``prompts/improve-translations.j2``: an explicit evaluation of each
LanguageTool finding, an MQM-classified issue list, and three rewrites of
increasing freedom. ``QAPairSelection`` matches
``prompts/qa-pick-translations.j2``: the final selection and harmonization of
one translated question-answer pair from the candidates those reviews
produced. ``TranslationSelection`` matches ``prompts/pick-translation.j2``:
the single-segment analogue — rank three candidate translations of ONE source
segment and return the best one verbatim (no harmonization, no pairing).
"""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

Category = Literal[
    "terminology",
    "accuracy",
    "linguistic_conventions",
    "style",
    "locale_conventions",
    "audience_appropriateness",
]
Severity = Literal["minor", "major", "serious"]


class LanguageToolEvaluation(BaseModel):
    """The model's verdict on a single LanguageTool finding."""

    finding_message: str = Field(description="The LanguageTool finding being evaluated.")
    is_real_issue: bool = Field(description="True if the finding is a genuine error.")
    reason: str = Field(description="Brief justification for the verdict.")


class TranslationIssue(BaseModel):
    """A concrete flaw found in the reviewed target translation."""

    category: Category
    severity: Severity
    span_in_target: str = Field(description="The exact offending span in the target.")
    explanation: str = Field(description="Why this span is an error.")
    suggested_fix: str = Field(description="The replacement for the span.")


class TranslationReview(BaseModel):
    """Structured review of a translation: evaluation, issues and rewrites."""

    overall_quality: str = Field(description="Overall quality verdict for the target.")
    languagetool_evaluations: List[LanguageToolEvaluation] = Field(default_factory=list)
    issues: List[TranslationIssue] = Field(default_factory=list)
    improved_primary: str = Field(description="Conservative, minimal-edit correction.")
    improved_polished: str = Field(description="Fluent, natural rewrite.")
    improved_alternative: str = Field(description="A meaningfully different rewrite.")


# --------------------------------------------------------------------------- #
# qa-pick-translations.j2 — final selection/harmonization of a Q&A pair
# --------------------------------------------------------------------------- #
Quality = Literal["good", "acceptable", "poor"]

# The selection stage judges a whole pair, so it can raise two MQM categories
# the per-segment review cannot: ``consistency`` (the chosen question and
# answer disagree with themselves) and ``relevance`` (the translated answer no
# longer answers the translated question).
PickCategory = Literal[
    "terminology",
    "accuracy",
    "linguistic_conventions",
    "style",
    "locale_conventions",
    "audience_appropriateness",
    "consistency",
    "relevance",
]
Segment = Literal["question", "answer"]
# Harmonization only ever aligns the pair or fixes a form, never re-translates.
HarmonizationCategory = Literal["consistency", "locale_conventions", "linguistic_conventions"]


class CandidateIssue(BaseModel):
    """A flaw found in one candidate, judged against its source segment."""

    category: PickCategory
    severity: Severity
    span: str = Field(description="The offending span in this candidate.")
    explanation: str = Field(description="Why this span is an error.")


class CandidateEvaluation(BaseModel):
    """The verdict on a single question or answer candidate."""

    id: str = Field(description="Candidate id, e.g. 'Q1' or 'A2'.")
    quality: Quality
    issues: List[CandidateIssue] = Field(default_factory=list)
    key_terms: List[str] = Field(
        default_factory=list, description="The main content terms this candidate uses."
    )


class SharedConcept(BaseModel):
    """How one source concept is rendered in each half of the selected pair."""

    concept: str = Field(description="The source concept, e.g. 'oven'.")
    question_form: str = Field(description="Term used in the selected question.")
    answer_form: str = Field(description="Term used in the selected answer.")
    consistent: bool


class ConsistencyCheck(BaseModel):
    """Whether the selected pair holds together as one coherent Q&A item."""

    shared_concepts: List[SharedConcept] = Field(default_factory=list)
    register_match: bool
    relevance_ok: bool
    relevance_note: str = Field(
        description="Confirm the answer addresses the question, or describe the break."
    )


class HarmonizationEdit(BaseModel):
    """A minimal edit aligning the selected pair's terminology/register/form."""

    segment: Segment
    span: str = Field(description="The exact text replaced in that segment.")
    replaced_with: str = Field(description="The replacement.")
    category: HarmonizationCategory
    reason: str = Field(description="Why the edit was made.")


class QAPairSelection(BaseModel):
    """Final selection and harmonization of a translated Q&A pair.

    Matches the output contract of ``prompts/qa-pick-translations.j2``: a
    per-candidate evaluation of each question and answer, the chosen
    question/answer ids and rationale, a consistency check over the chosen
    pair, the harmonization edits applied to it, and the resulting final pair.
    """

    question_candidate_evaluations: List[CandidateEvaluation] = Field(default_factory=list)
    answer_candidate_evaluations: List[CandidateEvaluation] = Field(default_factory=list)
    selected_question_id: str = Field(description="Id of the chosen question candidate.")
    selected_answer_id: str = Field(description="Id of the chosen answer candidate.")
    selection_rationale: str = Field(description="Why THIS combination was chosen.")
    consistency_check: ConsistencyCheck
    harmonization_edits: List[HarmonizationEdit] = Field(default_factory=list)
    final_question: str = Field(description="The selected question after harmonization edits.")
    final_answer: str = Field(description="The selected answer after harmonization edits.")
    overall_quality: Quality


# --------------------------------------------------------------------------- #
# pick-translation.j2 — select the best of three single-segment translations
# --------------------------------------------------------------------------- #
# Single-segment selection judges one source segment, so it raises only the
# six base MQM categories (``Category``); it has no second segment to be
# inconsistent or irrelevant with, so ``consistency``/``relevance`` do not apply.


class SegmentCandidateIssue(BaseModel):
    """A flaw found in one candidate, judged against the source segment."""

    category: Category
    severity: Severity
    span: str = Field(description="The offending span in this candidate.")
    explanation: str = Field(description="Why this span is an error.")


class SegmentCandidateEvaluation(BaseModel):
    """The verdict on a single candidate translation."""

    id: str = Field(description="Candidate id, e.g. 'C1'.")
    quality: Quality
    issues: List[SegmentCandidateIssue] = Field(default_factory=list)


class TranslationSelection(BaseModel):
    """Selection of the best of three candidate translations of one segment.

    Matches the output contract of ``prompts/pick-translation.j2``: a
    per-candidate evaluation, a full ranking, the chosen candidate's id and
    rationale, and the winning translation returned verbatim.
    """

    candidate_evaluations: List[SegmentCandidateEvaluation] = Field(default_factory=list)
    ranking: List[str] = Field(
        default_factory=list, description="Every candidate id exactly once, best first."
    )
    selected_id: str = Field(description="Id of the winning candidate; equals ranking[0].")
    selection_rationale: str = Field(description="Why this candidate beats the other two.")
    selected_translation: str = Field(description="The winning candidate, verbatim.")
