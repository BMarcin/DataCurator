"""Pydantic models used as structured-output schemas for LLM stages.

A :class:`~DataCurator.pipeline.stages.llm.LLMStage` is generic over its
output type — point its ``response_model`` at any ``BaseModel`` here (or
anywhere importable). ``TranslationReview`` matches the output contract of
``improve-translations.j2``: an explicit evaluation of each
LanguageTool finding, an MQM-classified issue list, and three rewrites of
increasing freedom. ``QAPairSelection`` matches
``qa-pick-translations.j2``: the final selection and harmonization of
one translated question-answer pair from the candidates those reviews
produced. ``TranslationSelection`` matches ``pick-translation.j2``:
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


# --------------------------------------------------------------------------- #
# translate-simple.j2 — a dead-simple one-shot translate stage
# --------------------------------------------------------------------------- #
class SimpleTranslation(BaseModel):
    """Single-field schema for a dead-simple translate stage."""

    translation: str = Field(description="The full Polish translation of the source text.")


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

    Matches the output contract of ``qa-pick-translations.j2``: a
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

    Matches the output contract of ``pick-translation.j2``: a
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


# --------------------------------------------------------------------------- #
# Gated pipeline: gate-translation.j2 / fix-translation.j2 /
# pick-translation-slim.j2 — the slim schemas of the routed flow. The gate is
# the single MQM detector: it annotates every record with span-level issues
# (an empty list marks a clean draft at near-zero output cost) and derives its
# verdict from them; only ``needs_edit`` records reach the fix and pick
# stages, which therefore carry no issue lists or rationale of their own.
# --------------------------------------------------------------------------- #
GateVerdict = Literal["publishable", "needs_edit"]


class GateIssue(BaseModel):
    """A defect the gate found in the reviewed draft.

    ``severity`` decodes last on purpose: the grade follows the span and the
    explanation that justify it, and the explanation tells the fix stage what
    is actually wrong with the span.

    ``explanation.max_length`` renders into the request's JSON schema, so
    vLLM's guided decoding clamps the string at generation time — the excess
    tokens are never produced. The model never sees this schema; the gate
    prompts carry the matching "max 10 words" instruction so the clamp is a
    backstop, not the norm.
    """

    category: Category
    span_in_target: str = Field(description="The exact offending span in the target.")
    explanation: str = Field(
        max_length=240,
        description=(
            "10 words: what is wrong with the span and what a correct "
            "rendering should convey."
        ),
    )
    severity: Severity


class TranslationGate(BaseModel):
    """Span-level MQM annotation of a draft plus the routing verdict.

    ``issues`` precedes ``verdict`` on purpose: the verdict is decoded after
    the defects it is derived from ('needs_edit' iff any major/serious issue).
    """

    issues: List[GateIssue] = Field(default_factory=list)
    verdict: GateVerdict = Field(
        description="'needs_edit' iff any issue has severity major or serious."
    )


class TranslationGateRewrite(TranslationGate):
    """Gate annotation that also carries a minimal-edit fix for minor-only drafts.

    ``improved_translation`` is populated only when ``verdict == 'publishable'``
    and ``issues`` is non-empty (all minor); it is '' otherwise. The inherited
    ``issues``/``verdict`` fields decode first, so the fix is decoded last —
    after the defects it corrects and the verdict that gates it.
    """

    improved_translation: str = Field(
        default="",
        description=(
            "Minimal edit of the target correcting exactly the minor issues; "
            "'' when issues is empty or verdict is needs_edit."
        ),
    )


class GateIssueVerdict(BaseModel):
    """The fix stage's second opinion on one gate issue.

    The :class:`LanguageToolEvaluation` pattern applied to the gate: the
    post-editor re-judges each detected issue instead of trusting it, so a
    gate false positive is dismissed rather than "fixed". ``reason`` carries
    the same decode-enforced ``max_length`` as ``GateIssue.explanation``.
    """

    span_in_target: str = Field(description="The issue's span, echoed verbatim.")
    is_valid: bool = Field(description="True if the issue is a genuine defect.")
    reason: str = Field(
        max_length=120,
        description="10 words: why the issue is genuine or a false positive.",
    )


class TranslationFix(BaseModel):
    """Two rewrites of a draft the gate flagged; the gate's issues drive them.

    ``issue_verdicts`` decodes first on purpose: the post-editor re-judges
    every gate issue (one verdict per issue, in the listed order) before
    ``approach`` commits to a fix plan covering only the valid ones, and the
    rewrites that plan drives decode last. When every issue is dismissed,
    ``improved_primary`` is the unchanged target.
    """

    issue_verdicts: List[GateIssueVerdict] = Field(
        default_factory=list,
        description="One verdict per detected issue, in the listed order.",
    )
    approach: str = Field(
        description=(
            "Brief plan: which valid issues to fix, how, and what must "
            "stay untouched. Keep it really short."
        )
    )
    improved_primary: str = Field(
        description="Minimal edit correcting exactly the valid detected issues."
    )
    improved_polished: str = Field(description="Fluent, natural rewrite.")


class TranslationPick(BaseModel):
    """Selection of the best of the two rewrites of one segment."""

    ranking: List[str] = Field(
        default_factory=list, description="Every candidate id exactly once, best first."
    )
    selected_id: str = Field(description="Id of the winning candidate; equals ranking[0].")
    selected_translation: str = Field(description="The winning candidate, verbatim.")


#: The slim pick always ranks exactly two candidates, C1 and C2. A future
#: 3-candidate variant must define its own id type rather than extend this.
PickId = Literal["C1", "C2"]


class TranslationPickById(BaseModel):
    """Selection of the best of two candidates by id — no reprinted text.

    ``selected_translation`` is intentionally absent: a
    :class:`~DataCurator.pipeline.modifiers.select_by_id.SelectByIdModifier`
    resolves ``selected_id`` back to the candidate text, so the model never
    reprints the whole translation (cheaper, and verbatim by construction).
    """

    ranking: List[PickId] = Field(
        default_factory=list, description="Both candidate ids, best first."
    )
    selected_id: PickId = Field(description="Id of the winning candidate; equals ranking[0].")


# --------------------------------------------------------------------------- #
# adjust-pair-terminology-batch.j2 — harmonize the terms a query shares with
# its positive passage(s), after both were translated independently. The model
# returns span edits only (never a reprinted text); EditSpanValidator grounds
# and bounds them, ApplyEditsModifier applies them.
# --------------------------------------------------------------------------- #
AdjustCategory = Literal["terminology", "named_entity", "title", "format"]


class TermEdit(BaseModel):
    """One span replacement in one segment of a query/positive group.

    ``source_term`` decodes before ``span`` on purpose: the model must first
    name the English term both sources share — the edit's grounding — and only
    then the Polish span that renders it.
    """

    segment_id: str = Field(description="Segment edited: 'Q' or a positive id like 'P1'.")
    source_term: str = Field(
        description="The English term, verbatim, shared by the query's and the passage's source."
    )
    span: str = Field(description="The exact span replaced in that segment's translation.")
    replacement: str = Field(description="The replacement for every occurrence of span.")
    category: AdjustCategory
    reason: str = Field(
        max_length=120,
        description="10 words: which rendering differs and why this one is kept.",
    )


class PairAdjustment(BaseModel):
    """Terminology harmonization of a query and its positives; empty = consistent."""

    edits: List[TermEdit] = Field(default_factory=list)
