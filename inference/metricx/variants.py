"""MetricX model variants: input templates and per-variant defaults.

Torch-free on purpose so the tests can pin the exact template strings without
the heavy MetricX stack. The templates reproduce upstream ``_make_input`` in
``metricx23/predict.py`` and ``metricx24/predict.py`` character for character.

``metricx-23-qe`` is reference-free (source + hypothesis). ``metricx-24-hybrid``
takes source + hypothesis + reference, and falls back to the QE template when
the reference is absent/empty — matching upstream, where ``--qe`` ignores the
reference entirely.
"""
from __future__ import annotations

from typing import Optional

METRICX_23_QE = "metricx-23-qe"
METRICX_24_HYBRID = "metricx-24-hybrid"
VARIANTS = (METRICX_23_QE, METRICX_24_HYBRID)

# The models were trained with these maximum input lengths (vendor README);
# exceeding them "may lead to unpredictable behavior".
_DEFAULT_MAX_INPUT_LENGTH = {METRICX_23_QE: 1024, METRICX_24_HYBRID: 1536}


def check_variant(variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"unknown MetricX variant {variant!r}; use one of {list(VARIANTS)}")
    return variant


def detect_variant(model_name: str) -> str:
    """Infer the variant from the checkpoint name (e.g. google/metricx-24-hybrid-xl-v2p6)."""
    return METRICX_24_HYBRID if "metricx-24" in model_name else METRICX_23_QE


def accepts_reference(variant: str) -> bool:
    return check_variant(variant) == METRICX_24_HYBRID


def default_max_input_length(variant: str) -> int:
    return _DEFAULT_MAX_INPUT_LENGTH[check_variant(variant)]


def make_input(
    variant: str,
    source: str,
    hypothesis: str,
    reference: Optional[str] = None,
) -> str:
    """Build the model input string; an absent/empty reference means QE mode."""
    check_variant(variant)
    if variant == METRICX_23_QE:
        if reference:
            raise ValueError("metricx-23-qe is reference-free; 'reference' is not supported")
        return "candidate: " + hypothesis + " source: " + source
    if reference:
        return "source: " + source + " candidate: " + hypothesis + " reference: " + reference
    return "source: " + source + " candidate: " + hypothesis
