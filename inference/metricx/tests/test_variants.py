"""Template tests for the variant module.

The expected strings are pinned character for character against upstream
``_make_input`` in ``metricx23/predict.py`` and ``metricx24/predict.py`` —
any drift here changes the model's scores silently.
"""
import pytest

import variants


def test_metricx_23_qe_template():
    assert (
        variants.make_input("metricx-23-qe", "src", "hyp")
        == "candidate: hyp source: src"
    )


def test_metricx_23_qe_rejects_reference():
    with pytest.raises(ValueError):
        variants.make_input("metricx-23-qe", "src", "hyp", "ref")


def test_metricx_24_hybrid_with_reference():
    assert (
        variants.make_input("metricx-24-hybrid", "src", "hyp", "ref")
        == "source: src candidate: hyp reference: ref"
    )


def test_metricx_24_hybrid_without_reference_is_qe():
    qe = "source: src candidate: hyp"
    assert variants.make_input("metricx-24-hybrid", "src", "hyp") == qe
    # empty reference means QE too, matching upstream --qe (reference ignored)
    assert variants.make_input("metricx-24-hybrid", "src", "hyp", "") == qe


def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        variants.make_input("metricx-99", "src", "hyp")


def test_detect_variant_from_model_name():
    assert variants.detect_variant("google/metricx-23-qe-xl-v2p0") == "metricx-23-qe"
    assert variants.detect_variant("google/metricx-24-hybrid-xl-v2p6") == "metricx-24-hybrid"


def test_default_max_input_length_follows_training():
    assert variants.default_max_input_length("metricx-23-qe") == 1024
    assert variants.default_max_input_length("metricx-24-hybrid") == 1536
