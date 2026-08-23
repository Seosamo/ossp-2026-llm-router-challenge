"""Reverse test suite for validate.py itself (§8.1's "역테스트"): proves each
check passes against a well-behaved pipeline AND fails against pipelines that
violate the specific invariant that check is supposed to police. A validator that
always passes would be worse than no validator at all.
"""

from __future__ import annotations

from router.validate import (
    check_batch_invariance,
    check_challenge_split_dependence,
    check_episode_id_dependence,
    check_perturbation_sensitivity,
    check_repeatability,
    check_schema,
)


def test_schema_passes_on_valid_data(dev_df):
    result = check_schema(dev_df)
    assert result.passed, result.details


def test_perturbation_sensitivity_passes(trained_pipeline, dev_df):
    sample_texts = dev_df["text"].head(5).tolist()
    result = check_perturbation_sensitivity(trained_pipeline, sample_texts)
    assert result.passed, result.details


def test_episode_id_alignment_passes(trained_pipeline, dev_df):
    result = check_episode_id_dependence(trained_pipeline, dev_df)
    assert result.passed, result.details


def test_metadata_isolation_passes(trained_pipeline, dev_df):
    result = check_challenge_split_dependence(trained_pipeline, dev_df)
    assert result.passed, result.details


def test_batch_invariance_passes_on_well_behaved_pipeline(trained_pipeline, dev_df):
    result = check_batch_invariance(trained_pipeline, dev_df)
    assert result.passed, result.details


def test_repeatability_passes(trained_pipeline, dev_df):
    result = check_repeatability(trained_pipeline, dev_df)
    assert result.passed, result.details


def test_batch_invariance_catches_batch_statistic_leakage(batch_leaking_pipeline, dev_df):
    result = check_batch_invariance(batch_leaking_pipeline, dev_df)
    assert not result.passed, "validator failed to catch a pipeline that leaks batch statistics into decisions"


def test_batch_invariance_catches_refit_on_inference(refitting_pipeline, dev_df):
    result = check_batch_invariance(refitting_pipeline, dev_df)
    assert not result.passed, "validator failed to catch a pipeline that refits its feature extractor per call"
