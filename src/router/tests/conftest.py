"""Shared fixtures: a synthetic dataset, a small well-behaved trained pipeline, and
two intentionally-broken pipeline fixtures used only in test_validator.py to prove
validate.py's checks actually discriminate rather than trivially passing.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from router import config as router_config
from router.decision import decide
from router.pipeline import RouterPipeline
from router.schema import join_prompts
from router.synthetic_data import generate_synthetic_dataset

# The embedding branch needs a real (possibly network-fetched) sentence-transformers
# model, which is unsuitable for a fast/offline test loop and orthogonal to what
# these tests are checking (§10.1 governs whether it's even usable at all). Tests
# run against a TF-IDF-only config, exercising the same fallback path the ARM gate
# would trigger on failure.
@pytest.fixture(scope="session")
def test_cfg():
    cfg = types.SimpleNamespace(**vars(router_config))
    cfg.USE_EMBEDDING_BRANCH = False
    return cfg


@pytest.fixture(scope="session")
def synthetic_data():
    return generate_synthetic_dataset(n_queries=80, seed=7)


@pytest.fixture(scope="session")
def trained_pipeline(test_cfg, synthetic_data):
    pipeline = RouterPipeline(cfg=test_cfg)
    pipeline.fit(synthetic_data["train"], synthetic_data["prompts"])
    return pipeline


@pytest.fixture(scope="session")
def dev_df(synthetic_data):
    return join_prompts(synthetic_data["dev"], synthetic_data["prompts"])


class BatchLeakingPipeline(RouterPipeline):
    """Violates the B4 invariant on purpose: predict_batch perturbs each row's
    p_hat using a batch-wide statistic (the batch mean of ax31-light's p_hat)
    before deciding, so its output differs from calling predict() row-by-row.
    """

    def predict_batch(self, df: pd.DataFrame, tier: str) -> pd.DataFrame:
        from router.config import K_M, TOKEN_ACCOUNTING, get_lambda
        from router.models.input_tokens import build_simple_features

        rows = []
        for _, row in df.iterrows():
            rows.append(self._predict_row(row["text"], tier))

        batch_mean_light = float(np.mean([r["p_hat"]["ax31-light"] for r in rows]))
        lam = get_lambda(tier, TOKEN_ACCOUNTING)

        records = []
        for (_, row), result in zip(df.iterrows(), rows):
            leaked_p_hat = dict(result["p_hat"])
            leaked_p_hat["ax31-light"] *= 1.0 + 0.5 * (batch_mean_light - 0.5)
            decision = decide(leaked_p_hat, result["cost_hat"], lam)
            records.append({"episode_id": row.get("episode_id"), "decision": decision})
        return pd.DataFrame.from_records(records)


class RefittingFeatureExtractorWrapper:
    """Violates the B4 invariant differently: refits the lexical SVD on whatever
    batch of texts is passed to transform(), instead of using the frozen,
    Train-fit vectorizer. This is exactly the §8.1 B4 failure mode ("배치 통계
    의존" / calling fit_transform at inference time) described in §5.1 note 4.
    """

    def __init__(self, base_extractor):
        self._base = base_extractor

    def __getattr__(self, name):
        return getattr(self._base, name)

    def transform(self, texts):
        # Refit on the current call's texts before transforming -- the bug under test.
        import copy

        refit = copy.deepcopy(self._base)
        refit.fit(texts)
        return refit.transform(texts)


@pytest.fixture(scope="session")
def batch_leaking_pipeline(test_cfg, synthetic_data):
    pipeline = BatchLeakingPipeline(cfg=test_cfg)
    pipeline.fit(synthetic_data["train"], synthetic_data["prompts"])
    return pipeline


@pytest.fixture(scope="session")
def refitting_pipeline(test_cfg, synthetic_data):
    pipeline = RouterPipeline(cfg=test_cfg)
    pipeline.fit(synthetic_data["train"], synthetic_data["prompts"])
    pipeline.feature_extractor = RefittingFeatureExtractorWrapper(pipeline.feature_extractor)
    return pipeline
