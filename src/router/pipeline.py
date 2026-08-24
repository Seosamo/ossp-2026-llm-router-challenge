"""End-to-end orchestrator (§4): ties feature extraction, the three model
families, and the decision rule together behind one fit/predict interface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

from router import config as default_config
from router.config import K_M, MODELS, TOKEN_ACCOUNTING, get_lambda
from router.decision import decide, estimate_total_cost
from router.features.extractor import FeatureExtractor
from router.models.input_tokens import InputTokenLinearModel, build_simple_features, train_input_token_models
from router.models.output_tokens import OutputTokenQuantileModel, train_output_token_models
from router.models.win_probability import WinProbabilityModel, train_all_win_probability_models

if TYPE_CHECKING:
    # pandas is a training/offline-only dependency (see the module docstring
    # in router/models/*.py) -- NOT imported at module level here, so that
    # inference-only callers (e.g. ossp_router.learned_router) can import
    # RouterPipeline without pandas installed at all (it is deliberately left
    # out of container/requirements-runtime.txt). This import only runs for
    # type checkers, never at runtime, thanks to `from __future__ import
    # annotations` above.
    import pandas as pd


class RouterPipeline:
    def __init__(self, cfg=default_config):
        self._cfg = cfg
        self.feature_extractor = FeatureExtractor(cfg=cfg)
        self.win_probability_models: Dict[str, WinProbabilityModel] = {}
        self.output_token_models: Dict[str, Dict[str, OutputTokenQuantileModel]] = {}
        self.input_token_models: Dict[str, InputTokenLinearModel] = {}
        self._fitted = False

    def fit(self, train_df: pd.DataFrame, prompts_df: pd.DataFrame) -> "RouterPipeline":
        from router.schema import join_prompts

        joined = join_prompts(train_df, prompts_df)
        texts: List[str] = joined["text"].tolist()

        self.feature_extractor.fit(texts)
        features = self.feature_extractor.transform(texts)

        self.win_probability_models = train_all_win_probability_models(joined, features)
        self.output_token_models = train_output_token_models(joined, features)
        self.input_token_models = train_input_token_models(joined, texts)

        self._fitted = True
        return self

    def estimate(self, query_text: str) -> dict:
        """p_hat / cost_hat for a single query, independent of tier/lambda.

        Exposed separately from predict() so calibration.py's bisection search
        (which needs to sweep many candidate lambdas over the same p_hat/cost_hat)
        doesn't have to re-run the feature extractor and all six models per
        candidate -- only decision.decide (a cheap arithmetic comparison) varies.
        """
        if not self._fitted:
            raise RuntimeError("RouterPipeline.estimate called before fit")
        features = self.feature_extractor.transform([query_text])
        simple_features = build_simple_features([query_text])

        p_hat = {m: float(self.win_probability_models[m].predict_proba(features)[0]) for m in MODELS}
        out_hat = {
            m: float(self.output_token_models[m]["p90"].predict(features)[0]) for m in MODELS
        }
        in_hat = {m: float(self.input_token_models[m].predict(simple_features)[0]) for m in MODELS}
        cost_hat = {m: estimate_total_cost(K_M[m], in_hat[m], out_hat[m]) for m in MODELS}
        return {"p_hat": p_hat, "cost_hat": cost_hat}

    def estimate_texts(self, texts: List[str]) -> List[dict]:
        """Batched counterpart to estimate(): one p_hat/cost_hat dict per text,
        computed via ONE feature-extractor/model call over the whole list
        instead of one call per text.

        Every sub-step here (embedding encode, TF-IDF+SVD transform, LightGBM/
        Ridge predict) is a frozen, per-row-independent function -- batching
        the array shape changes nothing about a given row's own output, it
        only avoids re-paying fixed per-call overhead (e.g. a transformer
        forward pass) once per row. This does NOT reintroduce the
        batch-relative-statistic risk predict_batch()'s docstring warns
        about (no mean/std over the batch is computed anywhere in this path)
        -- see validate.py's B4 check, which asserts exactly this equivalence.

        Exists because looping estimate()/predict() one text at a time is the
        dominant cost of a full-batch run (each call re-invokes the embedding
        backend for a single item) and risks exceeding the competition
        container's 90-seconds-per-tier budget (docs/RUNTIME.md) at real
        Train/Dev batch sizes.
        """
        if not self._fitted:
            raise RuntimeError("RouterPipeline.estimate_texts called before fit")
        features = self.feature_extractor.transform(texts)
        simple_features = build_simple_features(texts)

        p_hat_by_model = {m: self.win_probability_models[m].predict_proba(features) for m in MODELS}
        out_hat_by_model = {
            m: self.output_token_models[m]["p90"].predict(features) for m in MODELS
        }
        in_hat_by_model = {m: self.input_token_models[m].predict(simple_features) for m in MODELS}

        results = []
        for i in range(len(texts)):
            p_hat = {m: float(p_hat_by_model[m][i]) for m in MODELS}
            in_hat = {m: float(in_hat_by_model[m][i]) for m in MODELS}
            out_hat = {m: float(out_hat_by_model[m][i]) for m in MODELS}
            cost_hat = {m: estimate_total_cost(K_M[m], in_hat[m], out_hat[m]) for m in MODELS}
            results.append({"p_hat": p_hat, "cost_hat": cost_hat})
        return results

    def predict_texts(self, texts: List[str], tier: str) -> List[str]:
        """Batched counterpart to predict(): one decision per text. See
        estimate_texts() for why this matters for the 90s/tier runtime
        budget."""
        if not self._fitted:
            raise RuntimeError("RouterPipeline.predict_texts called before fit")
        lam = get_lambda(tier, TOKEN_ACCOUNTING)
        return [decide(r["p_hat"], r["cost_hat"], lam) for r in self.estimate_texts(texts)]

    def _predict_row(self, query_text: str, tier: str) -> dict:
        result = self.estimate(query_text)
        lam = get_lambda(tier, TOKEN_ACCOUNTING)
        decision = decide(result["p_hat"], result["cost_hat"], lam)
        return {**result, "decision": decision}

    def predict(self, query_text: str, tier: str) -> dict:
        if not self._fitted:
            raise RuntimeError("RouterPipeline.predict called before fit")
        return self._predict_row(query_text, tier)

    def predict_batch(self, df: pd.DataFrame, tier: str) -> pd.DataFrame:
        """Row-by-row mapping over `df` (which must have a `text` column) --
        deliberately not vectorized, so no batch-relative statistic can creep in.
        This is the concrete implementation behind validate.py's B4 guarantee.
        """
        if not self._fitted:
            raise RuntimeError("RouterPipeline.predict_batch called before fit")
        import pandas as pd  # training/offline-only dependency -- see the TYPE_CHECKING import above

        records = []
        for _, row in df.iterrows():
            result = self._predict_row(row["text"], tier)
            records.append({"episode_id": row.get("episode_id"), "decision": result["decision"]})
        return pd.DataFrame.from_records(records)

    def save(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.feature_extractor.save(out_dir / "feature_extractor.joblib")
        for m in MODELS:
            self.win_probability_models[m].save(out_dir / f"win_prob_{m}.joblib")
            self.output_token_models[m]["p90"].save(out_dir / f"out_tokens_p90_{m}.joblib")
            self.output_token_models[m]["p50"].save(out_dir / f"out_tokens_p50_{m}.joblib")
            self.input_token_models[m].save(out_dir / f"in_tokens_{m}.joblib")
        (out_dir / "meta.json").write_text(json.dumps({"models": list(MODELS)}))

    @classmethod
    def load(cls, out_dir: Path, cfg=default_config) -> "RouterPipeline":
        out_dir = Path(out_dir)
        pipeline = cls(cfg=cfg)
        pipeline.feature_extractor = FeatureExtractor.load(out_dir / "feature_extractor.joblib")
        for m in MODELS:
            pipeline.win_probability_models[m] = WinProbabilityModel.load(out_dir / f"win_prob_{m}.joblib")
            pipeline.output_token_models[m] = {
                "p90": OutputTokenQuantileModel.load(out_dir / f"out_tokens_p90_{m}.joblib"),
                "p50": OutputTokenQuantileModel.load(out_dir / f"out_tokens_p50_{m}.joblib"),
            }
            pipeline.input_token_models[m] = InputTokenLinearModel.load(out_dir / f"in_tokens_{m}.joblib")
        pipeline._fitted = True
        return pipeline
