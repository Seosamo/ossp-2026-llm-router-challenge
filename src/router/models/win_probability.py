"""Win-probability classifiers (§5.2), one LightGBM binary classifier per model.

Training data is exploded to the per-attempt level (schema.explode_to_attempts)
and weighted by regret (schema.compute_regret_weight) so that the ~64-74% of
queries where models tie (§1.3) don't dominate training signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression

from router.config import LGBM_COMMON_PARAMS, MODELS

if TYPE_CHECKING:
    # pandas + router.schema (also pandas-based) are training-only -- see
    # train_all_win_probability_models below, where both are imported lazily
    # so WinProbabilityModel (used at inference) doesn't require pandas.
    import pandas as pd


class WinProbabilityModel:
    def __init__(self, **lgbm_params):
        params = {"objective": "binary", **LGBM_COMMON_PARAMS, **lgbm_params}
        self._booster: lgb.LGBMClassifier = lgb.LGBMClassifier(**params)
        self._calibrator: IsotonicRegression | None = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> "WinProbabilityModel":
        self._booster.fit(X, y, sample_weight=sample_weight)
        self._fitted = True
        return self

    def calibrate(self, X_calib: np.ndarray, y_calib: np.ndarray) -> "WinProbabilityModel":
        """Isotonic calibration (§5.2 후처리) so predict_proba reflects true
        empirical win rates rather than raw LightGBM scores."""
        raw = self._booster.predict_proba(X_calib)[:, 1]
        self._calibrator = IsotonicRegression(out_of_bounds="clip")
        self._calibrator.fit(raw, y_calib)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("WinProbabilityModel.predict_proba called before fit")
        raw = self._booster.predict_proba(X)[:, 1]
        if self._calibrator is not None:
            return self._calibrator.predict(raw)
        return raw

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "WinProbabilityModel":
        return joblib.load(path)


def train_all_win_probability_models(
    train_df: pd.DataFrame,
    features_by_row: np.ndarray,
    calib_fraction: float = 0.2,
    seed: int = 0,
) -> Dict[str, WinProbabilityModel]:
    """Train one WinProbabilityModel per model in MODELS.

    train_df must already have `episode_id` aligned 1:1 (by row order) with
    features_by_row -- i.e. features_by_row[i] is the feature vector for
    train_df.iloc[i]. explode_to_attempts re-derives features per exploded row via
    a merge back onto `episode_id`; callers should pass a features lookup keyed the
    same way (see pipeline.py for the concrete wiring).
    """
    from router.schema import compute_regret_weights, explode_to_attempts  # training-only, see TYPE_CHECKING import above

    rng = np.random.default_rng(seed)
    n = len(train_df)
    calib_mask = rng.random(n) < calib_fraction

    models: Dict[str, WinProbabilityModel] = {}
    weights = compute_regret_weights(train_df, MODELS)

    for m in MODELS:
        exploded = explode_to_attempts(train_df, m)
        row_idx = exploded["_source_row"].to_numpy()
        X = features_by_row[row_idx]
        y = exploded["label"].to_numpy()
        w = weights.to_numpy()[row_idx]

        fit_rows = ~calib_mask[row_idx]
        calib_rows = calib_mask[row_idx]

        model = WinProbabilityModel()
        model.fit(X[fit_rows], y[fit_rows], sample_weight=w[fit_rows])
        if calib_rows.any():
            model.calibrate(X[calib_rows], y[calib_rows])
        models[m] = model

    return models
