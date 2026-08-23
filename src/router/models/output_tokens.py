"""Output-token p90 quantile regressors (§5.3) -- flagged in the planning doc as
the PRIMARY / lead component of the whole system, because §1.5 found that
per-query cost variance (not accuracy) is what makes think's routing decisions
consequential (p10/p90 spread ~20x).

Primary: LightGBM objective='quantile', alpha=0.9, target log(output_tokens+1).
Auxiliary: alpha=0.5 (median) -- diagnostic only, never fed into the cost formula.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from router.config import LGBM_COMMON_PARAMS, MODELS

# TODO(future, not required for this scaffold): a per-query output-token variance /
# outlier flag, to let the decision rule react to unusually uncertain predictions
# for think specifically. Deliberately left as a comment, not a stub function, so
# it isn't mistaken for something already implemented.


class OutputTokenQuantileModel:
    def __init__(self, alpha: float, **lgbm_params):
        self.alpha = alpha
        params = {"objective": "quantile", "alpha": alpha, **LGBM_COMMON_PARAMS, **lgbm_params}
        self._booster = lgb.LGBMRegressor(**params)
        self._fitted = False

    def fit(self, X: np.ndarray, log_output_tokens: np.ndarray) -> "OutputTokenQuantileModel":
        self._booster.fit(X, log_output_tokens)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("OutputTokenQuantileModel.predict called before fit")
        log_pred = self._booster.predict(X)
        return np.expm1(log_pred)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "OutputTokenQuantileModel":
        return joblib.load(path)


def train_output_token_models(
    train_df: pd.DataFrame, features_by_row: np.ndarray
) -> Dict[str, Dict[str, OutputTokenQuantileModel]]:
    """Returns {model_name: {"p90": ..., "p50": ...}}. Only "p90" feeds the cost
    formula (decision.estimate_total_cost); "p50" is diagnostic-only."""
    result: Dict[str, Dict[str, OutputTokenQuantileModel]] = {}
    for m in MODELS:
        log_tokens = np.log1p(train_df[f"output_tokens__{m}"].to_numpy())
        p90_model = OutputTokenQuantileModel(alpha=0.9).fit(features_by_row, log_tokens)
        p50_model = OutputTokenQuantileModel(alpha=0.5).fit(features_by_row, log_tokens)
        result[m] = {"p90": p90_model, "p50": p50_model}
    return result
