# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Input-token estimator (§5.4) -- deliberately simple.

§5.4 explicitly states plain linear regression on char/word counts and character-
class ratios is sufficient, and that estimation error here barely matters: cost
weights output tokens 4x, and think's output alone accounts for >90% of total
token mass, so input-token error is swamped. Do not overbuild this component.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import joblib
import numpy as np
from sklearn.linear_model import Ridge

from router.config import MODELS
from router.features.handcrafted import extract_handcrafted

if TYPE_CHECKING:
    # pandas is a training-only dependency -- see train_input_token_models
    # below (InputTokenLinearModel/build_simple_features, used at inference,
    # never need it).
    import pandas as pd

_SIMPLE_FEATURE_NAMES = ["char_len", "word_len", "digit_ratio", "uppercase_ratio", "punctuation_ratio"]


def build_simple_features(texts: List[str]) -> np.ndarray:
    rows = [extract_handcrafted(t) for t in texts]
    return np.array([[row[name] for name in _SIMPLE_FEATURE_NAMES] for row in rows], dtype=np.float32)


class InputTokenLinearModel:
    def __init__(self, alpha: float = 1.0):
        self._model = Ridge(alpha=alpha)
        self._fitted = False

    def fit(self, X_simple: np.ndarray, input_tokens: np.ndarray) -> "InputTokenLinearModel":
        self._model.fit(X_simple, input_tokens)
        self._fitted = True
        return self

    def predict(self, X_simple: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("InputTokenLinearModel.predict called before fit")
        return self._model.predict(X_simple)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "InputTokenLinearModel":
        return joblib.load(path)


def train_input_token_models(
    train_df: pd.DataFrame, texts: List[str]
) -> Dict[str, InputTokenLinearModel]:
    X_simple = build_simple_features(texts)
    result: Dict[str, InputTokenLinearModel] = {}
    for m in MODELS:
        input_tokens = train_df[f"input_tokens__{m}"].to_numpy()
        result[m] = InputTokenLinearModel().fit(X_simple, input_tokens)
    return result
