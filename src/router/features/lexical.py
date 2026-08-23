"""TF-IDF + SVD lexical features (§5.1): word 1-2gram -> SVD(80), char 3-5gram ->
SVD(40). Fit on Train only; frozen (pickled) for inference (§5.1 계산 위치, §8.1 B4).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from router.config import RANDOM_SEED, SVD_DIMS


# Uncapped vocabularies were fine against router/synthetic_data.py's small,
# single-language fixture, but real Train text mixes Korean/English/code/math
# (see router/README.md) -- char 3-5gram vocabulary especially explodes
# combinatorially over Hangul syllables. Without a cap, the fitted vectorizer's
# vocabulary_ (kept for inference) balloons the pickled artifact to ~500MB+ and
# slows every transform() call, threatening both the container image size and
# the 90s-per-tier runtime budget (docs/RUNTIME.md). SVD only keeps 40-80
# components anyway, so capping the input vocabulary by document frequency
# loses negligible signal.
_MAX_WORD_FEATURES = 50_000
_MAX_CHAR_FEATURES = 50_000


class LexicalFeatures:
    def __init__(self):
        self._word_vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), max_features=_MAX_WORD_FEATURES
        )
        self._word_svd = TruncatedSVD(n_components=SVD_DIMS["tfidf_word"], random_state=RANDOM_SEED)
        self._char_vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), max_features=_MAX_CHAR_FEATURES
        )
        self._char_svd = TruncatedSVD(n_components=SVD_DIMS["tfidf_char"], random_state=RANDOM_SEED)
        self._fitted = False

    def fit(self, texts: List[str]) -> "LexicalFeatures":
        word_sparse = self._word_vectorizer.fit_transform(texts)
        self._word_svd.fit(word_sparse)
        char_sparse = self._char_vectorizer.fit_transform(texts)
        self._char_svd.fit(char_sparse)
        self._fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LexicalFeatures.transform called before fit")
        word_sparse = self._word_vectorizer.transform(texts)
        word_dense = self._word_svd.transform(word_sparse)
        char_sparse = self._char_vectorizer.transform(texts)
        char_dense = self._char_svd.transform(char_sparse)
        return np.concatenate([word_dense, char_dense], axis=1).astype(np.float32)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "LexicalFeatures":
        return joblib.load(path)
