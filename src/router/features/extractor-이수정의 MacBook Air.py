"""The frozen-at-inference FeatureExtractor (§5.1, §8.1 B4).

Concatenates: embedding(128, optional) + word-tfidf-svd(80) + char-tfidf-svd(40) +
handcrafted(~20). Each sub-branch processes rows independently (no cross-row
statistics), so no batch-relative computation can leak into a feature -- this is
part of the structural guarantee validate.py's B4 check polices.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import numpy as np

from router import config as default_config
from router.features.embeddings import EmbeddingBackend, build_embedding_backend
from router.features.handcrafted import FEATURE_NAMES as HANDCRAFTED_FEATURE_NAMES
from router.features.handcrafted import extract_handcrafted
from router.features.lexical import LexicalFeatures
from router.features.preprocess import normalize_text


class FeatureExtractor:
    def __init__(self, cfg=default_config):
        # Only plain, picklable data is kept as instance state (§8.1 B4 note:
        # this object gets joblib-pickled for inference -- a stored reference to
        # the `router.config` module itself is not picklable).
        self._svd_dims = dict(cfg.SVD_DIMS)
        self._lexical = LexicalFeatures()
        # The embedding backend is a frozen pretrained model, independent of
        # Train -- built eagerly here rather than in fit(), since unlike the
        # lexical sub-vectorizers it needs no fitting.
        self._embedding_backend: EmbeddingBackend | None = build_embedding_backend(cfg)
        self._fitted = False

    @property
    def feature_names(self) -> List[str]:
        names: List[str] = []
        if self._embedding_backend is not None:
            names += [f"emb_{i}" for i in range(self._svd_dims["embedding"])]
        names += [f"tfidf_word_svd_{i}" for i in range(self._svd_dims["tfidf_word"])]
        names += [f"tfidf_char_svd_{i}" for i in range(self._svd_dims["tfidf_char"])]
        names += HANDCRAFTED_FEATURE_NAMES
        return names

    def fit(self, texts: List[str]) -> "FeatureExtractor":
        """Fit the TF-IDF+SVD sub-vectorizers on Train only. The embedding backend
        (if enabled) is a frozen pretrained model and is never fit here."""
        normalized = [normalize_text(t) for t in texts]
        self._lexical.fit(normalized)
        self._fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """The ONLY method inference is allowed to call.

        Calling fit / fit_transform at inference time (e.g. re-fitting the SVD per
        submission batch) is exactly what validate.py's B4 check is designed to
        catch -- it silently makes the decision depend on which other rows happen
        to be in the same batch. Always fit() once on Train, save(), and load()
        for inference.
        """
        if not self._fitted:
            raise RuntimeError("FeatureExtractor.transform called before fit")
        normalized = [normalize_text(t) for t in texts]

        blocks = []
        if self._embedding_backend is not None:
            blocks.append(self._embedding_backend.encode(normalized))
        blocks.append(self._lexical.transform(normalized))
        handcrafted_rows = [extract_handcrafted(t) for t in texts]  # full text, pre-normalize
        handcrafted_matrix = np.array(
            [[row[name] for name in HANDCRAFTED_FEATURE_NAMES] for row in handcrafted_rows],
            dtype=np.float32,
        )
        blocks.append(handcrafted_matrix)
        return np.concatenate(blocks, axis=1)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "FeatureExtractor":
        return joblib.load(path)
