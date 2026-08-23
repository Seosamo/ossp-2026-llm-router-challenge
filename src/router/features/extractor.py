"""The frozen-at-inference FeatureExtractor (§5.1, §8.1 B4).

Concatenates: embedding(128, optional) + word-tfidf-svd(80) + char-tfidf-svd(40) +
handcrafted(~20). Each sub-branch processes rows independently (no cross-row
statistics), so no batch-relative computation can leak into a feature -- this is
part of the structural guarantee validate.py's B4 check polices.
"""

from __future__ import annotations

import importlib
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
        self._cfg = cfg
        self._lexical = LexicalFeatures()
        self._embedding_backend: EmbeddingBackend | None = None
        self._fitted = False

    @property
    def feature_names(self) -> List[str]:
        names: List[str] = []
        if self._embedding_backend is not None:
            names += [f"emb_{i}" for i in range(self._cfg.SVD_DIMS["embedding"])]
        names += [f"tfidf_word_svd_{i}" for i in range(self._cfg.SVD_DIMS["tfidf_word"])]
        names += [f"tfidf_char_svd_{i}" for i in range(self._cfg.SVD_DIMS["tfidf_char"])]
        names += HANDCRAFTED_FEATURE_NAMES
        return names

    def fit(self, texts: List[str]) -> "FeatureExtractor":
        """Fit the TF-IDF+SVD sub-vectorizers on Train only. The embedding backend
        (if enabled) is a frozen pretrained model and is never fit here -- it is
        merely constructed."""
        normalized = [normalize_text(t) for t in texts]
        self._lexical.fit(normalized)
        self._embedding_backend = build_embedding_backend(self._cfg)
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

    def __getstate__(self) -> dict:
        """Two attributes need special handling to keep the saved artifact
        small and portable:

        - `self._cfg` is normally the live `router.config` module, which
          plain pickle/joblib cannot serialize ("cannot pickle 'module'
          object"). Persist its import path instead and re-import it in
          __setstate__ -- config is a fixed set of constants, not
          per-instance state, so re-importing by name reconstructs it
          exactly.
        - `self._embedding_backend` wraps a frozen pretrained model
          (SentenceTransformerBackend/OnnxEmbeddingBackend) that is
          constructed, never fit (see fit()'s docstring). Naively pickling it
          embeds the full model weights a second time inside this joblib
          file -- e.g. multilingual-e5-small alone balloons a ~1KB extractor
          into a ~1GB file, which alone blows the container image's 1GiB
          budget. Drop it here and rebuild it fresh in __setstate__ instead,
          so the weights are loaded exactly once, from wherever the runtime
          environment bundles them (e.g. the image's local HF cache), never
          from this pickle.
        """
        state = self.__dict__.copy()
        state["_cfg"] = getattr(self._cfg, "__name__", None)
        state["_embedding_backend"] = self._embedding_backend is not None
        return state

    def __setstate__(self, state: dict) -> None:
        cfg_name = state.get("_cfg")
        cfg = importlib.import_module(cfg_name) if isinstance(cfg_name, str) else default_config
        had_embedding_backend = state.get("_embedding_backend", False)
        state = {
            **state,
            "_cfg": cfg,
            "_embedding_backend": build_embedding_backend(cfg) if had_embedding_backend else None,
        }
        self.__dict__.update(state)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "FeatureExtractor":
        return joblib.load(path)
