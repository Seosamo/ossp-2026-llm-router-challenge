"""Sentence embedding backends (§5.1).

Two implementations behind one interface:
    - SentenceTransformerBackend: primary path, uses the `sentence-transformers`
      library, which already implements E5-correct mean pooling + L2 normalize.
    - OnnxEmbeddingBackend: fallback path for the §10.1 ARM-latency contingency --
      ONNX Runtime has no built-in pooler, so mean pooling (weighted by the
      attention mask) and L2 normalization are implemented by hand here.

Both backends apply the mandatory "query: " prefix (§5.1 필수 준수사항 #1) and
truncate at EMBEDDING_MAX_TOKENS (#3) identically at train and inference time.

build_embedding_backend returns None when the branch is disabled (config.
USE_EMBEDDING_BRANCH=False), so features/extractor.py can degrade to TF-IDF-only
mode without any structural change -- just a shorter feature vector.
"""

from __future__ import annotations

import abc
from typing import List

import numpy as np

from router.config import E5_QUERY_PREFIX, EMBEDDING_MAX_TOKENS, EMBEDDING_MODEL_PRIMARY_REVISION


class EmbeddingBackend(abc.ABC):
    @abc.abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Return an (n_texts, dim) float32 array of L2-normalized embeddings."""


class SentenceTransformerBackend(EmbeddingBackend):
    def __init__(self, model_name: str, revision: str | None = None):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name, revision=revision)
        self._model.max_seq_length = EMBEDDING_MAX_TOKENS

    def encode(self, texts: List[str]) -> np.ndarray:
        prefixed = [E5_QUERY_PREFIX + t for t in texts]
        embeddings = self._model.encode(
            prefixed,
            normalize_embeddings=True,  # L2 normalize, matches E5 convention
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)


class OnnxEmbeddingBackend(EmbeddingBackend):
    def __init__(self, model_path: str, tokenizer_path: str):
        import onnxruntime as ort  # lazy import
        from transformers import AutoTokenizer  # lazy import

        self._session = ort.InferenceSession(model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def _mean_pool(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        # ONNX Runtime has no pooler -- E5's mean pooling must be done by hand:
        # average token embeddings, weighted by the attention mask (so padding
        # tokens don't drag the mean down).
        mask = attention_mask[..., None].astype(np.float32)
        summed = (token_embeddings * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        return summed / counts

    def encode(self, texts: List[str]) -> np.ndarray:
        prefixed = [E5_QUERY_PREFIX + t for t in texts]
        encoded = self._tokenizer(
            prefixed,
            padding=True,
            truncation=True,
            max_length=EMBEDDING_MAX_TOKENS,
            return_tensors="np",
        )
        outputs = self._session.run(
            None,
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            },
        )
        token_embeddings = outputs[0]
        pooled = self._mean_pool(token_embeddings, encoded["attention_mask"])
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.clip(norms, 1e-9, None)).astype(np.float32)


def build_embedding_backend(cfg) -> EmbeddingBackend | None:
    """cfg is the router.config module (or a stand-in with the same attributes).

    Returns None when the embedding branch is disabled, letting FeatureExtractor
    skip it entirely rather than requiring a code change (§10.1's failure path).
    """
    if not cfg.USE_EMBEDDING_BRANCH:
        return None
    if cfg.EMBEDDING_BACKEND == "sentence_transformers":
        return SentenceTransformerBackend(
            cfg.EMBEDDING_MODEL_PRIMARY,
            revision=getattr(cfg, "EMBEDDING_MODEL_PRIMARY_REVISION", None),
        )
    if cfg.EMBEDDING_BACKEND == "onnxruntime":
        raise NotImplementedError(
            "ONNX backend requires exported model/tokenizer paths; wire these up "
            "once the §10.1 ARM benchmark selects this path."
        )
    raise ValueError(f"unknown embedding backend: {cfg.EMBEDDING_BACKEND!r}")
