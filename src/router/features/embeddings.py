"""Sentence embedding backends (§5.1).

Two implementations behind one interface:
    - SentenceTransformerBackend: uses the `sentence-transformers` library,
      which already implements E5-correct mean pooling + L2 normalize. Pulls
      in torch + transformers -- on linux/arm64 this drags in multi-GB NVIDIA
      CUDA libraries as a transitive dependency of plain "pip install torch"
      even though the submission container is CPU-only, and the full stack
      does not fit the container's 1 GiB compressed-image budget
      (docs/RUNTIME.md). Kept for local development/offline training use.
    - OnnxEmbeddingBackend: the §10.1 ARM-latency contingency path, and what
      the submission container actually uses -- onnxruntime alone is tens of
      MB with no torch dependency at all. ONNX Runtime has no built-in
      pooler, so mean pooling (weighted by the attention mask) and L2
      normalization are implemented by hand here. Verified numerically
      identical (cosine similarity 1.0) to SentenceTransformerBackend for the
      unquantized export; the shipped int8-quantized model trades a small,
      accepted accuracy loss (cosine similarity ~0.987-0.99) for a ~4x size
      reduction (470MB -> 118MB) -- see router/scripts/export_embedding_onnx.py.

Both backends apply the mandatory "query: " prefix (§5.1 필수 준수사항 #1) and
truncate at EMBEDDING_MAX_TOKENS (#3) identically at train and inference time.

build_embedding_backend returns None when the branch is disabled (config.
USE_EMBEDDING_BRANCH=False), so features/extractor.py can degrade to TF-IDF-only
mode without any structural change -- just a shorter feature vector.
"""

from __future__ import annotations

import abc
from pathlib import Path
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
    # SentenceTransformer.encode() batches internally (default batch_size=32);
    # this backend must do the same. Padding pads every sequence in a call to
    # the longest one IN THAT CALL, so passing hundreds/thousands of texts at
    # once (e.g. a full Train batch, see FeatureExtractor.transform) pads them
    # all to the single longest text's length and allocates the whole
    # batch x seq_len x hidden intermediate tensors at once -- this has been
    # observed to fail ("bad allocation" in a MatMul node) at real Train batch
    # sizes (~1,760 rows, some near the 512-token truncation limit).
    _BATCH_SIZE = 32

    def __init__(self, model_path: str, tokenizer_path: str):
        import onnxruntime as ort  # lazy import
        from transformers import AutoTokenizer  # lazy import

        # onnxruntime auto-sizes its thread pool from the number of CPUs it
        # sees, which inside a container can still be the HOST's full core
        # count (cgroup CPU quotas are not always honored) rather than the
        # container's --cpus 2 allocation. Left unbounded, this competes with
        # OpenBLAS's own thread pool (see router/config.py's OPENBLAS_NUM_THREADS
        # etc.) for the container's --pids-limit 32 budget -- observed in
        # practice to blow that budget (pthread_create failing past thread
        # ~32) and burn most of the 90s-per-tier budget on failed retries
        # rather than actual work (docs/RUNTIME.md). Capped explicitly here
        # since (unlike OpenBLAS) onnxruntime does not reliably respect the
        # OMP_NUM_THREADS-style environment variables.
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 2
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self._session = ort.InferenceSession(model_path, sess_options=session_options)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def _mean_pool(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        # ONNX Runtime has no pooler -- E5's mean pooling must be done by hand:
        # average token embeddings, weighted by the attention mask (so padding
        # tokens don't drag the mean down).
        mask = attention_mask[..., None].astype(np.float32)
        summed = (token_embeddings * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        return summed / counts

    def _encode_chunk(self, chunk: List[str]) -> np.ndarray:
        encoded = self._tokenizer(
            chunk,
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

    def encode(self, texts: List[str]) -> np.ndarray:
        prefixed = [E5_QUERY_PREFIX + t for t in texts]
        chunks = [
            self._encode_chunk(prefixed[i : i + self._BATCH_SIZE])
            for i in range(0, len(prefixed), self._BATCH_SIZE)
        ]
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)


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
        onnx_dir = Path(cfg.EMBEDDING_ONNX_DIR)
        return OnnxEmbeddingBackend(
            model_path=str(onnx_dir / cfg.EMBEDDING_ONNX_MODEL_FILENAME),
            tokenizer_path=str(onnx_dir),
        )
    raise ValueError(f"unknown embedding backend: {cfg.EMBEDDING_BACKEND!r}")
