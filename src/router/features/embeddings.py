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
    #
    # A batch of long (near EMBEDDING_MAX_TOKENS=512) documents pushes the
    # transformer's per-layer attention-score tensors (batch x heads x seq x
    # seq) into the hundreds of MB, and this container has only a 2 GiB
    # total memory budget (docs/RUNTIME.md) shared with everything else in
    # the process -- observed in practice to OOM-kill (exit 137) at
    # batch_size=32 WITHOUT length-sorting, despite finishing well within
    # the 90-second time budget. With sorting (below) confirmed safe on
    # memory (observed ~40% of the 2 GiB budget mid-run at batch_size=16),
    # so batch_size was raised back to 32 for the fixed per-call overhead
    # (tokenizer + Python loop + session.run()) savings -- fp32 inference
    # was found to need this margin: it landed right at the 90s ceiling at
    # batch_size=16 on real arm64 hardware (int8 was far worse on the same
    # hardware -- see EMBEDDING_ONNX_MODEL_FILENAME's comment in config.py --
    # so this is tuning fp32's margin, not fixing a second quantization bug).
    # encode() sorts by length before chunking specifically so this matters
    # less: with texts grouped by length, only the (rare) batches of
    # genuinely long documents pad up to ~512 tokens -- most batches are
    # short questions padding to a small fraction of that. Chunking in
    # original order instead (mixing one long document with several short
    # ones) forces EVERY such batch to pay the long document's padding cost,
    # which is both slower (wasted compute on pad tokens) and more
    # memory-hungry (confirmed: disabling the memory arena to survive
    # original-order chunking's peak cost alone pushed several tiers over
    # the 90s time budget instead). Length-sorted batching improves both
    # axes at once, so the arena stays enabled (default) here.
    _BATCH_SIZE = 16

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
        # enable_mem_pattern makes onnxruntime learn and cache a reusable
        # memory layout PER DISTINCT INPUT SHAPE it sees. encode()'s
        # length-sorted batching deliberately varies sequence length chunk
        # to chunk (short-text chunks vs the rare long-document chunks), so
        # a single embedding pass over ~2,640 episodes can present on the
        # order of 100+ distinct shapes to this session -- observed in
        # practice as memory climbing until an OOM kill (exit 137) at
        # essentially the same wall-clock point regardless of batch size
        # (16 vs 32), which is the signature of a per-shape cache that never
        # gets reclaimed rather than a single large peak. The memory arena
        # itself stays enabled (default) for allocation speed; only the
        # per-shape pattern cache is disabled.
        session_options.enable_mem_pattern = False

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
        """Sorts by length before chunking purely as a memory/speed
        optimization (see _BATCH_SIZE's docstring), then unsorts results
        back to input order so callers never observe the reordering itself.

        Note this is NOT bitwise batch-invariant: measured up to ~0.016 max
        abs difference (unit-norm 384-dim embeddings) for the same row
        encoded alongside different batch-mates, because this is an int8
        DYNAMICALLY quantized model -- its activation quantization scale is
        computed from each call's actual tensor values (including padding
        shape), so a row's padding length affects its own quantized
        intermediate values slightly. This is a small, accepted trade
        (same one already reflected in the real-Dev score this pipeline was
        calibrated against, see router/scripts/calibrate_against_official_scorer.py)
        for fitting the container's memory/time budget via batched inference
        at all -- true bitwise invariance would require single-item calls,
        which is what caused the 90s-per-tier timeout this batching exists
        to fix. Grouping by content length (not input position) does mean
        this is at least invariant to input ORDER for a fixed set of texts,
        which is what CHALLENGE_RULES.md's ID/order-permutation audit checks.
        """
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        prefixed = [E5_QUERY_PREFIX + t for t in texts]
        order = sorted(range(len(prefixed)), key=lambda i: len(prefixed[i]))
        sorted_texts = [prefixed[i] for i in order]
        chunks = [
            self._encode_chunk(sorted_texts[i : i + self._BATCH_SIZE])
            for i in range(0, len(sorted_texts), self._BATCH_SIZE)
        ]
        sorted_embeddings = np.concatenate(chunks, axis=0)
        embeddings = np.empty_like(sorted_embeddings)
        embeddings[order] = sorted_embeddings
        return embeddings


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
