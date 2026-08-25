"""Single source of truth for every tunable constant and every not-yet-resolved switch.

Every value here that the planning doc left ambiguous (§10.1, §10.2, embedding model
choice) is called out in a comment rather than silently baked into downstream code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Tuple

# --- Canonical model set -----------------------------------------------------
# Fixed order. Also used as the deterministic tie-break key in decision.decide
# (never rely on dict/insertion order for that — see §5.5).
MODELS: Tuple[str, str, str] = ("ax31-light", "ax31", "axk1-think")

# --- Cost model (§1.1) --------------------------------------------------------
# c_m(q) = K_M[m] * (in_m(q) + 4 * out_m(q)), fixed cost = 0.
K_M: Dict[str, float] = {
    "ax31-light": 1.000,
    "ax31": 2.127,
    "axk1-think": 6.565,
}

# Empirically observed total-cost ratio vs. ax31-light (§1.1 table). Diagnostics /
# sanity-checking only -- NOT used inside the cost formula or the decision rule.
R_M: Dict[str, float] = {
    "ax31-light": 1.000,
    "ax31": 2.102,
    "axk1-think": 23.795,
}

RANDOM_SEED = 42

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


# --- Tiers (§3.1) -------------------------------------------------------------
@dataclass(frozen=True)
class TierConfig:
    name: str
    budget_ratio: float
    baseline_model: str
    lambda_star_total: float
    lambda_star_per_call: float
    usage_target: Tuple[float, float]
    overshoot_max: float
    p90_coverage_target: float


TIERS: Dict[str, TierConfig] = {
    "Fast": TierConfig(
        name="Fast",
        budget_ratio=1.25,
        baseline_model="ax31-light",
        lambda_star_total=21.45,
        lambda_star_per_call=9.23,
        usage_target=(0.85, 0.90),
        overshoot_max=0.02,
        p90_coverage_target=0.90,
    ),
    "Balanced": TierConfig(
        name="Balanced",
        budget_ratio=2.00,
        baseline_model="ax31-light",
        lambda_star_total=9.39,
        lambda_star_per_call=3.87,
        usage_target=(0.85, 0.90),
        overshoot_max=0.02,
        p90_coverage_target=0.90,
    ),
    "Premium": TierConfig(
        name="Premium",
        budget_ratio=4.00,
        baseline_model="ax31",
        lambda_star_total=2.63,
        lambda_star_per_call=0.69,
        usage_target=(0.85, 0.90),
        overshoot_max=0.02,
        p90_coverage_target=0.90,
    ),
}

# --- Token accounting convention (§10.2) -------------------------------------
# UNRESOLVED as of writing: it is not yet confirmed whether the competition's cost
# accounting counts tokens across *all* num_generations attempts ("total") or per a
# single generation ("per_call"). This changes lambda* by ~2.3x (see TierConfig
# above). Run scripts/diagnose_token_convention.py against real Dev data before
# trusting this default, then set it explicitly -- do not guess silently.
TOKEN_ACCOUNTING: Literal["total", "per_call"] = "total"


def get_lambda(tier: str, accounting: str | None = None) -> float:
    """Look up lambda for a tier under the given (or configured) token-accounting convention."""
    cfg = TIERS[tier]
    mode = accounting or TOKEN_ACCOUNTING
    if mode == "total":
        return cfg.lambda_star_total
    if mode == "per_call":
        return cfg.lambda_star_per_call
    raise ValueError(f"unknown token accounting convention: {mode!r}")


# --- Embedding branch (§5.1, §10.1) ------------------------------------------
# Whether the ARM-latency-gated embedding branch runs at all. Flip to False to fall
# back to TF-IDF-only mode (§10.1 "실패" path) without restructuring the extractor.
#
# §10.1 ARM gate: FAILED. On real arm64 hardware (Apple Silicon, not
# emulated), the embedding branch (ONNX Runtime, fp32, EMBEDDING_MAX_TOKENS
# trimmed 512->256, batch_size=16, memory-pattern leak fixed) still needed
# 150s for real Train+Dev -- well over the 90s-per-tier budget
# (docs/RUNTIME.md) with no further safe lever found in the time available
# before the submission deadline. TF-IDF+handcrafted only (this setting) is
# what ships; router.RouterPipeline was retrained/recalibrated for this.
USE_EMBEDDING_BRANCH: bool = False

# §10.1 ARM latency/image-size gate result: "sentence_transformers" pulls in
# torch, which resolves to a multi-GB CUDA-bundled build on linux/arm64 and
# does not fit the submission container's 1 GiB compressed-image budget
# (docs/RUNTIME.md) -- see router/scripts/export_embedding_onnx.py and
# features/embeddings.py's module docstring. "onnxruntime" is what ships.
EMBEDDING_BACKEND: Literal["sentence_transformers", "onnxruntime"] = "onnxruntime"

# Directory containing the exported {model file, tokenizer files} produced by
# router/scripts/export_embedding_onnx.py. Overridable via
# ROUTER_EMBEDDING_ONNX_DIR so the same config resolves a local dev path and
# the container's baked-in path (e.g. /opt/router/artifacts/e5-small-onnx,
# set by container/Dockerfile) without a code change.
EMBEDDING_ONNX_DIR: str = os.environ.get("ROUTER_EMBEDDING_ONNX_DIR", "router/artifacts/e5-small-onnx")
# int8 dynamic quantization (model.int8.onnx, 470MB -> 118MB) was tried first
# for extra image-size margin, but onnxruntime's CPU int8 GEMM kernels turned
# out to be dramatically SLOWER than fp32 on real arm64 hardware (observed:
# 880 rows did not finish in 5+ minutes on Apple Silicon at 200% CPU, vs 33s
# for the equivalent fp32 sentence-transformers backend on a weaker x86_64
# laptop) -- likely missing/unoptimized ARM int8 dot-product code paths.
# fp32 costs ~350MB more image size (still comfortably under the 1 GiB
# budget alongside everything else) and is the one actually shipped.
EMBEDDING_ONNX_MODEL_FILENAME: str = "model.onnx"

# Primary per §5.1: 384-dim, MIT license, chosen for its small size relative to the
# 1,760-example training set (larger models were observed/reasoned to overfit).
EMBEDDING_MODEL_PRIMARY = "intfloat/multilingual-e5-small"
# Pinned so training and the offline (no-network) submission container always
# resolve the exact same snapshot -- must match container/Dockerfile's
# E5_SMALL_REVISION build arg and container/MODEL_AND_DEPENDENCY_NOTICES.md.
EMBEDDING_MODEL_PRIMARY_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
# Fallback ONLY if the §10.1 ARM latency benchmark passes with margin.
EMBEDDING_MODEL_FALLBACK = "intfloat/multilingual-e5-base"
# Documented contingency/backup embedding model (§5.1 table), different license
# (Apache 2.0) and architecture family -- kept as a config constant, not wired in.
EMBEDDING_MODEL_CONTINGENCY = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# E5 convention: inputs MUST be prefixed with this string, identically at train and
# inference time, or embedding quality silently degrades (§5.1 "필수 준수사항" #1).
E5_QUERY_PREFIX = "query: "
# Lowered from the design doc's original 512: attention cost scales
# quadratically with sequence length, and the container's 90s-per-tier /
# 2 GiB memory budget (docs/RUNTIME.md) could not fit real Train+Dev
# (2,640 episodes) through the embedding branch at 512 even after fixing an
# unrelated onnxruntime memory-pattern leak (confirmed clean completion at
# 185s -- still ~2x over budget) on real arm64 hardware. 256 directly cuts
# the worst-case (long-document) attention cost ~4x; most episodes are far
# shorter than either cap so this mainly affects the rare long-context
# episodes (e.g. the BABILong 4K/16K component, THIRD_PARTY_NOTICES.md).
# RouterPipeline was retrained/recalibrated against this exact value --
# changing it again requires redoing both (router/scripts/train.py,
# router/scripts/calibrate_against_official_scorer.py).
EMBEDDING_MAX_TOKENS = 256

# --- Feature dimensionality (§5.1) -------------------------------------------
SVD_DIMS: Dict[str, int] = {
    "embedding": 128,
    "tfidf_word": 80,
    "tfidf_char": 40,
}

# --- LightGBM defaults (§5.2, §5.3) ------------------------------------------
# num_threads=1 + deterministic=True are load-bearing for validate.py's
# repeatability check (§8.1 C) -- do not relax these for a speed win.
LGBM_COMMON_PARAMS: Dict[str, object] = {
    "num_threads": 1,
    "deterministic": True,
    "force_row_wise": True,
    "seed": RANDOM_SEED,
}
