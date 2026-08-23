"""Single source of truth for every tunable constant and every not-yet-resolved switch.

Every value here that the planning doc left ambiguous (§10.1, §10.2, embedding model
choice) is called out in a comment rather than silently baked into downstream code.
"""

from __future__ import annotations

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
USE_EMBEDDING_BRANCH: bool = True

EMBEDDING_BACKEND: Literal["sentence_transformers", "onnxruntime"] = "sentence_transformers"

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
EMBEDDING_MAX_TOKENS = 512

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
