# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Lambda calibration (§6) and the §10.2 token-accounting-convention diagnostic.

Kept as reusable, pure-function/dataclass logic (importable and unit-testable
against synthetic fixtures) so it doesn't require real Dev data or a CLI
invocation to exercise in tests. scripts/run_calibration.py and
scripts/diagnose_token_convention.py are thin argparse wrappers around this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from router.config import RANDOM_SEED, TierConfig


def diagnose_token_convention(wide_df: pd.DataFrame, models: Sequence[str]) -> Dict[str, float]:
    """§10.2: for each model, ratio of mean output_tokens between
    num_generations=4 and num_generations=2 groups.

    ratio ~= 2.0  -> "total" convention (tokens scale with attempt count)
    ratio ~= 1.0  -> "per_call" convention (already normalized per attempt)
    """
    ratios: Dict[str, float] = {}
    for m in models:
        grouped = wide_df.groupby(f"num_generations__{m}")[f"output_tokens__{m}"].mean()
        if 4.0 in grouped.index and 2.0 in grouped.index and grouped[2.0] != 0:
            ratios[m] = float(grouped[4.0] / grouped[2.0])
        else:
            ratios[m] = float("nan")
    return ratios


def compute_usage_rate(decisions: pd.Series, baseline_model: str) -> float:
    """Fraction of decisions that are NOT the tier's baseline model -- i.e. how
    often the paid-up budget is actually being spent (§6.2 step 1-2)."""
    return float((decisions != baseline_model).mean())


def overshoot_threshold(g: float, s_base: float) -> float:
    """§3.3: the tolerable overshoot probability p such that expected overshoot
    cost stays within slack g: p < g / (S_base + g)."""
    return g / (s_base + g)


def bootstrap_overshoot_probability(
    costs: np.ndarray, budget: float, n_boot: int = 1000, seed: int = RANDOM_SEED
) -> float:
    """Bootstrap the per-query cost distribution to estimate P(cost > budget)
    (§6.2 step 3)."""
    rng = np.random.default_rng(seed)
    n = len(costs)
    exceed_fracs = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(costs, size=n, replace=True)
        exceed_fracs[i] = float((sample > budget).mean())
    return float(exceed_fracs.mean())


def bisection_calibrate_lambda(
    usage_rate_fn: Callable[[float], float],
    lo: float,
    hi: float,
    target_range: Tuple[float, float] = (0.85, 0.90),
    tol: float = 1e-3,
    max_iter: int = 50,
) -> float:
    """Bisection search for lambda such that usage_rate_fn(lambda) lands inside
    target_range (§6.2 step 2). usage_rate_fn is expected to be monotonically
    non-increasing in lambda (larger lambda -> more conservative -> lower usage)."""
    mid = (lo + hi) / 2
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        rate = usage_rate_fn(mid)
        if target_range[0] <= rate <= target_range[1]:
            return mid
        if rate > target_range[1]:
            lo = mid  # usage too high -> need larger lambda (more conservative)
        else:
            hi = mid  # usage too low -> need smaller lambda
        if hi - lo < tol:
            break
    return mid


@dataclass
class CalibrationResult:
    lam: float
    usage_rate: float
    overshoot_prob: float
    p90_coverage: float


def calibrate_tier(
    dev_predictions: pd.DataFrame,
    tier_cfg: TierConfig,
    token_accounting: str,
    usage_rate_fn: Callable[[float], float],
    cost_fn: Callable[[float], np.ndarray],
    lam_search_bounds: Tuple[float, float] = (0.1, 200.0),
) -> CalibrationResult:
    """Orchestrates §6.2's three steps: (1) bisection-search lambda for the usage
    target, (2) bootstrap the realized cost distribution at that lambda to check
    overshoot probability, (3) warn if the calibrated lambda ended up below
    lambda* (a red flag that cost is being systematically underestimated, since
    prediction error should push calibration to be MORE conservative than the
    oracle-fit lambda*, not less)."""
    lo, hi = lam_search_bounds
    lam = bisection_calibrate_lambda(usage_rate_fn, lo, hi, tier_cfg.usage_target)

    lambda_star = tier_cfg.lambda_star_total if token_accounting == "total" else tier_cfg.lambda_star_per_call
    if lam < lambda_star:
        import warnings

        warnings.warn(
            f"[{tier_cfg.name}] calibrated lambda ({lam:.3f}) is SMALLER than the "
            f"oracle-fit lambda* ({lambda_star:.3f}) for token_accounting="
            f"{token_accounting!r}. Per §6, the calibrated lambda should end up "
            f"larger than lambda* because predictions are noisier than the oracle "
            f"ground truth -- a smaller value suggests cost is being "
            f"systematically underestimated. Investigate before submitting.",
            stacklevel=2,
        )

    usage_rate = usage_rate_fn(lam)
    costs = cost_fn(lam)
    budget = tier_cfg.budget_ratio  # in the same units as costs, per caller's cost_fn
    overshoot_prob = bootstrap_overshoot_probability(costs, budget)
    p90_coverage = float((costs <= budget).mean())

    return CalibrationResult(
        lam=lam, usage_rate=usage_rate, overshoot_prob=overshoot_prob, p90_coverage=p90_coverage
    )
