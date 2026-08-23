"""Cost estimation and the decision rule (§2, §5.5)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from router.config import MODELS


def estimate_total_cost(k_m: float, in_hat: float, out_hat: float) -> float:
    """c_m(q) = k_m * (in_hat + 4 * out_hat), fixed cost = 0 (§1.1)."""
    return k_m * (in_hat + 4.0 * out_hat)


def compute_rho(p_hat_m: float, p_hat_light: float, cost_m: float, cost_light: float) -> float:
    """rho_m(q) from §2: accuracy gain over light per unit of extra cost over
    light. Analysis/diagnostics use only -- not part of the decision rule itself.
    Returns None-like (float('nan')) when the denominator is ~0 to avoid a
    division-by-zero surprising a caller (§2 notes the denominator is 0 for ~30%
    of queries when comparing against a model with identical cost)."""
    denom = cost_m - cost_light
    if denom == 0:
        return float("nan")
    return (p_hat_m - p_hat_light) / denom


def decide(
    p_hat: Dict[str, float],
    cost_hat: Dict[str, float],
    lam: float,
    model_order: Sequence[str] = MODELS,
) -> str:
    """argmax_m (p_hat_m - lam * cost_hat_m), §5.5.

    Utility is rounded to 9 decimals before comparison so floating-point noise
    can't flip an argmax tie, and ties are broken by a fixed model_order (never by
    dict/insertion order, which is not guaranteed stable across runs/versions).
    """
    utilities = {m: p_hat[m] - lam * cost_hat[m] for m in model_order}
    return min(model_order, key=lambda m: (-round(utilities[m], 9), model_order.index(m)))


def decide_batch(rows: Iterable[Dict], lam: float, model_order: Sequence[str] = MODELS) -> List[str]:
    """Decide for each row independently via a plain Python loop.

    Deliberately NOT vectorized: a vectorized batch implementation risks
    accidentally introducing a batch-relative computation (e.g. normalizing by a
    batch mean/std) that would violate the "decision depends only on this query"
    invariant validate.py's B4 check enforces. Each row here is a dict with
    "p_hat" and "cost_hat" sub-dicts, e.g. {"p_hat": {...}, "cost_hat": {...}}.
    """
    return [decide(row["p_hat"], row["cost_hat"], lam, model_order) for row in rows]
