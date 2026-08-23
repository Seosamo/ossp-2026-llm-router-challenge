"""Calibrate lambda per tier directly against ossp_router's own official Decimal
scorer on real Dev, instead of trusting router/calibration.py's internal
usage-rate/bootstrap proxy in isolation.

Why: the competition scores a submission by its AGGREGATE cost ratio
(sum(selected cost) / sum(all-light cost) <= budget_multiplier, see
ossp_router/scoring.py), not by any per-query cost distribution. The
competition's own hash-regex baseline passed public Dev at 3.985/4.0 and then
exceeded budget (~4.2/4.0) on the private eval set (baselines/README.md) --
i.e. a thin, distribution-based safety margin does not reliably survive the
train/dev-to-eval generalization gap. This script instead bisects lambda so
that the REAL aggregate budget_ratio, computed by the exact scorer that will
grade the submission, lands at a target fraction of the tier's budget
(default 0.90x), leaving explicit headroom for that gap.

Requires ossp_router on PYTHONPATH (see router/ossp_adapter.py's docstring
for the same requirement and rationale).

Run from the repo root:
    python -m router.scripts.calibrate_against_official_scorer \
        --dev-input <ossp inputs.json> --dev-outcomes <ossp outcomes.json> \
        --pipeline-dir <trained pipeline dir> \
        --out-path <pipeline dir>/calibrated_lambda.json \
        [--safety-factor 0.90]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from router.config import MODELS, TIERS
from router.decision import decide
from router.pipeline import RouterPipeline

# router's capitalized tier names -> ossp's lowercase tier names + budget policy.
ROUTER_TIER_TO_OSSP_TIER = {"Fast": "fast", "Balanced": "balanced", "Premium": "premium"}


def _import_ossp_router():
    from ossp_router import heuristic, protocol, scoring

    return heuristic, protocol, scoring


def _build_submission_dict(input_batch, policy, ossp_tier: str, episode_ids: List[str], decisions: List[str]):
    return {
        "schema_version": input_batch.schema_version,
        "challenge_id": input_batch.challenge_id,
        "policy_id": policy.policy_id,
        "split": input_batch.split,
        "tier": ossp_tier,
        "decisions": [
            {"episode_id": episode_id, "model_id": model_id}
            for episode_id, model_id in zip(episode_ids, decisions)
        ],
    }


def bisect_lambda_for_budget(
    *,
    estimates: List[dict],
    episode_ids: List[str],
    input_batch,
    outcome_batch,
    policy,
    ossp_tier: str,
    target_ratio: float,
    lo: float = 0.0,
    hi: float = 500.0,
    iterations: int = 40,
) -> Dict[str, float]:
    """Bisect for the SMALLEST lambda whose real (official-scorer) budget_ratio
    is <= target_ratio. Smaller lambda -> more aggressive (upgrades more
    queries) -> higher cost; larger lambda -> more conservative -> lower cost.
    This mirrors bisection_calibrate_lambda's monotonicity assumption
    (router/calibration.py) but bisects on the actual scored ratio."""

    _, protocol, scoring = _import_ossp_router()

    def ratio_at(lam: float) -> float:
        decisions = [decide(e["p_hat"], e["cost_hat"], lam, MODELS) for e in estimates]
        submission_dict = _build_submission_dict(input_batch, policy, ossp_tier, episode_ids, decisions)
        submission = protocol.parse_submission(submission_dict)
        report = scoring._score_tier(
            inputs=input_batch,
            submission=submission,
            outcome_by_key=scoring._outcome_index(input_batch, outcome_batch, policy),
            policy=policy,
            policy_digest=scoring.policy_sha256(policy),
        )
        return float(report["budget_ratio"]), float(report["quality_score"])

    # hi must be conservative enough to already satisfy the target (all-light
    # cost ratio is 1.0, always <= any budget_multiplier >= 1).
    hi_ratio, _ = ratio_at(hi)
    if hi_ratio > target_ratio:
        raise RuntimeError(
            f"even lambda={hi} (very conservative) exceeds target_ratio={target_ratio}; "
            "widen --lambda-hi"
        )

    best_lambda = hi
    for _ in range(iterations):
        mid = (lo + hi) / 2
        mid_ratio, _ = ratio_at(mid)
        if mid_ratio <= target_ratio:
            best_lambda = mid
            hi = mid
        else:
            lo = mid

    final_ratio, final_quality = ratio_at(best_lambda)
    return {"lambda": best_lambda, "budget_ratio": final_ratio, "quality_score": final_quality}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-input", required=True, type=Path, help="ossp materialized Dev inputs.json")
    parser.add_argument("--dev-outcomes", required=True, type=Path, help="ossp Dev outcomes.json")
    parser.add_argument("--pipeline-dir", required=True, type=Path)
    parser.add_argument("--out-path", required=True, type=Path)
    parser.add_argument(
        "--safety-factor",
        type=float,
        default=0.90,
        help="target budget_ratio as a fraction of each tier's budget_multiplier",
    )
    args = parser.parse_args()

    _, protocol, _ = _import_ossp_router()
    input_batch = protocol.load_input(args.dev_input)
    outcome_batch = protocol.load_outcomes(args.dev_outcomes)
    policy = protocol.load_bundled_policy()

    from .. import ossp_adapter

    prompts_df = ossp_adapter.input_batch_to_prompts_df(input_batch)
    episode_ids = prompts_df["episode_id"].tolist()
    texts = prompts_df["text"].tolist()

    pipeline = RouterPipeline.load(args.pipeline_dir)
    estimates = pipeline.estimate_texts(texts)

    results = {}
    for router_tier, ossp_tier in ROUTER_TIER_TO_OSSP_TIER.items():
        budget_multiplier = float(policy.tiers[ossp_tier].budget_multiplier)
        target_ratio = budget_multiplier * args.safety_factor
        result = bisect_lambda_for_budget(
            estimates=estimates,
            episode_ids=episode_ids,
            input_batch=input_batch,
            outcome_batch=outcome_batch,
            policy=policy,
            ossp_tier=ossp_tier,
            target_ratio=target_ratio,
        )
        results[router_tier] = result
        print(
            f"{router_tier} ({ossp_tier}): lambda={result['lambda']:.4f} "
            f"budget_ratio={result['budget_ratio']:.4f} (target<={target_ratio:.4f}, "
            f"multiplier={budget_multiplier}) quality_score={result['quality_score']:.6f}"
        )

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(
        json.dumps({tier: r["lambda"] for tier, r in results.items()}, indent=2)
    )
    print(f"\nwrote calibrated lambdas to {args.out_path}")


if __name__ == "__main__":
    main()
