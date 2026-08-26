# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Calibrates lambda for one tier against Dev (§6.2), writes the result as JSON.

Costs are expressed as a ratio to that row's own ax31-light cost estimate, matching
how tier budgets are defined in §3.1 ("budget_ratio" is relative to light).

Run from the repo root:
    python -m router.scripts.run_calibration --dev-path <p> --tier Balanced \\
        --pipeline-dir <p> --token-accounting total [--prompts-path <p>]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from router.calibration import calibrate_tier
from router.config import ARTIFACTS_DIR, TIERS
from router.decision import decide
from router.pipeline import RouterPipeline
from router.schema import join_prompts, load_dev, load_prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate lambda for a tier against Dev (§6.2).")
    parser.add_argument("--dev-path", required=True, type=Path)
    parser.add_argument("--prompts-path", type=Path, default=None)
    parser.add_argument("--pipeline-dir", required=True, type=Path)
    parser.add_argument("--tier", required=True, choices=list(TIERS.keys()))
    parser.add_argument("--token-accounting", choices=["total", "per_call"], default="total")
    parser.add_argument("--out-path", type=Path, default=None)
    args = parser.parse_args()

    dev_df = load_dev(args.dev_path)
    if args.prompts_path is not None:
        prompts_df = load_prompts(args.prompts_path)
        dev_df = join_prompts(dev_df, prompts_df)

    pipeline = RouterPipeline.load(args.pipeline_dir)
    estimates = pipeline.estimate_texts(dev_df["text"].tolist())
    tier_cfg = TIERS[args.tier]

    def usage_rate_fn(lam: float) -> float:
        decisions = [decide(e["p_hat"], e["cost_hat"], lam) for e in estimates]
        return float(np.mean([d != tier_cfg.baseline_model for d in decisions]))

    def cost_fn(lam: float) -> np.ndarray:
        ratios = []
        for e in estimates:
            decision = decide(e["p_hat"], e["cost_hat"], lam)
            ratios.append(e["cost_hat"][decision] / e["cost_hat"]["ax31-light"])
        return np.array(ratios)

    result = calibrate_tier(
        dev_predictions=dev_df,
        tier_cfg=tier_cfg,
        token_accounting=args.token_accounting,
        usage_rate_fn=usage_rate_fn,
        cost_fn=cost_fn,
    )

    print(result)

    out_path = args.out_path or (ARTIFACTS_DIR / f"calibration_{args.tier}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(result), indent=2))
    print(f"wrote calibration result to {out_path}")


if __name__ == "__main__":
    main()
