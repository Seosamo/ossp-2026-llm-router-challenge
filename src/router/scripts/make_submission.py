"""Runs predict_batch on Dev, writes a submission file, then automatically runs
validate.py's full check suite before reporting success (§8.1: run before every
submission).

Run from the repo root:
    python -m router.scripts.make_submission --dev-path <p> --pipeline-dir <p> \\
        --tier Balanced --out-path <p> [--prompts-path <p>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from router.pipeline import RouterPipeline
from router.schema import join_prompts, load_dev, load_prompts
from router.validate import run_all_checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and validate a submission.")
    parser.add_argument("--dev-path", required=True, type=Path)
    parser.add_argument("--prompts-path", type=Path, default=None)
    parser.add_argument("--pipeline-dir", required=True, type=Path)
    parser.add_argument("--tier", required=True, choices=["Fast", "Balanced", "Premium"])
    parser.add_argument("--out-path", required=True, type=Path)
    args = parser.parse_args()

    dev_df = load_dev(args.dev_path)
    if args.prompts_path is not None:
        prompts_df = load_prompts(args.prompts_path)
        dev_df = join_prompts(dev_df, prompts_df)

    pipeline = RouterPipeline.load(args.pipeline_dir)
    submission = pipeline.predict_batch(dev_df, tier=args.tier)
    submission = submission.rename(columns={"decision": "model_id"})

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_json(args.out_path, orient="records", force_ascii=False)
    print(f"wrote submission to {args.out_path}")

    print("\nrunning validate.py checks before declaring success...")
    results = run_all_checks(pipeline, dev_df)
    all_passed = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name}: {r.details}")
        all_passed = all_passed and r.passed

    if not all_passed:
        print("\nvalidation FAILED -- do not submit.")
        return 1
    print("\nall validation checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
