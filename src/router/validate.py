"""First-class deliverable (§8.1): run before every submission.

Checks:
    A  schema           -- missing/extra columns, duplicate/missing IDs, score range
    B1 perturbation      -- decision-relevant output changes when input text changes
    B2 episode_id align  -- decisions stay correctly attached to episode_id under
                             row reordering (no positional-index bugs)
    B3 metadata isolation -- decisions do NOT change when challenge_id/split values
                             are perturbed while text is held fixed (no metadata
                             leaking into the decision -- see §4's "no batch
                             stats/order/ID" contract)
    B4 batch invariance  -- (the check the doc calls out hardest to guard) running
                             predict() one row at a time must be bit-identical to
                             running predict_batch() on the full batch
    C  repeatability     -- same input run twice gives the same decision

test_validator.py is the reverse test suite: it checks that THIS module actually
discriminates a well-behaved pipeline from intentionally-broken fixtures, rather
than trivially passing everything.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from router.config import MODELS
from router.pipeline import RouterPipeline
from router.schema import validate_schema


@dataclass
class ValidationResult:
    name: str
    passed: bool
    details: str


def check_schema(df: pd.DataFrame) -> ValidationResult:
    problems = validate_schema(df, MODELS)
    return ValidationResult(name="A_schema", passed=not problems, details="; ".join(problems))


def check_perturbation_sensitivity(pipeline: RouterPipeline, sample_texts: List[str]) -> ValidationResult:
    changed = 0
    for text in sample_texts:
        original = pipeline.predict(text, tier="Balanced")
        perturbed_text = text + " 추가적인 세부 설명을 덧붙여줘 please add more detail step by step"
        perturbed = pipeline.predict(perturbed_text, tier="Balanced")
        if original["p_hat"] != perturbed["p_hat"] or original["cost_hat"] != perturbed["cost_hat"]:
            changed += 1
    passed = changed >= max(1, len(sample_texts) // 2)
    return ValidationResult(
        name="B1_perturbation",
        passed=passed,
        details=f"{changed}/{len(sample_texts)} samples changed output after perturbation",
    )


def check_episode_id_dependence(pipeline: RouterPipeline, df: pd.DataFrame) -> ValidationResult:
    shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    original_result = pipeline.predict_batch(df, tier="Balanced").set_index("episode_id")
    shuffled_result = pipeline.predict_batch(shuffled, tier="Balanced").set_index("episode_id")
    aligned = original_result.reindex(shuffled_result.index)
    mismatched = (aligned["decision"].to_numpy() != shuffled_result["decision"].to_numpy()).sum()
    passed = mismatched == 0
    return ValidationResult(
        name="B2_episode_id_alignment",
        passed=passed,
        details=f"{mismatched} decisions misaligned with episode_id after row reordering",
    )


def check_challenge_split_dependence(pipeline: RouterPipeline, df: pd.DataFrame) -> ValidationResult:
    rng = np.random.default_rng(2)
    perturbed = df.copy()
    if "challenge_id" in perturbed.columns:
        perturbed["challenge_id"] = rng.permutation(perturbed["challenge_id"].to_numpy())
    if "split" in perturbed.columns:
        perturbed["split"] = rng.choice(["train", "dev"], size=len(perturbed))

    original_result = pipeline.predict_batch(df, tier="Balanced")
    perturbed_result = pipeline.predict_batch(perturbed, tier="Balanced")
    mismatched = (original_result["decision"].to_numpy() != perturbed_result["decision"].to_numpy()).sum()
    passed = mismatched == 0
    return ValidationResult(
        name="B3_metadata_isolation",
        passed=passed,
        details=f"{mismatched} decisions changed after perturbing challenge_id/split only",
    )


def check_batch_invariance(pipeline: RouterPipeline, df: pd.DataFrame) -> ValidationResult:
    try:
        batch_result = pipeline.predict_batch(df, tier="Balanced")
    except Exception as exc:  # a pipeline whose batch path is broken enough to
        # crash is exactly as much a B4 failure as one that silently disagrees.
        return ValidationResult(
            name="B4_batch_invariance", passed=False, details=f"predict_batch raised: {exc!r}"
        )

    mismatched = 0
    for _, row in df.iterrows():
        try:
            single_decision = pipeline.predict(row["text"], tier="Balanced")["decision"]
        except Exception as exc:
            single_decision = f"<error: {exc!r}>"
        matches = batch_result.loc[batch_result["episode_id"] == row["episode_id"], "decision"]
        batch_decision = matches.iloc[0] if len(matches) else "<missing from batch output>"
        if single_decision != batch_decision:
            mismatched += 1
    passed = mismatched == 0
    return ValidationResult(
        name="B4_batch_invariance",
        passed=passed,
        details=f"{mismatched}/{len(df)} rows differ between single-row and full-batch prediction",
    )


def check_repeatability(pipeline: RouterPipeline, df: pd.DataFrame, n_runs: int = 2) -> ValidationResult:
    results = [pipeline.predict_batch(df, tier="Balanced")["decision"].tolist() for _ in range(n_runs)]
    passed = all(r == results[0] for r in results[1:])
    return ValidationResult(
        name="C_repeatability", passed=passed, details=f"{n_runs} runs identical: {passed}"
    )


def run_all_checks(pipeline: RouterPipeline, df: pd.DataFrame) -> List[ValidationResult]:
    sample_texts = df["text"].head(min(5, len(df))).tolist()
    return [
        check_schema(df),
        check_perturbation_sensitivity(pipeline, sample_texts),
        check_episode_id_dependence(pipeline, df),
        check_challenge_split_dependence(pipeline, df),
        check_batch_invariance(pipeline, df),
        check_repeatability(pipeline, df),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a router pipeline / submission before submit.")
    parser.add_argument("--pipeline-dir", required=True, type=Path)
    parser.add_argument("--dev-path", required=True, type=Path)
    parser.add_argument("--prompts-path", type=Path, default=None)
    args = parser.parse_args()

    from router.schema import join_prompts, load_dev, load_prompts

    dev_df = load_dev(args.dev_path)
    if args.prompts_path is not None:
        prompts_df = load_prompts(args.prompts_path)
        dev_df = join_prompts(dev_df, prompts_df)

    pipeline = RouterPipeline.load(args.pipeline_dir)
    results = run_all_checks(pipeline, dev_df)

    all_passed = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name}: {r.details}")
        all_passed = all_passed and r.passed

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
