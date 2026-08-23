# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Learned prompt-only router: loads a pre-trained `router.RouterPipeline`
(feature extractor + per-model win-probability / output-token / input-token
LightGBM models, see router/pipeline.py) and applies its decision rule
(router/decision.py::decide) per episode.

Provides the same CLI contract as `ossp_router.heuristic.main` (frozen by
docs/RUNTIME.md as `router-run --input ... --tier ... --output ...`), so it
is a drop-in replacement wired via container/entrypoint.py and setup.cfg's
`router-run` entry point. `heuristic.py` itself is left untouched as the
reference baseline.

Deliberately does not use `RouterPipeline.predict_batch` (which iterates via
pandas.DataFrame.iterrows): this keeps pandas/pyarrow out of the runtime
image entirely -- they are only needed offline, for router/scripts/train.py
and router/ossp_adapter.py. The decision rule itself is per-episode already
(router/decision.py::decide), so looping here is behavior-equivalent, just
without the pandas dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

from router.decision import decide
from router.pipeline import RouterPipeline

from .heuristic import episode_text, write_submission_atomic
from .protocol import (
    Decision,
    InputBatch,
    ProtocolError,
    Submission,
    load_bundled_policy,
    load_input,
    load_policy,
    parse_submission,
    submission_to_dict,
)

# ossp_router.protocol.TIERS is ("fast", "balanced", "premium"); router's own
# TierConfig table (router/config.py::TIERS) is keyed by the capitalized
# names used throughout router/decision.py, router/calibration.py, etc.
# Keeping this mapping as an explicit, named constant (rather than e.g.
# `tier.capitalize()`) means a typo or a future tier rename fails loudly
# instead of silently mis-routing a whole tier's budget.
OSSP_TIER_TO_ROUTER_TIER = {
    "fast": "Fast",
    "balanced": "Balanced",
    "premium": "Premium",
}

DEFAULT_PIPELINE_DIR = Path(os.environ.get("ROUTER_PIPELINE_DIR", "/opt/router/artifacts/router-model"))
CALIBRATED_LAMBDA_FILENAME = "calibrated_lambda.json"


def load_calibrated_lambdas(pipeline_dir: Path) -> Optional[Dict[str, float]]:
    """Load per-tier lambdas produced by
    router/scripts/calibrate_against_official_scorer.py, if present.

    router/config.py's built-in TIERS.lambda_star_total/per_call are
    theoretical pre-real-data estimates (see router/README.md's "미결 항목").
    Once a pipeline has been calibrated against real Dev through the actual
    OSSP scorer (which is what grades the real submission), that calibrated
    value should always win over the stale built-in constant.
    """
    path = pipeline_dir / CALIBRATED_LAMBDA_FILENAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def make_submission(
    inputs: InputBatch,
    pipeline: RouterPipeline,
    policy,
    tier: str,
    calibrated_lambdas: Optional[Dict[str, float]] = None,
) -> Submission:
    """Create one complete v1 submission for a single tier.

    Mirrors ossp_router.heuristic.make_submission's structure/validation so
    both baseline and learned routers produce protocol-identical output.
    """

    if inputs.schema_version != policy.schema_version:
        raise ProtocolError("입력과 정책의 schema_version이 일치하지 않습니다.")
    if tier not in OSSP_TIER_TO_ROUTER_TIER:
        raise ProtocolError(f"알 수 없는 tier: {tier}")
    router_tier = OSSP_TIER_TO_ROUTER_TIER[tier]

    # Batched via pipeline.estimate_texts (one feature-extraction/model call
    # over every episode's text) rather than looping one episode at a time --
    # the per-episode path re-runs the embedding backend once per call, which
    # does not fit the container's 90-seconds-per-tier budget
    # (docs/RUNTIME.md) at real batch sizes. See RouterPipeline.estimate_texts's
    # docstring for why this is numerically identical to the per-row path,
    # not just faster.
    texts = [episode_text(episode) for episode in inputs.episodes]
    estimates = pipeline.estimate_texts(texts)

    if calibrated_lambdas is not None and router_tier in calibrated_lambdas:
        lam = calibrated_lambdas[router_tier]
        model_ids = [decide(e["p_hat"], e["cost_hat"], lam) for e in estimates]
    else:
        model_ids = pipeline.predict_texts(texts, tier=router_tier)

    decisions = [
        Decision(episode.episode_id, model_id)
        for episode, model_id in zip(inputs.episodes, model_ids)
    ]

    submission = Submission(
        schema_version=inputs.schema_version,
        challenge_id=inputs.challenge_id,
        policy_id=policy.policy_id,
        split=inputs.split,
        tier=tier,
        decisions=tuple(decisions),
    )
    # Round-trip through the strict v1 parser, exactly like heuristic.py does,
    # so a malformed decision (e.g. an unknown model_id) fails loudly here
    # rather than producing a submission.json that fails validation later.
    return parse_submission(submission_to_dict(submission))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="router-run",
        description="학습된 router.RouterPipeline으로 한 등급의 선택 결과를 만듭니다.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tier", choices=tuple(OSSP_TIER_TO_ROUTER_TIER), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=DEFAULT_PIPELINE_DIR,
        help="RouterPipeline.save() 아티팩트 디렉터리 (기본: ROUTER_PIPELINE_DIR 환경변수 또는 이미지에 구운 경로)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inputs = load_input(args.input)
        policy = load_policy(args.policy) if args.policy is not None else load_bundled_policy()
        pipeline = RouterPipeline.load(args.pipeline_dir)
        calibrated_lambdas = load_calibrated_lambdas(args.pipeline_dir)
        submission = make_submission(inputs, pipeline, policy, args.tier, calibrated_lambdas)
        write_submission_atomic(args.output, submission)
    except (OSError, ProtocolError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    print(f"OK: {args.tier} 제출 파일을 생성했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
