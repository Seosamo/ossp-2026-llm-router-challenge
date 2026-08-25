# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# Temporary diagnostic: reuses tools/check_runtime.py's exact working
# plumbing (combined Train+Dev input, home-directory temp dirs -- avoids the
# macOS Documents/TCC bind-mount permission issue entirely) but with the 90s
# per-tier limit raised, to measure how long a tier actually needs to
# complete. Not part of the submission; delete after use.
#
# Run from the repo root: PYTHONPATH=src python3 -u tools/measure_uncapped.py --image router:check --tier balanced --seconds 900
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

import check_runtime  # tools/check_runtime.py -- this script lives next to it
from ossp_router.runtime import PHASE_C_CANDIDATE_LIMITS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--tier", required=True, choices=["fast", "balanced", "premium"])
    parser.add_argument("--seconds", type=int, default=900)
    args = parser.parse_args()

    check_runtime.PHASE_C_CANDIDATE_LIMITS = dataclasses.replace(
        PHASE_C_CANDIDATE_LIMITS, wall_time_seconds=args.seconds
    )

    print(f"running {args.tier} with a {args.seconds}s cap (this may take a while, no periodic output)...")
    report = check_runtime.check_image(
        docker=check_runtime._resolve_docker("docker"),
        requested_image=args.image,
        tiers=(args.tier,),
        repetitions=1,
        train_input=check_runtime.DEFAULT_TRAIN_INPUT,
        dev_input=check_runtime.DEFAULT_DEV_INPUT,
        registry=check_runtime.DEFAULT_PUBLIC_DATA_REGISTRY,
    )
    check_runtime._print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
