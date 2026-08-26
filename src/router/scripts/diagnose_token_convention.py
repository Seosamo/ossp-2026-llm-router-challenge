# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""§10.2 diagnostic CLI: prints, per model, the ratio of mean output_tokens
between num_generations=4 and num_generations=2 groups.

ratio ~= 2.0 -> "total" token accounting; ratio ~= 1.0 -> "per_call".

Does NOT write to config.py automatically -- a human should confirm the result
against the competition's cost-accounting docs and set config.TOKEN_ACCOUNTING
explicitly.

Run from the repo root:
    python -m router.scripts.diagnose_token_convention --outcomes-path <path>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from router.calibration import diagnose_token_convention
from router.config import MODELS
from router.schema import load_train


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose total vs per_call token accounting (§10.2).")
    parser.add_argument("--outcomes-path", required=True, type=Path)
    args = parser.parse_args()

    df = load_train(args.outcomes_path)
    ratios = diagnose_token_convention(df, MODELS)

    for model, ratio in ratios.items():
        convention = "total" if abs(ratio - 2.0) < abs(ratio - 1.0) else "per_call"
        print(f"{model}: ratio={ratio:.3f} -> looks like {convention!r}")

    print(
        "\nCross-check against the competition's public cost-accounting docs before "
        "setting config.TOKEN_ACCOUNTING -- this script does not modify config.py."
    )


if __name__ == "__main__":
    main()
