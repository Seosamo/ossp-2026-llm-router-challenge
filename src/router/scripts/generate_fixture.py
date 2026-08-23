"""CLI wrapper around synthetic_data.write_fixture.

Run from the repo root (parent of router/):
    python -m router.scripts.generate_fixture --out-dir router/artifacts/fixture
"""

from __future__ import annotations

import argparse
from pathlib import Path

from router.synthetic_data import write_fixture


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic train/dev/prompts fixture.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    write_fixture(args.out_dir, n_queries=args.n_queries, seed=args.seed)
    print(f"wrote fixture to {args.out_dir}")


if __name__ == "__main__":
    main()
