"""Trains a RouterPipeline on Train data and saves it.

Run from the repo root:
    python -m router.scripts.train --train-path <p> --prompts-path <p> --out-dir <p>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from router.pipeline import RouterPipeline
from router.schema import load_prompts, load_train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the router pipeline.")
    parser.add_argument("--train-path", required=True, type=Path)
    parser.add_argument("--prompts-path", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    train_df = load_train(args.train_path)
    prompts_df = load_prompts(args.prompts_path)

    pipeline = RouterPipeline()
    pipeline.fit(train_df, prompts_df)
    pipeline.save(args.out_dir)
    print(f"trained pipeline saved to {args.out_dir}")


if __name__ == "__main__":
    main()
