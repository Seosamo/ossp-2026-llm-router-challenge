"""Adapter between the OSSP 2026 challenge's nested-JSON wire protocol and
router/schema.py's wide-DataFrame format.

router/ was built and validated against synthetic fixtures before real
competition data existed (see router/README.md). The competition
(ossp-2026-llm-router-challenge, a sibling directory) ships real Train/Dev
data as nested JSON (InputBatch/OutcomeBatch, defined in
ossp_router.protocol) rather than the wide outcomes table + separate prompts
table router/schema.py expects. This module is the one-time offline
conversion step; nothing at inference time depends on it (see
router/scripts/... invoked from ossp_router.learned_router, which reads
InputBatch directly).

Requires ossp_router to be importable (run with the ossp repo's `src/` on
PYTHONPATH, matching that repo's own documented convention), since it reuses
ossp_router.protocol's strict parsers and ossp_router.heuristic.episode_text
so the train-time text extraction is byte-identical to the inference-time
extraction in learned_router.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from router.config import MODELS


def _import_ossp_router():
    try:
        from ossp_router import heuristic, protocol
    except ImportError as exc:
        raise ImportError(
            "ossp_router is not importable. Run with the ossp repo's src/ on "
            "PYTHONPATH, e.g.:\n"
            "  PYTHONPATH=<ossp-repo>/src;. python -m router.ossp_adapter ..."
        ) from exc
    return heuristic, protocol


def load_ossp_input(path: Path):
    _, protocol = _import_ossp_router()
    return protocol.load_input(Path(path))


def load_ossp_outcomes(path: Path):
    _, protocol = _import_ossp_router()
    return protocol.load_outcomes(Path(path))


def input_batch_to_prompts_df(input_batch) -> pd.DataFrame:
    """episode_id, challenge_id, text -- router/schema.py's REQUIRED_PROMPT_COLUMNS.

    Text extraction reuses ossp_router.heuristic.episode_text verbatim so that
    a prompt vs. messages episode is joined into `text` the exact same way here
    (train time) and in learned_router.py (inference time). A divergence here
    would silently skew every downstream feature.
    """
    heuristic, _ = _import_ossp_router()
    rows = [
        {
            "episode_id": episode.episode_id,
            "challenge_id": input_batch.challenge_id,
            "text": heuristic.episode_text(episode),
        }
        for episode in input_batch.episodes
    ]
    return pd.DataFrame.from_records(rows)


def outcome_batch_to_wide_df(input_batch, outcome_batch, models=MODELS) -> pd.DataFrame:
    """episode_id, challenge_id, split, then per model:
    num_generations__{m}, score__{m}, output_tokens__{m}, input_tokens__{m}.

    Matches router/schema.py's REQUIRED_OUTCOME_* column convention exactly.
    """
    if outcome_batch.challenge_id != input_batch.challenge_id:
        raise ValueError(
            f"challenge_id mismatch: input={input_batch.challenge_id!r} "
            f"outcomes={outcome_batch.challenge_id!r}"
        )
    by_episode: dict = {}
    for outcome in outcome_batch.outcomes:
        by_episode.setdefault(outcome.episode_id, {})[outcome.model_id] = outcome

    rows = []
    for episode in input_batch.episodes:
        per_model = by_episode.get(episode.episode_id)
        if per_model is None or set(per_model) != set(models):
            raise ValueError(
                f"episode {episode.episode_id!r} is missing outcomes for one or "
                f"more of {models}"
            )
        row = {
            "episode_id": episode.episode_id,
            "challenge_id": input_batch.challenge_id,
            "split": outcome_batch.split,
        }
        for m in models:
            outcome = per_model[m]
            row[f"num_generations__{m}"] = outcome.num_generations
            row[f"score__{m}"] = float(outcome.score)
            row[f"output_tokens__{m}"] = outcome.output_tokens
            row[f"input_tokens__{m}"] = outcome.input_tokens
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def convert_split(input_path: Path, outcomes_path: Path, out_dir: Path, split: str) -> None:
    input_batch = load_ossp_input(input_path)
    outcome_batch = load_ossp_outcomes(outcomes_path)
    if input_batch.split != split or outcome_batch.split != split:
        raise ValueError(
            f"expected split={split!r}, got input.split={input_batch.split!r}, "
            f"outcomes.split={outcome_batch.split!r}"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts_df = input_batch_to_prompts_df(input_batch)
    prompts_path = out_dir / f"prompts_{split}.parquet"
    prompts_df.to_parquet(prompts_path, index=False)

    wide_df = outcome_batch_to_wide_df(input_batch, outcome_batch, MODELS)
    wide_path = out_dir / f"{split}.parquet"
    wide_df.to_parquet(wide_path, index=False)

    print(f"{split}: {len(input_batch.episodes)} episodes -> {wide_path}, {prompts_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert OSSP challenge input/outcomes JSON into router's wide-DataFrame format."
    )
    parser.add_argument("--input", required=True, type=Path, help="OSSP InputBatch JSON (e.g. data/materialized/train/inputs.json)")
    parser.add_argument("--outcomes", required=True, type=Path, help="OSSP OutcomeBatch JSON (e.g. data/train/outcomes.json)")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "dev"])
    args = parser.parse_args()

    convert_split(args.input, args.outcomes, args.out_dir, args.split)


if __name__ == "__main__":
    main()
