"""Data contract for outcomes/prompts tables, plus the label-exploding logic that
turns the competition's "num_generations attempts, fractional score" data (§1.2)
into a per-attempt binary classification target (§5.2).

Outcomes table (wide, one row per episode/query):
    episode_id, challenge_id, split,
    then per model m in config.MODELS:
        num_generations__{m}, score__{m}, output_tokens__{m}, input_tokens__{m}

    num_generations__{m} is expected to be identical across all three models for a
    given row (it is a property of the query, not the model) -- validate_schema
    checks this as a consistency guard, matching the exact column-naming
    convention the planning doc's own §10.2 diagnostic snippet uses:
        wide.groupby(f"num_generations__{m}")[f"output_tokens__{m}"].mean()

Prompts table (separate, joined by episode_id -- see §10.3: prompt text is not
currently available in the outcomes file at all):
    episode_id, challenge_id, text
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

REQUIRED_OUTCOME_BASE_COLUMNS = ["episode_id", "challenge_id", "split"]
REQUIRED_OUTCOME_MODEL_SUFFIXES = ["num_generations", "score", "output_tokens", "input_tokens"]
REQUIRED_PROMPT_COLUMNS = ["episode_id", "challenge_id", "text"]


def _model_columns(model: str) -> Dict[str, str]:
    return {suffix: f"{suffix}__{model}" for suffix in REQUIRED_OUTCOME_MODEL_SUFFIXES}


def required_outcome_columns(models: Sequence[str]) -> List[str]:
    cols = list(REQUIRED_OUTCOME_BASE_COLUMNS)
    for m in models:
        cols.extend(_model_columns(m).values())
    return cols


def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".json", ".jsonl"):
        return pd.read_json(path, lines=path.suffix == ".jsonl")
    return pd.read_csv(path)


def load_train(path: Path) -> pd.DataFrame:
    return _read_table(path)


def load_dev(path: Path) -> pd.DataFrame:
    return _read_table(path)


def load_prompts(path: Path) -> pd.DataFrame:
    return _read_table(path)


def join_prompts(outcomes: pd.DataFrame, prompts: pd.DataFrame) -> pd.DataFrame:
    """Left-join prompt text onto the outcomes table by episode_id.

    §10.3: prompt text lives in a separate file from scores/outcomes today, so this
    join is a first-class step of the pipeline, not an afterthought.
    """
    missing = set(REQUIRED_PROMPT_COLUMNS) - set(prompts.columns)
    if missing:
        raise ValueError(f"prompts table missing required columns: {sorted(missing)}")
    return outcomes.merge(
        prompts[["episode_id", "text"]], on="episode_id", how="left", validate="one_to_one"
    )


def validate_schema(df: pd.DataFrame, models: Sequence[str]) -> List[str]:
    """Structural checks (§8.1 test A). Returns a list of problem descriptions;
    empty list means the schema is valid."""
    problems: List[str] = []

    required = required_outcome_columns(models)
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        problems.append(f"missing columns: {missing_cols}")
        return problems  # further checks would just raise KeyError

    if df["episode_id"].isna().any():
        problems.append("episode_id contains missing values")
    dup = df["episode_id"][df["episode_id"].duplicated()].unique().tolist()
    if dup:
        problems.append(f"duplicate episode_id values: {dup[:10]}")

    # num_generations must agree across all three models for a given row -- it is a
    # property of the query, not of the model being scored.
    gen_cols = [f"num_generations__{m}" for m in models]
    gen_values = df[gen_cols].to_numpy()
    inconsistent = np.any(gen_values != gen_values[:, [0]], axis=1)
    if inconsistent.any():
        bad_ids = df.loc[inconsistent, "episode_id"].tolist()
        problems.append(f"num_generations disagrees across models for episode_id: {bad_ids[:10]}")

    for m in models:
        score_col = f"score__{m}"
        out_of_range = ~df[score_col].between(0.0, 1.0)
        if out_of_range.any():
            problems.append(f"{score_col} has values outside [0, 1]")

    return problems


def explode_to_attempts(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Expand each query's fractional score into per-attempt binary rows (§1.2, §5.2).

    A row with num_generations__{model}=n and score__{model}=s becomes
    round(s*n) rows with label=1 and n-round(s*n) rows with label=0, all sharing the
    same episode_id (and therefore the same feature vector once features are joined
    in). This is the fix for §1.2's finding that the target is classification-
    shaped (a handful of discrete pass rates), not a continuous regression target.
    """
    n = df[f"num_generations__{model}"].to_numpy()
    s = df[f"score__{model}"].to_numpy()
    n_correct = np.clip(np.round(s * n).astype(int), 0, n.astype(int))
    n_incorrect = n.astype(int) - n_correct

    rows = []
    for idx, (episode_id, correct, incorrect) in enumerate(
        zip(df["episode_id"].to_numpy(), n_correct, n_incorrect)
    ):
        rows.append((idx, episode_id, correct, 1))
        rows.append((idx, episode_id, incorrect, 0))

    exploded = pd.DataFrame(rows, columns=["_source_row", "episode_id", "_count", "label"])
    exploded = exploded[exploded["_count"] > 0].drop(columns="_count")
    return exploded.merge(
        df.drop(columns=[c for c in df.columns if c == "label"]),
        on="episode_id",
        how="left",
    ).reset_index(drop=True)


def compute_regret_weight(row: pd.Series, models: Sequence[str]) -> float:
    """Per-query training weight w(q) = u_(1)(q) - u_(2)(q) (§1.3, §5.2 "Regret 가중").

    Deliberately computed from GROUND-TRUTH scores (score__{m}), not from any
    classifier's own p_hat -- using a not-yet-trained model's predictions as its own
    training weight would be circular. The planning doc does not spell this choice
    out explicitly; it is recorded here because it matters for anyone retraining.

    Ties (all models agree) contribute ~0 weight, concentrating learning capacity on
    the ~26-36% of queries where the routing decision is actually consequential.
    """
    utilities = sorted((row[f"score__{m}"] for m in models), reverse=True)
    return float(utilities[0] - utilities[1])


def compute_regret_weights(df: pd.DataFrame, models: Sequence[str]) -> pd.Series:
    return df.apply(lambda row: compute_regret_weight(row, models), axis=1)
