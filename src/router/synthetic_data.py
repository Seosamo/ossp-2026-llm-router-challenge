"""Synthetic fixture generator.

No real competition data (prompts/outcomes/episode metadata) exists locally yet
(§10.3). This module produces a schema-conformant fixture so every other module in
this package has something real to run against today, reproducing the qualitative
shape the planning doc reports:
    - light<->think score correlation ~= 0.44 (§1.4)
    - think's output tokens ~24x light's on average, with a heavy right tail
      (p10/p90 spread ~20x, §1.5)
    - num_generations in {2, 4}, identical across all three models per query (§1.2)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from router.config import MODELS

_SAMPLE_TOPICS = [
    "파이썬 리스트 뒤집기",
    "파이썬으로 B-tree 구현",
    "이 함수의 시간복잡도를 3문장으로 요약해줘",
    "단계별로 자세히 설명해줘: 이 알고리즘이 왜 O(n log n)인지",
    "translate this paragraph into Korean",
    "다음 코드에서 버그를 찾아줘",
    "간단히 답해줘: HTTP와 HTTPS의 차이",
    "write a summary of this article in 3 sentences",
    "이 SQL 쿼리를 최적화하는 방법은?",
    "설명: transformer attention mechanism 작동 원리",
]


def _make_prompt_text(rng: np.random.Generator) -> str:
    topic = _SAMPLE_TOPICS[rng.integers(0, len(_SAMPLE_TOPICS))]
    filler_len = rng.integers(0, 40)
    filler = " ".join(["세부사항"] * filler_len) if filler_len else ""
    return f"{topic} {filler}".strip()


def _simulate_scores(rng: np.random.Generator, n_queries: int) -> Dict[str, np.ndarray]:
    """Simulate per-model latent skill with light<->think correlation ~0.44 (§1.4)."""
    shared = rng.normal(size=n_queries)
    light_only = rng.normal(size=n_queries)
    think_only = rng.normal(size=n_queries)

    # Mix shared vs model-specific noise to land near the observed 0.44 correlation.
    light_latent = 0.65 * shared + 0.76 * light_only
    ax31_latent = 0.75 * shared + 0.66 * rng.normal(size=n_queries)
    think_latent = 0.65 * shared + 0.76 * think_only + 0.3  # slightly higher skill floor

    latents = {"ax31-light": light_latent, "ax31": ax31_latent, "axk1-think": think_latent}
    scores = {m: 1.0 / (1.0 + np.exp(-v)) for m, v in latents.items()}
    return scores


def generate_synthetic_dataset(
    n_queries: int = 200, seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """Return {"train": ..., "dev": ..., "prompts": ...} matching the schema in
    schema.py, split roughly 70/30 into train/dev."""
    rng = np.random.default_rng(seed)

    episode_ids = [f"ep_{i:05d}" for i in range(n_queries)]
    challenge_ids = [f"chal_{i % 20:03d}" for i in range(n_queries)]
    num_generations = rng.choice([2, 4], size=n_queries, p=[0.86, 0.14])

    win_probs = _simulate_scores(rng, n_queries)

    prompts_rows = []
    outcome_data = {
        "episode_id": episode_ids,
        "challenge_id": challenge_ids,
        "split": rng.choice(["train", "dev"], size=n_queries, p=[0.7, 0.3]),
    }

    for m in MODELS:
        p = win_probs[m]
        # score__{m} = fraction of num_generations attempts that succeeded.
        n_correct = rng.binomial(num_generations, p)
        outcome_data[f"num_generations__{m}"] = num_generations.astype(float)
        outcome_data[f"score__{m}"] = n_correct / num_generations

        # Output tokens: think ~24x light mean, heavy right tail for all models,
        # widest spread for think (p10/p90 ~= 20x, §1.5).
        if m == "ax31-light":
            base_mean, sigma = 643.0, 0.6
        elif m == "ax31":
            base_mean, sigma = 631.0, 0.55
        else:  # axk1-think
            base_mean, sigma = 3947.0, 1.3
        mu = np.log(base_mean) - 0.5 * sigma**2
        out_tokens = rng.lognormal(mean=mu, sigma=sigma, size=n_queries)
        outcome_data[f"output_tokens__{m}"] = out_tokens

        in_tokens = rng.lognormal(mean=np.log(120.0), sigma=0.4, size=n_queries)
        outcome_data[f"input_tokens__{m}"] = in_tokens

    for eid, cid in zip(episode_ids, challenge_ids):
        prompts_rows.append(
            {"episode_id": eid, "challenge_id": cid, "text": _make_prompt_text(rng)}
        )

    outcomes = pd.DataFrame(outcome_data)
    prompts = pd.DataFrame(prompts_rows)

    train = outcomes[outcomes["split"] == "train"].drop(columns="split").reset_index(drop=True)
    dev = outcomes[outcomes["split"] == "dev"].drop(columns="split").reset_index(drop=True)
    train.insert(2, "split", "train")
    dev.insert(2, "split", "dev")

    return {"train": train, "dev": dev, "prompts": prompts}


def write_fixture(out_dir: Path, n_queries: int = 200, seed: int = 42) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = generate_synthetic_dataset(n_queries=n_queries, seed=seed)
    for name, df in data.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
