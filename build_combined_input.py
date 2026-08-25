# Temporary helper: reproduces check_runtime.py's combined Train+Dev input
# so it can be run manually against the container WITHOUT the 90s harness
# timeout, to measure the actual uncapped completion time.
# Run from the repo root: PYTHONPATH=src python3 build_combined_input.py
from pathlib import Path

from ossp_router.protocol import load_input, write_json
from ossp_router.public_runtime import combine_public_inputs
from ossp_router.protocol import submission_to_dict  # noqa: F401 (unused, keep import simple)

train = load_input(Path("data/materialized/train/inputs.json"))
dev = load_input(Path("data/materialized/dev/inputs.json"))
combined = combine_public_inputs(train, dev)

out = {
    "schema_version": combined.schema_version,
    "challenge_id": combined.challenge_id,
    "split": combined.split,
    "episodes": [
        ({"episode_id": e.episode_id, "prompt": e.prompt} if e.prompt is not None
         else {"episode_id": e.episode_id, "messages": [{"role": m.role, "content": m.content} for m in e.messages]})
        for e in combined.episodes
    ],
}
write_json(Path("build/combined-train-dev-inputs.json"), out)
print(f"wrote {len(combined.episodes)} episodes to build/combined-train-dev-inputs.json")
