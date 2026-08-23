"""Export config.EMBEDDING_MODEL_PRIMARY to ONNX + int8-quantize it, for the
OnnxEmbeddingBackend runtime path (features/embeddings.py, §10.1).

Why this exists: the submission container (docs/RUNTIME.md) is linux/arm64,
CPU-only, no network at runtime, with a 1 GiB compressed-image budget. Plain
`pip install torch` on linux/arm64 resolves to a build with multi-GB NVIDIA
CUDA libraries bundled (server-GPU support), even though this container never
uses a GPU -- that alone blows the image budget. onnxruntime has no such
dependency, so exporting the model once (offline, on a dev machine, network
allowed) and shipping only the resulting .onnx + tokenizer files removes
torch/transformers-with-torch/sentence-transformers from the runtime image
entirely.

Exports the plain AutoModel (not the sentence-transformers wrapper) so the
ONNX graph's output is last_hidden_state, matching what
OnnxEmbeddingBackend._mean_pool expects to pool itself. Run once, commit the
output directory (or store it wherever container/Dockerfile bakes it from).

Run from the repo root (needs torch + transformers + onnxruntime installed --
these are dev-only, NOT part of container/requirements-runtime.txt):
    python -m router.scripts.export_embedding_onnx --out-dir router/artifacts/e5-small-onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

from router.config import EMBEDDING_MAX_TOKENS, EMBEDDING_MODEL_PRIMARY, EMBEDDING_MODEL_PRIMARY_REVISION


def export(out_dir: Path) -> Path:
    import torch
    from transformers import AutoModel, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PRIMARY, revision=EMBEDDING_MODEL_PRIMARY_REVISION)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL_PRIMARY, revision=EMBEDDING_MODEL_PRIMARY_REVISION)
    model.eval()

    sample = tokenizer(
        ["query: hello world", "query: 안녕하세요"],
        padding=True,
        truncation=True,
        max_length=EMBEDDING_MAX_TOKENS,
        return_tensors="pt",
    )

    fp32_path = out_dir / "model.onnx"
    torch.onnx.export(
        model,
        (sample["input_ids"], sample["attention_mask"]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    tokenizer.save_pretrained(out_dir)
    return fp32_path


def quantize(fp32_path: Path, int8_path: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--keep-fp32",
        action="store_true",
        help="also keep the unquantized model.onnx (not needed at runtime; larger)",
    )
    args = parser.parse_args()

    fp32_path = export(args.out_dir)
    int8_path = args.out_dir / "model.int8.onnx"
    quantize(fp32_path, int8_path)

    print(f"fp32: {fp32_path} ({fp32_path.stat().st_size / 1e6:.1f} MB)")
    print(f"int8: {int8_path} ({int8_path.stat().st_size / 1e6:.1f} MB)")

    if not args.keep_fp32:
        fp32_path.unlink()
        print(f"removed {fp32_path} (config.EMBEDDING_ONNX_MODEL_FILENAME should point at the int8 file)")


if __name__ == "__main__":
    main()
