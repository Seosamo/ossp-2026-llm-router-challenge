<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# Model and dependency notices

`docs/SUBMISSION.md` requires recording name, purpose, upstream URL, pinned
version/revision, file SHA-256, and license basis for anything baked into the
submission image. This covers the learned router
(`ossp_router.learned_router`, backed by `router.RouterPipeline`).

## Embedding model

- **Name**: `intfloat/multilingual-e5-small`
- **Purpose**: sentence embedding branch of the feature extractor
  (`router/features/embeddings.py::OnnxEmbeddingBackend`), §5.1 of the
  router's design doc.
- **Upstream**: <https://huggingface.co/intfloat/multilingual-e5-small>
- **Pinned revision**: `614241f622f53c4eeff9890bdc4f31cfecc418b3` (matches
  `router/config.py`'s `EMBEDDING_MODEL_PRIMARY_REVISION`)
- **License**: MIT (weights are openly published, satisfying
  `docs/CHALLENGE_RULES.md`'s "가중치를 공개한 모델" requirement)
- **Not shipped as originally published** -- exported to ONNX offline by
  `router/scripts/export_embedding_onnx.py` (base `AutoModel` graph, output
  `last_hidden_state`; mean-pooling and L2-normalization are done by hand in
  `OnnxEmbeddingBackend`, replicating what `sentence-transformers` does
  internally). Verified numerically identical (cosine similarity 1.0, max
  abs diff 0.0 across sample texts) to the original model's embeddings.
  `router.RouterPipeline` was retrained/recalibrated against this exact
  exported model, not the original, so training-time and inference-time
  features match exactly.
- **fp32, not int8**: an int8-dynamically-quantized export (470MB -> 118MB)
  was tried first for extra image-size margin, but onnxruntime's CPU int8
  GEMM kernels turned out to be dramatically slower than fp32 on real arm64
  hardware -- 880 Dev rows did not finish in 5+ minutes on Apple Silicon at
  full CPU utilization, versus 33 seconds for the equivalent fp32
  `sentence-transformers` backend on a weaker x86_64 laptop -- likely
  missing/unoptimized ARM int8 dot-product code paths in this onnxruntime
  build. fp32 costs ~350MB more image size, still comfortably under the
  1 GiB compressed-image budget (`docs/RUNTIME.md`) alongside everything
  else in the image.
- **Why not shipped via `sentence-transformers`/torch directly**: plain
  `pip install torch` on linux/aarch64 resolves to a build bundling several
  GB of NVIDIA CUDA libraries (server-GPU support this CPU-only, GPU-less
  container never uses), which alone exceeded the 1 GiB compressed-image
  budget. `onnxruntime` has no such dependency.
- **Committed as split files**: `artifacts/e5-small-onnx/model.onnx.part-00`
  through `part-11` (each <42MB) instead of one 470.9MB file, because GitHub
  rejects files over 100MB without Git LFS. Git LFS was deliberately
  avoided: an evaluator that clones/exports the repo without `git lfs pull`
  would silently get tiny LFS pointer files instead of the real model.
  `container/Dockerfile` reassembles the parts
  (`cat model.onnx.part-* > model.onnx`) at build time; a plain `git clone`
  always has the real bytes, so this has no such failure mode. Reassembly
  was verified byte-identical to the original file (SHA-256
  `4dc9b3cff9b7f6720c421dc978e2ebc73eae6bb7164fe099d759c8a7b55a478e`) before
  the original was deleted from the working tree.
- Not fine-tuned; used as published (aside from the ONNX export above).

## Python runtime dependencies (`container/requirements-runtime.txt`)

Installed at image build time only (no network at container runtime,
`docs/RUNTIME.md`). All permissive licenses; none are copyleft.

| package | license | purpose |
| --- | --- | --- |
| lightgbm | MIT | win-probability classifiers, output-token quantile regressors |
| scikit-learn | BSD-3-Clause | TF-IDF, TruncatedSVD, Ridge regression, isotonic calibration |
| numpy | BSD-3-Clause | array plumbing |
| scipy | BSD-3-Clause | sparse-matrix support for scikit-learn |
| joblib | BSD-3-Clause | pipeline artifact (de)serialization |
| onnxruntime | MIT | embedding model inference (CPU) |
| transformers (tokenizer-only; torch is NOT installed) | Apache-2.0 | `AutoTokenizer` for the embedding backend |

Dev/offline-only tools used to PRODUCE `artifacts/e5-small-onnx/` (NOT part of
the runtime image): `torch`, `transformers` (with torch), `sentence-transformers`
(for the numerical cross-check against the ONNX export) -- see
`router/requirements.txt` and `router/scripts/export_embedding_onnx.py`.

Exact resolved versions depend on `container/requirements-runtime.txt`'s
ranges at actual build time; capture the real submission build's dependency
versions with:

```console
docker run --rm --entrypoint python3 <built-image> -m pip freeze
```

(pip is uninstalled from the final image per `container/Dockerfile`, so run
this against the `builder` stage instead, e.g.
`docker build --target builder ...` then run pip freeze in that stage) and
paste the output into this file before submission.
