<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# Model and dependency notices

`docs/SUBMISSION.md` requires recording name, purpose, upstream URL, pinned
version/revision, file SHA-256, and license basis for anything baked into the
submission image. This covers the learned router
(`ossp_router.learned_router`, backed by `router.RouterPipeline`).

## AI model: not used in the submission image

The shipped image contains **no AI model** (no embedding model, no neural
network of any kind). The router is `router.RouterPipeline`: TF-IDF word/char
n-gram features + `TruncatedSVD` + handcrafted lexical features
(`router/features/lexical.py`, `router/features/handcrafted.py`), scored by
LightGBM classifiers/regressors trained on the public Train split. Per
`docs/SUBMISSION.md`, a router with no AI model in the execution image states
this explicitly in the result report's AI model field ("해당 없음 — 실행
이미지에 AI 모델을 탑재하지 않음").

### Embedding model considered and abandoned (not shipped)

An optional sentence-embedding branch (`intfloat/multilingual-e5-small`,
MIT-licensed, weights openly published, exported to ONNX and run via
`onnxruntime` -- `router/features/embeddings.py::OnnxEmbeddingBackend`) was
built and tuned during development as a §10.1 ARM-latency contingency path.
It is **disabled and not part of the submission image**:
`router/config.py`'s `USE_EMBEDDING_BRANCH = False`, and
`container/requirements-runtime.txt` installs no `onnxruntime`,
`transformers`, or `torch`. Even the tuned fp32 ONNX export still needed
~150s for real Train+Dev on real arm64 hardware (Apple Silicon), over the
90-seconds-per-tier budget (`docs/RUNTIME.md`) with no further safe
optimization found before the deadline.

No exported model weights for this abandoned branch are present in the
submitted repository or image; only the (now-dead-code) loading path in
`router/features/embeddings.py` and the offline export tooling
(`router/scripts/export_embedding_onnx.py`, dev-only) remain, documenting the
attempt for transparency. See `container/Dockerfile`'s header comment and
`router/features/embeddings.py`'s module docstring for the full history if
this branch is revisited post-deadline.

## Python runtime dependencies (`container/requirements-runtime.txt`)

Installed at image build time only (no network at container runtime,
`docs/RUNTIME.md`). All permissive licenses; none are copyleft. Exact
resolved versions captured from the actual submission build
(`docker build --target builder ...` then `pip freeze --path
<install-prefix>/lib/python3.11/site-packages` inside that stage):

| package | resolved version | license | purpose |
| --- | --- | --- | --- |
| lightgbm | 4.5.0 | MIT | win-probability classifiers, output-token quantile regressors |
| scikit-learn | 1.5.2 | BSD-3-Clause | TF-IDF, TruncatedSVD, Ridge regression, isotonic calibration |
| numpy | 1.26.4 | BSD-3-Clause | array plumbing |
| scipy | 1.17.1 | BSD-3-Clause | sparse-matrix support for scikit-learn |
| joblib | 1.5.3 | BSD-3-Clause | pipeline artifact (de)serialization |
| threadpoolctl | 3.6.0 | BSD-3-Clause | transitive dependency of scikit-learn (thread-pool introspection) |

Dev/offline-only tools used to explore the abandoned embedding branch above
(NOT part of the runtime image): `torch`, `transformers`, `onnxruntime`,
`sentence-transformers` -- see `router/requirements.txt` and
`router/scripts/export_embedding_onnx.py`.
