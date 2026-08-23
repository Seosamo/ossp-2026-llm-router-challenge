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
  (`router/features/embeddings.py::SentenceTransformerBackend`), §5.1 of the
  router's design doc.
- **Upstream**: <https://huggingface.co/intfloat/multilingual-e5-small>
- **Pinned revision**: `614241f622f53c4eeff9890bdc4f31cfecc418b3` (also set as
  `container/Dockerfile`'s `E5_SMALL_REVISION` build arg -- keep both in sync)
- **License**: MIT (weights are openly published, satisfying
  `docs/CHALLENGE_RULES.md`'s "가중치를 공개한 모델" requirement)
- **File SHA-256** (at the pinned revision):

  | file | SHA-256 |
  | --- | --- |
  | `model.safetensors` | `1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477`* |
  | `tokenizer.json` | `0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39`* |
  | `sentencepiece.bpe.model` | `cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865`* |
  | `config.json` | `69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959`* |
  | `sentence_bert_config.json` | `948201d8329907aae938fa62f9ceeed53f5694dacc2b87b9f3b78b37ee986529`* |
  | `special_tokens_map.json` | `d05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7`* |
  | `tokenizer_config.json` | `a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b`* |
  | `modules.json` | `c6e29747481e8b5dd2b58401966aeac910de39092f90cda9a704b1545f902b04`* |

  \* Computed locally from the huggingface_hub cache after downloading the
  pinned revision; re-verify against the upstream repo (or
  `huggingface-cli scan-cache` output) before final submission, since this
  table was produced on a development machine, not in the submission build
  itself.
- Not fine-tuned; used as published.

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
| sentence-transformers | Apache-2.0 | embedding backend wrapper |
| torch (installed separately, CPU-only from `download.pytorch.org/whl/cpu`) | BSD-3-Clause | sentence-transformers' inference backend |
| transformers (transitive) | Apache-2.0 | tokenizer/model loading for the embedding backend |
| tokenizers (transitive) | Apache-2.0 | fast tokenization |
| safetensors (transitive) | Apache-2.0 | model weight format |

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
