# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# Base image swapped from the provided python:3.11.15-alpine3.23 example to a
# glibc (Debian) base: lightgbm/scikit-learn/scipy/numpy do not ship
# musllinux (Alpine) wheels for linux/arm64, so pip install on Alpine falls
# back to a source build (requires a Fortran/C toolchain + OpenBLAS, slow and
# fragile) or fails outright. README.md's own baseline Dockerfile note says a
# different base image is allowed as long as linux/arm64 + the router-run
# interface + no-runtime-network are preserved (docs/RUNTIME.md,
# docs/SUBMISSION.md) -- see BASE_IMAGE.md for this base's pinned digest and
# rationale.
#
# Multi-stage: the builder stage has pip/a resolvable toolchain; the final
# stage copies only the installed packages + baked model weights + app code,
# and (matching the original single-stage example) removes pip/setuptools/
# wheel from the final image since nothing is installed at runtime.

FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# PyPI's plain "torch" wheel for linux/aarch64 bundles NVIDIA CUDA libraries
# (cuBLAS/cuDNN/NCCL/...) intended for ARM server GPUs (e.g. NVIDIA Grace) --
# several GB by itself, even though this router only ever does CPU inference
# and the container has no GPU (docs/RUNTIME.md). Installing torch ALONE
# first, from PyTorch's dedicated CPU-only index, avoids that resolution
# entirely. Must be its own pip invocation with no other index: mixing in
# --extra-index-url PyPI here risks pip preferring a newer default (CUDA)
# build over the CPU one.
RUN python3 -m pip install --no-cache-dir --prefix=/install \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch>=1.11.0"

COPY container/requirements-runtime.txt .
# PYTHONPATH lets this second pip see the CPU-only torch already installed
# above (same --prefix) and skip re-resolving it from the default PyPI index
# via sentence-transformers' torch>=1.11.0 dependency.
RUN PYTHONPATH=/install/lib/python3.11/site-packages \
    python3 -m pip install --no-cache-dir --prefix=/install -r requirements-runtime.txt

# Bake the embedding model into the image at build time (network access is
# allowed during build, never at runtime -- docs/SUBMISSION.md). Pinned to an
# exact commit so the image is reproducible from source, per
# docs/CHALLENGE_RULES.md's "고정 버전 또는 리비전" requirement. This revision
# and the file-level SHA-256 manifest are recorded in
# THIRD_PARTY_NOTICES.md -- update both together if this ARG changes.
ARG E5_SMALL_REVISION=614241f622f53c4eeff9890bdc4f31cfecc418b3
ENV HF_HOME=/hf-cache
RUN PYTHONPATH=/install/lib/python3.11/site-packages \
    python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    repo_id='intfloat/multilingual-e5-small', \
    revision='${E5_SMALL_REVISION}', \
    cache_dir='/hf-cache', \
)"

FROM python:3.11-slim-bookworm

# lightgbm's Linux wheel links against libgomp (OpenMP) at runtime, which the
# slim base does not include by default.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Execution does not install packages at runtime, so build-only tooling is
# removed from the final image (mirrors the original Alpine example).
RUN python3 -m pip uninstall --yes pip setuptools wheel

ARG SOURCE_MANIFEST_SHA256=unbound

LABEL io.sktelecom.ossp.source-manifest-sha256="${SOURCE_MANIFEST_SHA256}"

COPY --from=builder /install /usr/local
COPY --from=builder /hf-cache /opt/router/hf-cache

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/router \
    PYTHONUNBUFFERED=1 \
    TMPDIR=/tmp \
    HF_HOME=/opt/router/hf-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    ROUTER_PIPELINE_DIR=/opt/router/artifacts/router-model

COPY --chown=65532:65532 src container/entrypoint.py /opt/router/
COPY --chown=65532:65532 artifacts/router-model /opt/router/artifacts/router-model

WORKDIR /opt/router

USER 65532:65532

ENTRYPOINT ["python3", "/opt/router/entrypoint.py"]
