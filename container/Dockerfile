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
# router/config.py's USE_EMBEDDING_BRANCH=False: the §10.1 ARM latency/memory
# gate FAILED. An ONNX Runtime CPU embedding branch (intfloat/multilingual-
# e5-small) was built and tuned (fp32 export, thread caps, length-sorted
# batching, a memory-pattern-cache leak fix) but still needed ~150s for real
# Train+Dev on real arm64 hardware (Apple Silicon) -- over the
# 90-seconds-per-tier budget with no further safe lever found in the time
# available. The shipped pipeline is TF-IDF+SVD+handcrafted features only
# (router/features/lexical.py, handcrafted.py); no embedding model, no
# onnxruntime/transformers/torch in this image at all. See
# router/features/embeddings.py's module docstring for the full history if
# the embedding branch is revisited post-deadline.
#
# Multi-stage: the builder stage has pip; the final stage copies only the
# installed packages + app code, and (matching the original single-stage
# example) removes pip/setuptools/wheel from the final image since nothing
# is installed at runtime.

FROM python:3.11-slim-bookworm AS builder

WORKDIR /build
COPY container/requirements-runtime.txt .
RUN python3 -m pip install --no-cache-dir --prefix=/install -r requirements-runtime.txt

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

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/router \
    PYTHONUNBUFFERED=1 \
    TMPDIR=/tmp \
    ROUTER_PIPELINE_DIR=/opt/router/artifacts/router-model \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1

# numpy/scipy/scikit-learn (via OpenBLAS) auto-detect thread count from the
# CPU count they can see, which inside a container is not reliably clamped
# to --cpus 2 -- observed in practice trying to spawn up to 64 threads
# (matching the *host's* core count) and hitting the container's
# --pids-limit 32 (process+thread total, docs/RUNTIME.md), with every failed
# pthread_create burning time inside the 90-seconds-per-tier budget. The
# OPENBLAS/OMP/MKL/NUMEXPR *_NUM_THREADS env vars above are the fix.
# LightGBM is separately pinned via router/config.py's LGBM_COMMON_PARAMS
# (num_threads=1, load-bearing for validate.py's repeatability check).

COPY --chown=65532:65532 src container/entrypoint.py /opt/router/
COPY --chown=65532:65532 artifacts/router-model /opt/router/artifacts/router-model

WORKDIR /opt/router

USER 65532:65532

ENTRYPOINT ["python3", "/opt/router/entrypoint.py"]
