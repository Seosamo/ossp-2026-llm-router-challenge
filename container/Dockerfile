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
# The embedding branch uses ONNX Runtime (router/config.py's
# EMBEDDING_BACKEND="onnxruntime"), NOT sentence-transformers/torch: plain
# "pip install torch" resolves to a build with several GB of bundled NVIDIA
# CUDA libraries on linux/aarch64 (server-GPU support this CPU-only, GPU-less
# container never uses -- docs/RUNTIME.md), which alone blows the 1 GiB
# compressed-image budget. The exported {model.int8.onnx, tokenizer files}
# under artifacts/e5-small-onnx/ are produced offline by
# router/scripts/export_embedding_onnx.py and just copied in here -- nothing
# is downloaded at build OR run time for the model itself.
#
# Multi-stage: the builder stage has pip; the final stage copies only the
# installed packages + baked model files + app code, and (matching the
# original single-stage example) removes pip/setuptools/wheel from the final
# image since nothing is installed at runtime.

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
    ROUTER_EMBEDDING_ONNX_DIR=/opt/router/artifacts/e5-small-onnx \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
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
# OPENBLAS/OMP/MKL/NUMEXPR *_NUM_THREADS env vars above are the fix;
# LightGBM is separately pinned via router/config.py's LGBM_COMMON_PARAMS
# (num_threads=1, load-bearing for validate.py's repeatability check), and
# onnxruntime's own thread pool is capped in code
# (router/features/embeddings.py::OnnxEmbeddingBackend) since it does not
# reliably honor these environment variables.

COPY --chown=65532:65532 src container/entrypoint.py /opt/router/
COPY --chown=65532:65532 artifacts/router-model /opt/router/artifacts/router-model

# model.int8.onnx is committed to git split into <100MB chunks
# (model.int8.onnx.part-NN) because GitHub rejects files over 100MB without
# Git LFS -- and LFS was deliberately avoided here: an evaluator that clones
# or exports the repo without running `git lfs pull` would silently get a
# tiny LFS pointer file instead of the real model, which the router would
# then try to load as an ONNX graph and fail (or worse, fail in a way that's
# not obviously about LFS). Plain chunked files reassembled at build time
# have no such failure mode -- `git clone` always gets the real bytes.
COPY --chown=65532:65532 artifacts/e5-small-onnx/tokenizer.json artifacts/e5-small-onnx/tokenizer_config.json artifacts/e5-small-onnx/special_tokens_map.json /opt/router/artifacts/e5-small-onnx/
COPY artifacts/e5-small-onnx/model.int8.onnx.part-* /tmp/onnx-parts/
RUN cat /tmp/onnx-parts/model.int8.onnx.part-* > /opt/router/artifacts/e5-small-onnx/model.int8.onnx \
    && chown 65532:65532 /opt/router/artifacts/e5-small-onnx/model.int8.onnx \
    && rm -rf /tmp/onnx-parts

WORKDIR /opt/router

USER 65532:65532

ENTRYPOINT ["python3", "/opt/router/entrypoint.py"]
