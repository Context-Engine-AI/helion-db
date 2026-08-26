# ──────────────────────────────────────────────────────────────────────
# Helion — production container image
#
# Multi-stage build:
#   1. builder  — compiles the release binary with LTO
#   2. runtime  — minimal Debian image with just the binary
#
# Usage:
#   docker build -t helion-db .
#   docker run -p 6969:6969 -v helion-data:/data helion-db
#
# Environment variables (all optional):
#   HELIX_PORT                   Listen port                 (default: 6969)
#   HELIX_DATA_DIR               Persistent data root        (default: /data)
#   HELIX_KEEP_ALIVE_SECS        Idle connection timeout     (default: 30)
#   HELIX_CONFIG_PATH            Path to config.hx.json      (default: built-in)
#   HELIX_LOG                    Log level filter             (default: info)
#   HELIX_POOL_SIZE              Worker thread pool size      (default: 2048)
#   HELIX_MAX_OPEN_COLLECTIONS   LRU cache for open envs     (default: 256)
#   HELIX_MAX_MAP_SIZE_GB        Per-collection LMDB cap      (default: 48)
#   HELIX_MAX_DBS                Named LMDB DBI slots/env     (default: 65536)
#   HELIX_UPSERT_HEADROOM_MB     Upsert pre-grow floor        (default: 8)
#   HELIX_REQUEST_TIMEOUT_SECS   Per-request timeout          (default: 300)
#
# Scale notes (5 000+ collections / 10k users):
#   - Initial LMDB map: 64 MB/collection, grows on demand in fixed increments.
#   - 256 hot collections × 64–512 MB = 16–128 GB VA (safe on Linux).
#   - LRU evicts 25% of idle envs when cache full (~2 ms reopen cost).
#   - Set nofile ulimit ≥ 65536 (each LMDB env = 2 FDs).
# ──────────────────────────────────────────────────────────────────────

# ── Stage 1: Build ───────────────────────────────────────────────────
# Build inside the target platform container.
#
# This avoids fragile cross-linker/OpenSSL setup when using buildx from an
# arm64 Mac to produce linux/amd64 images. buildx/qemu handles the target
# architecture emulation for us.
FROM --platform=$TARGETPLATFORM rust:1.94-bookworm AS builder
ARG RUST_TARGET_CPU=generic
ARG HELIX_DEBUG_SYMBOLS=false

RUN apt-get update && apt-get install -y \
    clang \
    llvm-dev \
    protobuf-compiler \
    pkg-config \
    libssl-dev \
    make \
    autoconf \
    && rm -rf /var/lib/apt/lists/*
# `make` and `autoconf` are required so that `jemalloc-sys` (transitive dep
# of `tikv-jemallocator` in helix-container) can build jemalloc from vendored
# source during `cargo build`. Runtime jemalloc is statically linked, so no
# additional package is needed in the runtime stage.


WORKDIR /src

# Cache dependency builds: copy manifests first, build a dummy, then
# copy real source. Docker layer caching means deps only rebuild when
# Cargo.toml / Cargo.lock change.
COPY Cargo.toml Cargo.lock ./
COPY helixdb/Cargo.toml helixdb/Cargo.toml
COPY helix-container/Cargo.toml helix-container/Cargo.toml
COPY get_routes/Cargo.toml get_routes/Cargo.toml
COPY helix-cli/Cargo.toml helix-cli/Cargo.toml
COPY hbuild/Cargo.toml hbuild/Cargo.toml

# Create stub lib/main files so cargo can resolve the workspace
RUN mkdir -p helixdb/src helix-container/src get_routes/src \
             helix-cli/src hbuild/src && \
    echo "fn main(){}" > helix-container/src/main.rs && \
    echo "fn main(){}" > helix-cli/src/main.rs && \
    echo "fn main(){}" > hbuild/src/main.rs && \
    touch helixdb/src/lib.rs get_routes/src/lib.rs && \
    cargo build --release -p helix-container 2>/dev/null || true

# Now copy the real source and build for real
COPY . .
RUN touch helixdb/src/lib.rs helix-container/src/main.rs

# The repository-level Cargo config may tune arm64 builds for production
# CPUs. Docker Desktop's Linux VM can expose a more generic aarch64 CPU, so
# keep container builds portable unless a build arg opts into a specific CPU.
RUN mkdir -p .cargo && \
    printf '[target.aarch64-unknown-linux-gnu]\nrustflags = ["-C", "target-cpu=%s"]\n' "$RUST_TARGET_CPU" > .cargo/config.toml

RUN if [ "$HELIX_DEBUG_SYMBOLS" = "true" ]; then \
      CARGO_PROFILE_RELEASE_DEBUG=2 \
      CARGO_PROFILE_RELEASE_STRIP=false \
      cargo build --release -p helix-container; \
    else \
      cargo build --release -p helix-container && \
      strip target/release/helix-container; \
    fi && \
    cp target/release/helix-container /tmp/helix-container

# ── Stage 2: Runtime ─────────────────────────────────────────────────
FROM --platform=$TARGETPLATFORM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user with high FD limits for multi-tenant LMDB.
# Each collection opens 2 FDs; at 5000+ collections we need headroom.
# Docker --ulimit or k8s securityContext can override at runtime.
RUN useradd --create-home --shell /bin/false helix && \
    echo "helix soft nofile 65536" >> /etc/security/limits.conf && \
    echo "helix hard nofile 65536" >> /etc/security/limits.conf
USER helix

COPY --from=builder /tmp/helix-container /usr/local/bin/helix-container

# Persistent data volume
VOLUME /data

ENV HELIX_PORT=6969 \
    HELIX_DATA_DIR=/data \
    HELIX_KEEP_ALIVE_SECS=30 \
    HELIX_LOG=info \
    HELIX_MAX_OPEN_COLLECTIONS=256

EXPOSE 6969

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -sf http://localhost:${HELIX_PORT}/health || exit 1

ENTRYPOINT ["/usr/local/bin/helix-container"]
