# =============================================================================
# VESUVIO DINS Analysis — Codespace Dockerfile
# =============================================================================
#
# Base image
# ----------
# Official Mantid development image from the GitHub Container Registry,
# pinned to an immutable digest for reproducibility and supply-chain safety:
#   ghcr.io/mantidproject/mantid-development-alma9@sha256:60268b2fa11213a3dc6517b2927afb07afdda6926047f581f82a54155491f6a9
#
# This AlmaLinux 9 image ships with the full Mantid C++ build environment
# (compiler toolchain, system libraries, LaTeX) and is maintained by the
# Mantid team:
#   https://github.com/mantidproject/dockerfiles/tree/main/Linux/development/docker/Alma9.Dockerfile
#   https://github.com/orgs/mantidproject/packages/container/package/mantid-development-alma9
#
# Package management: pixi
# ------------------------
# We use pixi (https://pixi.sh) instead of Miniforge/mamba.
# Pixi is a single Rust binary (~10 MB) that:
#   • Natively resolves both conda channels (mantid, conda-forge) AND PyPI
#     packages in one unified lockfile — no separate pip install step needed.
#   • Is significantly faster than mamba thanks to its libsolv-based solver.
#   • Already declared as the project's package manager in pyproject.toml
#     via [tool.pixi.*] sections.
#
# The pixi binary is pinned to a specific release and its SHA-256 checksum
# is verified at build time to prevent supply-chain attacks.
#
# Prebuild caching
# ----------------
# The pixi binary is baked into this Docker image layer.  On first Codespace
# creation (or during a Prebuild run), `pixi install` resolves and downloads
# all packages into `.pixi/envs/default/` inside the workspace.  Subsequent
# Codespace creations from a Prebuild snapshot start with the full environment
# already in place — no runtime dependency resolution.
#
# User
# ----
# The Mantid alma9 image defines user 'abc' (uid 911).  remoteUser in
# devcontainer.json is set to match.
# =============================================================================

FROM ghcr.io/mantidproject/mantid-development-alma9@sha256:60268b2fa11213a3dc6517b2927afb07afdda6926047f581f82a54155491f6a9

# VS Code devcontainer overrides the ENTRYPOINT/CMD with its own keep-alive
# process, so the Mantid image's custom entrypoint.sh is not invoked.
ENTRYPOINT []
CMD ["/bin/bash"]

# ---------------------------------------------------------------------------
# 1. Install pixi — single binary, replaces Miniforge + mamba entirely.
#    Pinned to v0.39.0 and verified against its published SHA-256 checksum
#    to prevent supply-chain attacks from a compromised install script.
#    PIXI_HOME=/usr/local places the binary at /usr/local/bin/pixi so it is
#    on PATH for all users (including 'abc').
# ---------------------------------------------------------------------------
ARG PIXI_VERSION=0.39.0
# SHA-256 of pixi-x86_64-unknown-linux-musl.tar.gz for v0.39.0
ARG PIXI_SHA256=68a87f0a48e24a43c9c5e4b9db6ee57428effd5b753bc91b43e52a58b4d8f636
RUN set -eux; \
    curl -fsSL \
      "https://github.com/prefix-dev/pixi/releases/download/v${PIXI_VERSION}/pixi-x86_64-unknown-linux-musl.tar.gz" \
      -o /tmp/pixi.tar.gz; \
    echo "${PIXI_SHA256}  /tmp/pixi.tar.gz" | sha256sum -c -; \
    tar -xzf /tmp/pixi.tar.gz -C /usr/local/bin pixi; \
    chmod a+x /usr/local/bin/pixi; \
    rm -f /tmp/pixi.tar.gz

# ---------------------------------------------------------------------------
# 2. Matplotlib headless backend — set system-wide so every script inherits it
#    without needing  matplotlib.use('Agg')  at the top of each file.
# ---------------------------------------------------------------------------
ENV MPLBACKEND=Agg

# ---------------------------------------------------------------------------
# 3. Numba and BLAS thread configuration
#    Defaults tuned for the 4-core standard Codespace machine.
#    Override at runtime for 16-core machines:
#      NUMBA_NUM_THREADS=16 OMP_NUM_THREADS=16 python BaH2_500C.py
# ---------------------------------------------------------------------------
ENV NUMBA_NUM_THREADS=4
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4

# ---------------------------------------------------------------------------
# 4. VESUVIO_RUNNING_TEST fast-track flag
#    Default to "0" in the image so production-like runs in the same container
#    are unaffected.  The Codespace sets it to "1" via devcontainer.json
#    containerEnv, keeping fast-track mode scoped to the Codespace only.
# ---------------------------------------------------------------------------
ENV VESUVIO_RUNNING_TEST=0

WORKDIR /workspaces/scatt_scripts

