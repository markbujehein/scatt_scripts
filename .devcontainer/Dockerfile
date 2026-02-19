# =============================================================================
# VESUVIO DINS Analysis — Codespace Dockerfile
# =============================================================================
#
# Base image
# ----------
# Official Mantid development image from the GitHub Container Registry:
#   ghcr.io/mantidproject/mantid-development-alma9:latest
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

FROM ghcr.io/mantidproject/mantid-development-alma9:latest

# VS Code devcontainer overrides the ENTRYPOINT/CMD with its own keep-alive
# process, so the Mantid image's custom entrypoint.sh is not invoked.
ENTRYPOINT []
CMD ["/bin/bash"]

# ---------------------------------------------------------------------------
# 1. Install pixi — single binary, replaces Miniforge + mamba entirely.
#    PIXI_HOME=/usr/local places the binary at /usr/local/bin/pixi so it is
#    on PATH for all users (including 'abc').
# ---------------------------------------------------------------------------
RUN curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=/usr/local bash \
    && chmod a+x /usr/local/bin/pixi

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
#    Caps noOfMSIterations to 0 and limits nSamples ≤ 3 for high-frequency
#    code validation.  Set to "0" or unset for production.
# ---------------------------------------------------------------------------
ENV VESUVIO_RUNNING_TEST=1

WORKDIR /workspaces/scatt_scripts

