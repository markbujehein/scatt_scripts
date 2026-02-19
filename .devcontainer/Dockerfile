# =============================================================================
# VESUVIO DINS Analysis — Codespace Dockerfile
# =============================================================================
#
# Base image
# ----------
# We build on top of the official Mantid development image published to the
# GitHub Container Registry by the Mantid project:
#   ghcr.io/mantidproject/mantid-development-alma9:latest
#
# This AlmaLinux 9 image ships with the full Mantid C++ build environment
# (compiler toolchain, system libraries, LaTeX) and is maintained by the
# Mantid team.  Using it here means:
#   • No custom Mantid C++ build step — all system deps are pre-baked.
#   • The image is served directly from GHCR, so GitHub Codespace Prebuilds
#     pull it without Docker Hub rate limits.
#   • Any future Mantid build-system changes are automatically picked up
#     by pulling a new digest of :latest.
#
# Image source & Dockerfile:
#   https://github.com/mantidproject/dockerfiles/tree/main/Linux/development/docker/Alma9.Dockerfile
# GHCR package:
#   https://github.com/orgs/mantidproject/packages/container/package/mantid-development-alma9
#
# On top of this base we install Miniforge (conda + mamba) and create an
# isolated 'vesuvio' conda environment with the Mantid Python package and
# the full VESUVIO scientific stack.
#
# Prebuild caching
# ----------------
# Because both the Mantid system layer (in the base image) and the conda
# environment (baked into this layer) are resolved at image-build time,
# GitHub Codespace Prebuilds will cache this image.  Researchers then get an
# instantly-ready environment with no runtime dependency resolution.
#
# User
# ----
# The Mantid alma9 image defines user 'abc' (uid 911) as the non-root user.
# All VS Code devcontainer tooling runs as 'abc'; remoteUser in
# devcontainer.json is set to match.
# =============================================================================

FROM ghcr.io/mantidproject/mantid-development-alma9:latest

# VS Code devcontainer overrides the ENTRYPOINT/CMD with its own keep-alive
# process, so the Mantid image's custom entrypoint.sh is not invoked.
# We reset it here to make the intent explicit.
ENTRYPOINT []
CMD ["/bin/bash"]

# ---------------------------------------------------------------------------
# 1. Install Miniforge (conda + mamba) into /opt/conda (owned by root,
#    readable by all users including 'abc').
# ---------------------------------------------------------------------------
ENV CONDA_DIR=/opt/conda
ENV PATH="${CONDA_DIR}/bin:${PATH}"

RUN curl -fsSL \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniforge.sh \
    && conda clean --all --yes \
    && chmod -R a+rX "${CONDA_DIR}"

# ---------------------------------------------------------------------------
# 2. Create the 'vesuvio' conda environment.
#
#    Channels (priority order):
#      mantid      — official Mantid Python package (framework + Python bindings)
#      conda-forge — everything else (numpy ≥ 2.0, numba, iminuit, pydantic…)
#
#    BLAS backend: OpenBLAS (conda-forge default) integrates cleanly with
#    NumPy ≥ 2.0.  Swap 'openblas' → 'mkl' on Intel hardware for single-
#    thread LAPACK speed-up during the Phase 6 bootstrap resampling loops.
# ---------------------------------------------------------------------------
RUN mamba create -y -n vesuvio -c mantid -c conda-forge \
        python=3.11 \
        mantid \
        "numpy>=2.0" \
        scipy \
        matplotlib \
        numba \
        scikit-learn \
        pyyaml \
        iminuit \
        jacobi \
        "pydantic>=2.0" \
        openblas \
        "pytest>=7.0" \
        pytest-cov \
        streamlit \
        plotly \
        dash \
        ipykernel \
        jupyterlab \
    && conda clean --all --yes \
    && chmod -R a+rX "${CONDA_DIR}/envs/vesuvio"

# ---------------------------------------------------------------------------
# 3. Activate the 'vesuvio' environment for the 'abc' user's interactive
#    shells (bash / zsh).
# ---------------------------------------------------------------------------
RUN echo "conda activate vesuvio" >> /home/abc/.bashrc \
    && echo "conda activate vesuvio" >> /etc/bash.bashrc

# ---------------------------------------------------------------------------
# 4. Matplotlib headless backend — set system-wide so every script inherits it
#    without needing  matplotlib.use('Agg')  at the top of each file.
# ---------------------------------------------------------------------------
ENV MPLBACKEND=Agg

# ---------------------------------------------------------------------------
# 5. Numba and BLAS thread configuration
#    Defaults tuned for the 4-core standard Codespace machine.
#    Override at runtime for 16-core machines:
#      NUMBA_NUM_THREADS=16 OMP_NUM_THREADS=16 python BaH2_500C.py
# ---------------------------------------------------------------------------
ENV NUMBA_NUM_THREADS=4
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4

# ---------------------------------------------------------------------------
# 6. VESUVIO_RUNNING_TEST fast-track flag
#    Activating this flag caps noOfMSIterations to 0 and limits nSamples ≤ 3,
#    enabling high-frequency code validation without a full production run.
#    Set to "0" or unset for production.
# ---------------------------------------------------------------------------
ENV VESUVIO_RUNNING_TEST=1

# Update PATH so the devcontainer picks up the environment's Python
ENV PATH="${CONDA_DIR}/envs/vesuvio/bin:${PATH}"

WORKDIR /workspaces/scatt_scripts

