# =============================================================================
# VESUVIO DINS Analysis — Codespace Dockerfile
# =============================================================================
#
# Strategy
# --------
# We build on top of the Microsoft Python 3.11 devcontainer base so that the
# VS Code remote-containers toolchain (non-root 'vscode' user, git, openssh,
# common utilities) is already in place.  Mantid is then installed via conda
# into an isolated 'vesuvio' environment, keeping the base Python clean.
#
# Prebuild caching
# ----------------
# Because the Mantid conda layer (~1.5 GB of C++ libraries) is baked into this
# Docker image, GitHub Codespace Prebuilds will resolve it on GitHub's infra
# and cache the result.  Researchers then get an instantly-ready environment.
#
# Official Mantid Docker references
# ----------------------------------
# https://github.com/mantidproject/dockerfiles/tree/main/Linux
# https://hub.docker.com/r/mantidproject/mantid
#
# For the full Workbench GUI image use:
#   FROM mantidproject/mantidworkbench:latest
# We omit the GUI to keep the image lean for headless Codespace use.
# =============================================================================

FROM mcr.microsoft.com/devcontainers/python:3.11

# ---------------------------------------------------------------------------
# 1. System dependencies required by Mantid's C++ extensions
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        wget \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. Install Miniforge (conda + mamba) into /opt/conda
#    Miniforge ships with conda-forge as the default channel, which provides
#    the most up-to-date scientific Python stack and is compatible with the
#    'mantid' conda channel.
# ---------------------------------------------------------------------------
ENV CONDA_DIR=/opt/conda
ENV PATH="${CONDA_DIR}/bin:${PATH}"

RUN curl -fsSL \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniforge.sh \
    && conda clean --all --yes

# ---------------------------------------------------------------------------
# 3. Create the 'vesuvio' conda environment with Mantid and the full
#    scientific Python stack used by the analysis pipeline.
#
#    Mantid channel:  https://anaconda.org/mantid/
#    Mantid package:  mantid  (framework only — no GUI, smaller footprint)
#
#    BLAS backend: OpenBLAS is installed by default via conda-forge and
#    integrates cleanly with NumPy ≥ 2.0.  MKL can be substituted for
#    better single-thread LAPACK performance on Intel hardware by replacing
#    'openblas' with 'mkl' below.
#
#    Numba: installed from conda-forge to ensure LLVM compatibility.
# ---------------------------------------------------------------------------
RUN mamba create -y -n vesuvio -c mantid -c conda-forge \
        python=3.11 \
        mantid \
        numpy">=2.0" \
        scipy \
        matplotlib \
        numba \
        scikit-learn \
        pyyaml \
        iminuit \
        jacobi \
        pydantic">=2.0" \
        openblas \
        pytest">=7.0" \
        pytest-cov \
        streamlit \
        plotly \
        dash \
        ipykernel \
        jupyterlab \
    && conda clean --all --yes

# Make the vesuvio environment the default for all subsequent RUN / CMD
SHELL ["conda", "run", "-n", "vesuvio", "/bin/bash", "-c"]

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
#    Activating this flag in the container environment ensures that every
#    script execution inside the Codespace caps noOfMSIterations to 0 and
#    limits nSamples ≤ 3, enabling high-frequency code validation without
#    waiting for a full production run.  Set to "0" or unset for production.
# ---------------------------------------------------------------------------
ENV VESUVIO_RUNNING_TEST=1

# ---------------------------------------------------------------------------
# 7. Ensure the conda 'vesuvio' environment is activated in interactive shells
# ---------------------------------------------------------------------------
RUN echo "conda activate vesuvio" >> /etc/bash.bashrc \
    && echo "conda activate vesuvio" >> /home/vscode/.bashrc 2>/dev/null || true

# Update PATH so the devcontainer picks up the environment's Python
ENV PATH="${CONDA_DIR}/envs/vesuvio/bin:${PATH}"

WORKDIR /workspaces/scatt_scripts
