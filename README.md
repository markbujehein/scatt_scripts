# scatt_scripts

## High-Performance DINS Analysis Pipeline for the VESUVIO Spectrometer

`vesuvio_analysis` is a Python package for Deep Inelastic Neutron Scattering (DINS)
data reduction and analysis at the [ISIS Neutron and Muon Source](https://www.isis.stfc.ac.uk/).
It extends the [Mantid Project](https://www.mantidproject.org/) framework with a
fully validated, high-performance fitting pipeline suitable for publication-quality research.

### Core Capabilities

- **Numba-Accelerated NCP Models** — Resolution kernels (`calcGaussianResolution`,
  `calcLorentzianResolution`, `pseudoVoigt`, `numericalThirdDerivative`) are
  implemented as `@njit(cache=True)` functions in `numba_routines.py`, delivering
  7×–10× speedups over the baseline NumPy path on the inner optimisation loop.

- **Dual-Optimizer NCP Fitting** — Each spectrum is fitted independently using
  `scipy.optimize.minimize` (SLSQP, primary) alongside `iminuit.Minuit` (MIGRAD,
  cross-validation).  After every fit the **iMinuit–Scipy Numerical Agreement Check**
  compares χ² values and parameter vectors at a 1 % tolerance threshold
  (`_AGREEMENT_THRESHOLD = 0.01`), issuing diagnostic warnings for any spectrum
  where the two optimisers disagree beyond that bound.

- **y-Space Momentum Distribution** — Following the DINS impulse approximation, raw
  time-of-flight spectra are converted to the West scaling variable

$$y = \frac{M}{\hbar q}\left(E_0 - E_1 - E_\mathrm{recoil}\right)$$

  where $M$ is the nuclear mass, $\hbar q$ is the transferred momentum, and
  $E_\mathrm{recoil} = \hbar^2 q^2 / 2M$ is the recoil energy.  The resulting
  single-particle momentum distribution $J(y)$ is fitted via iMinuit using
  model backends including `SINGLE_GAUSSIAN`, `GC_C4`, `GC_C6`, `GC_C4_C6`,
  `DOUBLE_WELL`, and `ANSIO_GAUSSIAN`.

- **Full-Stack Statistical Workflow (Phase 6)** — A post-fit statistical analysis
  pipeline provides:
  - Hardware outlier detection via PCA + `EllipticEnvelope` (`HardwareOutlierDetector`)
  - Physics-trend clustering on instrument geometry features $(L_1, \theta)$ via DBSCAN
    (`PhysicsTrendClusterer`)
  - Bayesian uncertainty quantification via Rubin-style Dirichlet-weighted bootstrap
    on NCP residuals (`BayesianBootstrap`)

- **Pydantic v2 Validation** — All `InitialConditions` parameter classes are validated
  at runtime via `ic_validation.py`, which flags physical inconsistencies (e.g. mass–ratio
  mismatches, out-of-range spectra) via `RuntimeWarning` diagnostics during data reduction.

---

## Installation

### Option 1 — Pixi (recommended)

[Pixi](https://prefix.dev/) manages the Conda environment and editable Python install
in a single step:

```bash
# Clone the repository
git clone https://github.com/markbujehein/scatt_scripts.git
cd scatt_scripts

# Install all dependencies (reads pyproject.toml + pixi.toml)
pixi install

# Run the test suite (no Mantid required)
pixi run pytest
```

### Option 2 — Conda + pip

```bash
conda create -n vesuvio python=3.11
conda activate vesuvio

# Install Mantid (required for full data reduction; skip for test-only installs)
conda install -c mantid mantid

# Install the package and its dependencies in editable mode
pip install -e ".[dev]"

# Run the test suite
python -m pytest tests/ -v
```

### Option 3 — pip only (CI / test-only)

```bash
pip install -e ".[test]"
python -m pytest tests/ -v
```

Core runtime dependencies (`numpy`, `scipy`, `iminuit`, `numba`, `scikit-learn`,
`pydantic>=2.0`, `jacobi`, `matplotlib`, `pyyaml`) are declared in `pyproject.toml`
and installed automatically by any of the above methods.

---

## Quick Start

Four fully annotated example scripts are provided:

| Script | Sample | Notes |
|---|---|---|
| `starch_80_RD.py` | Starch 80% RD | Most complete inline comments — **start here** |
| `BaH2_500C.py` | Barium hydride at 500 °C | JOINT procedure with H-ratio constraint |
| `D_HMT.py` | Deuterated HMT | Forward-only procedure |
| `thymol_10K_Gauss1D.py` | Thymol at 10 K | Full Phase 6 flags enabled |

### Running a new sample

1. Copy an existing script (e.g. `starch_80_RD.py`) and rename it for the new sample.

2. Edit the nine parameter classes (`LoadVesuvioBackParameters`,
   `LoadVesuvioFrontParameters`, `GeneralInitialConditions`,
   `BackwardInitialConditions`, `ForwardInitialConditions`,
   `YSpaceFitInitialConditions`, `UserScriptControls`,
   `BootstrapInitialConditions`, `BootstrapAnalysis`) to reflect the
   sample's run numbers, masses, and fit bounds.

3. Execute the script.  On first run, `LoadVesuvio` (Mantid) fetches the raw
   data from the ISIS archive and caches it as Nexus files under a
   **versioned subdirectory** of `experiments/<sample>/input_ws/`, for example:

       experiments/<sample>/input_ws/backward_1.0/
       experiments/<sample>/input_ws/forward_1.0/

   Alongside the `.nxs` files, a matching JSON parameter log is written
   (e.g. `backward_1.0.json`) and is used by `ICHelpers.inputDirsForSample()`
   to re-use an existing cache.  Subsequent runs load from this versioned
   cache instead of calling `LoadVesuvio` again.

   If `LoadVesuvio` is unavailable on your system, **do not** copy files only
   into `experiments/<sample>/input_ws/`.  Instead, create the appropriate
   versioned directory (e.g. `experiments/<sample>/input_ws/backward_1.0/`)
   and place the `.nxs` files there, following the naming convention of the
   example samples, and ensure that a corresponding JSON file
   (e.g. `backward_1.0.json`) exists in `experiments/<sample>/input_ws/`
   describing the same parameters.  This layout matches what the helper
   functions expect and prevents unnecessary attempts to call `LoadVesuvio`.

4. All subsequent data reduction reads from the local Nexus cache in
   `experiments/<sample>/input_ws/<version_tag>/`.
5. Bootstrap resampling results are stored under
   `experiments/<sample>/bootstrap_data/` (residual / Gaussian-error
   resampling) or `experiments/<sample>/jackknife_data/` (jackknife).

6. Post-hoc analysis of stored bootstrap data is performed by calling
   `runAnalysisOfStoredBootstrap()` — this reads from the directories in
   step 5 and does not re-run the resampling loop.

---

## Repository Structure

```
vesuvio_analysis/
├── core_functions/
│   ├── analysis_functions.py   # NCP fitting: scipy SLSQP + iMinuit MIGRAD
│   ├── numba_routines.py       # @njit-accelerated resolution kernels
│   ├── iminuit_costs.py        # NCPCostFunction (_parameters dict interface)
│   ├── fit_in_yspace.py        # J(y)-space fitting (multiple model backends)
│   ├── statistical_plugins.py  # Phase 6: outlier detection, clustering, bootstrap
│   ├── run_script.py           # Top-level pipeline entry (runScript) used by submission scripts
│   ├── procedures.py           # Shared orchestration utilities used by run_script.py (requires Mantid)
│   ├── log_manager.py          # Structured YAML run logging
│   └── ic_validation.py        # Pydantic v2 InitialConditions validation
├── ip_files/                   # Instrument parameter files for LoadVesuvio
└── mcp_server/                 # MCP servers for log inspection and ADS state
tests/                          # unittest-based test suite (zero Mantid dependency)
experiments/                    # Per-sample data and results (git-ignored)
```

---

## Development Roadmap

All phases target the `dev` branch.  The full architectural rationale and
data-flow maps are documented in [`ARCHITECTURE_AUDIT.md`](ARCHITECTURE_AUDIT.md).

| Phase | Goal | Status | Test Coverage |
|---|---|---|---|
| 1 — Numba Acceleration | `@njit` resolution functions in `numba_routines.py`; 7×–10× inner-loop speedup | ✅ | `test_numba_regression.py` (17 tests) |
| 2 — iMinuit NCP Cost | `NCPCostFunction` class for TOF-domain fitting with `_parameters` dict interface | ✅ | `test_iminuit_cross_check.py` (15 tests) |
| 3 — Unified Interface | `_parameters`, `ndata`, `errordef` on all cost classes; `GlobalNCPCostFunction` | ✅ | `test_interface_unification.py` (23 tests) |
| 4 — Workspace Lifecycle | Entry/exit guards, boundary enforcement, Mantid naming conventions | ✅ | `test_workspace_safety.py` (23 tests) |
| 5 — Numerical Validation | iMinuit–Scipy Numerical Agreement Check (1 % tolerance); NumPy 2.x compatibility | ✅ | `test_iminuit_cross_check.py` (+ 4 agreement tests) |
| 6 — Statistical Workflow | `HardwareOutlierDetector`, `PhysicsTrendClusterer`, `BayesianBootstrap`; post-fit pipeline in `run_script.py` | ✅ | `test_statistical_workflow.py` (12 tests) |

### Phase 7 — Planned

The following capabilities are deferred to future phases:

- Profile-likelihood scans for asymmetric confidence intervals on NCP parameters
- Goodness-of-fit dashboard (reduced χ², residual maps, autocorrelation diagnostics)
- Systematic error budget (instrument geometry uncertainties, background model sensitivity)
- Publication export pipeline (HDF5 archival, LaTeX table generation)

---

## Physics Background

### Neutron Compton Profile

Under the impulse approximation, the double-differential neutron scattering
cross-section from a nucleus of mass $M$ is proportional to the single-particle
momentum distribution $J(y)$:

$$\frac{\mathrm{d}^2\sigma}{\mathrm{d}\Omega\,\mathrm{d}E_1} \propto
  \frac{\sigma_\mathrm{tot}}{4\pi}\,\frac{q}{k_1}\,J(y)$$

where the West scaling variable is

$$y = \frac{M}{\hbar q}\left(E_0 - E_1 - \frac{\hbar^2 q^2}{2M}\right)$$

with $E_0$ ($E_1$) the incident (final) neutron energy and $\hbar q$ the
transferred momentum.  At the VESUVIO spectrometer the final energy is fixed
at $E_f = 4906\,\text{meV}$ by a gold-foil analyser in back-scattering geometry.

### NCP Line Shape

Each nucleus contributes a neutron Compton profile (NCP) $C(t)$ in
time-of-flight space.  The NCP is modelled as a pseudo-Voigt convolution of
the Gaussian momentum distribution $J(y)$ with the instrumental resolution
function $R(y)$, plus a final-state-effect (FSE) correction:

$$C(t) = \int \left[J(y') + J_\mathrm{FSE}(y')\right] R(y - y')\,\mathrm{d}y'$$

where $J_\mathrm{FSE}$ is proportional to the numerical third derivative of $J(y)$
(Sears correction).  The resolution function $R$ contains both Gaussian and
Lorentzian components; the corresponding kernels are evaluated by Numba-compiled
`@njit` functions for maximum throughput.

---

## Citation

If you use this software in published research, please cite the original VESUVIO
analysis codebase:

> G. Maciel Pereira, *scatt_scripts — Python analysis scripts for the VESUVIO
> spectrometer*, ISIS Neutron and Muon Source, Rutherford Appleton Laboratory.
> https://github.com/GuiMacielPereira/scatt_scripts/

---

## Licence

As of this writing, neither this repository nor the upstream `scatt_scripts`
project provides an explicit license file. No licence is therefore granted for
use, copying, modification, or distribution beyond what is permitted by
applicable law. If you wish to use this code in a way that requires a licence,
please contact the maintainers to agree appropriate terms.
