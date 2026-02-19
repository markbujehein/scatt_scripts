"""
Benchmarking suite for VESUVIO numerical-core functions.

Exercises the pure-numpy computation kernels that are candidates for Numba
acceleration.  The replicas below closely mirror the production code in
``vesuvio_analysis/core_functions/analysis_functions.py`` and
``fit_in_yspace.py``, including the same custom VESUVIO constants.

Run with pytest-benchmark so that results can be compared against a stored
baseline (main-branch JSON):

    pip install pytest pytest-benchmark
    # Save a baseline on the main branch:
    pytest scripts/run_benchmarks.py --benchmark-save=baseline

    # On the PR branch, compare against the baseline:
    pytest scripts/run_benchmarks.py \\
        --benchmark-compare=baseline \\
        --benchmark-compare-fail=mean:20%

The suite intentionally does NOT import Mantid, so it can run in any
standard Python environment (NumPy only).
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# VESUVIO custom constants (mirrors loadConstants() in analysis_functions.py)
# ---------------------------------------------------------------------------

_MN = 1.008          # a.m.u.
_EF = 4906.0         # meV – final energy selected by gold foil
_EN_TO_VEL = 4.3737e-4   # converts sqrt(meV) → m/μs
_VF = np.sqrt(_EF) * _EN_TO_VEL  # final neutron velocity [m/μs]
_HBAR = 2.0445       # custom reduced Planck constant in VESUVIO units


# ---------------------------------------------------------------------------
# Inline replicas of the benchmarkable kernels
# ---------------------------------------------------------------------------

def _calculate_kinematics(dataX: np.ndarray, instr_pars: np.ndarray) -> tuple:
    """
    Faithful replica of calculateKinematicsArrays from analysis_functions.py.

    dataX      : (n_spec, n_tof)  ToF array [μs]
    instr_pars : (n_spec, 6)      [det, plick, angle_deg, T0_us, L0_m, L1_m]
    Returns (v0, E0, delta_E, delta_Q), each shape (n_spec, n_tof).
    """
    # Unpack per-spectrum instrument parameters (broadcast over ToF axis)
    angle = instr_pars[:, 2, np.newaxis]  # degrees
    T0 = instr_pars[:, 3, np.newaxis]    # μs (electronic delay)
    L0 = instr_pars[:, 4, np.newaxis]    # m  (primary flight path)
    L1 = instr_pars[:, 5, np.newaxis]    # m  (secondary flight path)

    # Effective ToF after electronic delay
    t_us = dataX - T0

    # Incident velocity [m/μs] – production formula
    with np.errstate(divide="ignore", invalid="ignore"):
        v0 = np.where(t_us > 0, _VF * L0 / (_VF * t_us - L1), 0.0)

    # Incident energy [meV]
    E0 = np.square(v0 / _EN_TO_VEL)

    delta_E = E0 - _EF

    # |Q| via cosine rule [Å⁻¹]
    cos_angle = np.cos(angle * np.pi / 180.0)
    with np.errstate(invalid="ignore"):
        delta_Q2 = (2.0 * _MN / _HBAR ** 2) * (
            E0 + _EF - 2.0 * np.sqrt(E0 * _EF) * cos_angle
        )
    delta_Q = np.sqrt(np.maximum(delta_Q2, 0.0))

    return v0, E0, delta_E, delta_Q


def _calc_gaussian_resolution(
    masses: np.ndarray,
    v0: np.ndarray,
    E0: np.ndarray,
    delta_E: np.ndarray,
    delta_Q: np.ndarray,
    res_pars: np.ndarray,
    instr_pars: np.ndarray,
) -> np.ndarray:
    """
    Faithful vectorised replica of calcGaussianResolution from
    analysis_functions.py (the function identified as the fitting bottleneck).

    masses     : (n_mass,)   – neutron masses [a.m.u.]
    v0, E0, delta_E, delta_Q : (n_spec,) arrays at y-space centres
    res_pars   : (n_spec, 6) – [dE1, dTOF, dTheta, dL0, dL1, dE1_lorz]
    instr_pars : (n_spec, 6) – [det, plick, angle_deg, T0, L0, L1]
    Returns gaussianResWidth shape (n_mass, n_spec).
    """
    n_mass = len(masses)
    n_spec = v0.shape[0]

    # Unpack resolution parameters (broadcast over mass axis later)
    dE1 = res_pars[:, 0]
    dTOF = res_pars[:, 1]
    dTheta = res_pars[:, 2]
    dL0 = res_pars[:, 3]
    dL1 = res_pars[:, 4]

    angle = instr_pars[:, 2] * np.pi / 180.0  # convert deg → rad
    L0 = instr_pars[:, 4]
    L1 = instr_pars[:, 5]

    # Broadcast (n_mass, n_spec)
    masses_col = masses[:, np.newaxis]  # (n_mass, 1)
    v0 = v0[np.newaxis, :]              # (1, n_spec)
    E0 = E0[np.newaxis, :]
    delta_Q = delta_Q[np.newaxis, :]
    dE1 = dE1[np.newaxis, :]
    dTOF = dTOF[np.newaxis, :]
    dTheta = dTheta[np.newaxis, :]
    dL0 = dL0[np.newaxis, :]
    dL1 = dL1[np.newaxis, :]
    angle = angle[np.newaxis, :]
    L0 = L0[np.newaxis, :]
    L1 = L1[np.newaxis, :]

    # Production partial derivatives for W (energy transfer)
    dWdE1 = 1.0 + (E0 / _EF) ** 1.5 * (L1 / L0)
    dWdTOF = 2.0 * E0 * v0 / L0
    dWdL1 = 2.0 * E0 ** 1.5 / _EF ** 0.5 / L0
    dWdL0 = 2.0 * E0 / L0

    dW2 = (dWdE1 ** 2 * dE1 ** 2 + dWdTOF ** 2 * dTOF ** 2
           + dWdL1 ** 2 * dL1 ** 2 + dWdL0 ** 2 * dL0 ** 2) * (masses_col / _HBAR ** 2 / (delta_Q + 1e-30)) ** 2

    # Production partial derivatives for Q (momentum transfer)
    dQdE1 = (1.0 - (E0 / _EF) ** 1.5 * L1 / L0
             - np.cos(angle) * ((E0 / _EF) ** 0.5 - L1 / L0 * E0 / _EF))
    dQdTOF = 2.0 * E0 * v0 / L0
    dQdL1 = 2.0 * E0 ** 1.5 / L0 / _EF ** 0.5
    dQdL0 = 2.0 * E0 / L0
    dQdTheta = 2.0 * np.sqrt(E0 * _EF) * np.sin(angle)

    dQ2 = (dQdE1 ** 2 * dE1 ** 2
           + (dQdTOF ** 2 * dTOF ** 2 + dQdL1 ** 2 * dL1 ** 2 + dQdL0 ** 2 * dL0 ** 2)
           * np.abs(_EF / (E0 + 1e-30) * np.cos(angle) - 1.0)
           + dQdTheta ** 2 * dTheta ** 2) * (_MN / _HBAR ** 2 / (delta_Q + 1e-30)) ** 2

    return np.sqrt(dW2 + dQ2)  # shape (n_mass, n_spec)


def _pseudo_voigt(x: np.ndarray, sigma: float, gamma: float) -> np.ndarray:
    """
    Faithful replica of pseudoVoigt from analysis_functions.py.

    Uses the same Thompson–Cox–Hastings approximation coefficients and the
    same component formulas as the production code.  normVoigt is disabled
    (norm=1) to keep the benchmark self-contained.
    """
    fg = 2.0 * sigma * np.sqrt(2.0 * np.log(2.0))
    fl = 2.0 * gamma
    f = 0.5346 * fl + np.sqrt(0.2166 * fl ** 2 + fg ** 2)
    ratio = fl / (f + 1e-30)
    eta = 1.36603 * ratio - 0.47719 * ratio ** 2 + 0.11116 * ratio ** 3

    sigma_v = f / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma_v = f / 2.0

    gauss = np.exp(-x ** 2 / (2.0 * sigma_v ** 2)) / (sigma_v * np.sqrt(2.0 * np.pi))
    lorentz = (gamma_v / np.pi) / (x ** 2 + gamma_v ** 2)

    return eta * lorentz + (1.0 - eta) * gauss


def _weighted_sym_arr(
    dataY: np.ndarray, dataE: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Faithful replica of weightedSymArr from fit_in_yspace.py.

    Symmetrises a 2-D array of spectra (n_spec, n_bins) using inverse-variance
    weighting and np.flip along axis=1, exactly as in production.
    """
    if not (dataY.ndim == 2 and dataE.ndim == 2):
        raise ValueError("Arrays must be 2-D (n_spec, n_bins)")

    cutoff_mask = dataY == 0
    dataE = dataE.copy()
    dataE[cutoff_mask] = np.inf  # exclude cut-offs from weighting

    y_flip = np.flip(dataY, axis=1)
    e_flip = np.flip(dataE, axis=1)

    w = 1.0 / (dataE ** 2)
    w_flip = 1.0 / (e_flip ** 2)

    dataYS = (dataY * w + y_flip * w_flip) / (w + w_flip)
    dataES = 1.0 / np.sqrt(w + w_flip)

    # Zero out entries that became NaN or Inf from inf-weight cut-offs
    bad = ~np.isfinite(dataYS) | ~np.isfinite(dataES)
    dataYS[bad] = 0.0
    dataES[bad] = 0.0

    return dataYS, dataES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def kinematics_arrays():
    """
    Generate synthetic ToF and instrument-parameter arrays matching the
    production layout: instr_pars columns = [det, plick, angle_deg, T0_us, L0_m, L1_m].
    """
    rng = np.random.default_rng(42)
    n_spec, n_tof = 64, 512
    dataX = np.tile(np.linspace(110.0, 430.0, n_tof), (n_spec, 1))
    instr_pars = np.column_stack([
        np.arange(n_spec, dtype=float),          # det
        np.zeros(n_spec),                         # plick (unused in kinematics)
        rng.uniform(30.0, 60.0, n_spec),          # angle [degrees]
        rng.uniform(-0.5, 0.5, n_spec),           # T0   [μs]
        np.full(n_spec, 11.0),                    # L0   [m]
        rng.uniform(0.3, 0.7, n_spec),            # L1   [m]
    ])
    return dataX, instr_pars


@pytest.fixture(scope="module")
def resolution_pars(kinematics_arrays):
    """
    Synthetic resolution parameters matching production layout:
    [dE1, dTOF, dTheta, dL0, dL1, dE1_lorz] (6 values per spectrum).
    """
    rng = np.random.default_rng(7)
    n_spec = kinematics_arrays[1].shape[0]
    dE1 = np.where(kinematics_arrays[1][:, 0] < 135, 88.7, 73.0)
    dE1_lorz = np.where(kinematics_arrays[1][:, 0] < 135, 40.3, 24.0)
    return np.column_stack([
        dE1,
        np.full(n_spec, 0.37),   # dTOF [μs]
        np.full(n_spec, 0.016),  # dTheta [rad]
        np.full(n_spec, 0.021),  # dL0 [m]
        np.full(n_spec, 0.023),  # dL1 [m]
        dE1_lorz,
    ])


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------

def test_kinematics_benchmark(benchmark, kinematics_arrays):
    """Benchmark kinematics array construction (∼ calculateKinematicsArrays)."""
    dataX, instr_pars = kinematics_arrays
    result = benchmark(_calculate_kinematics, dataX, instr_pars)
    v0, E0, delta_E, delta_Q = result
    assert v0.shape == dataX.shape


def test_gaussian_resolution_benchmark(benchmark, kinematics_arrays, resolution_pars):
    """
    Benchmark Gaussian resolution computation (∼ calcGaussianResolution).

    Uses a single representative v0/E0/delta_Q per spectrum (at the ToF
    mid-point), matching the per-spectrum calling pattern of the production
    code at y-space centres.
    """
    dataX, instr_pars = kinematics_arrays
    v0_full, E0_full, delta_E_full, delta_Q_full = _calculate_kinematics(dataX, instr_pars)
    # Take mid-point of ToF axis as a representative y-centre value
    mid = dataX.shape[1] // 2
    v0_c = v0_full[:, mid]          # (n_spec,)
    E0_c = E0_full[:, mid]
    delta_E_c = delta_E_full[:, mid]
    delta_Q_c = delta_Q_full[:, mid]

    masses = np.array([1.0079, 12.0, 16.0, 27.0])
    result = benchmark(
        _calc_gaussian_resolution,
        masses, v0_c, E0_c, delta_E_c, delta_Q_c, resolution_pars, instr_pars,
    )
    assert result.shape == (len(masses), dataX.shape[0])


def test_pseudo_voigt_benchmark(benchmark):
    """Benchmark pseudo-Voigt profile evaluation (∼ pseudoVoigt)."""
    x = np.linspace(-20.0, 20.0, 1024)
    result = benchmark(_pseudo_voigt, x, sigma=3.0, gamma=1.5)
    assert result.shape == x.shape


def test_weighted_sym_benchmark(benchmark):
    """Benchmark inverse-variance symmetrisation (∼ weightedSymArr, 2-D)."""
    rng = np.random.default_rng(0)
    n_spec, n_bins = 32, 512
    dataY = rng.standard_normal((n_spec, n_bins))
    dataE = rng.uniform(0.01, 0.5, (n_spec, n_bins))
    ysym, esym = benchmark(_weighted_sym_arr, dataY, dataE)
    assert ysym.shape == dataY.shape


# ---------------------------------------------------------------------------
# Regression gate
# ---------------------------------------------------------------------------

def test_no_regression_vs_baseline():
    """
    Documents the regression-gate strategy.

    The actual comparison is performed by the --benchmark-compare-fail CLI
    flag in CI (see .github/workflows/ci-main.yml).  This test is intentionally
    skipped so it never generates a false pass.
    """
    pytest.skip(
        "Regression gate handled by pytest-benchmark via --benchmark-compare-fail; "
        "this test exists only to document the CI workflow."
    )
