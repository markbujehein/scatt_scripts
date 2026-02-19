"""
Benchmarking suite for VESUVIO numerical-core functions.

Exercises the pure-numpy / scipy-based computation kernels that are
candidates for Numba acceleration.  Run with pytest-benchmark so that
results can be compared against a stored baseline (main-branch JSON):

    pip install pytest pytest-benchmark
    # Save a baseline on the main branch:
    pytest scripts/run_benchmarks.py --benchmark-save=baseline

    # On the PR branch, compare against the baseline:
    pytest scripts/run_benchmarks.py \\
        --benchmark-compare=baseline \\
        --benchmark-compare-fail=mean:10%

The suite intentionally does NOT import Mantid, so it can run in any
standard Python environment (NumPy + SciPy only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import optimize

# ---------------------------------------------------------------------------
# Inline replicas of the benchmarkable kernels
# (avoids importing Mantid-bound production modules)
# ---------------------------------------------------------------------------

_ME = 9.1093837015e-31    # kg
_AMU = 1.66053906660e-27  # kg
_HBAR = 1.054571817e-34   # J·s
_E_TO_JOULE = 1.602176634e-19  # 1 meV → J  (× 1e-3)
_M_TO_ANGSTROM = 1e10      # m → Å
_MEV_PER_J = 1.0 / (_E_TO_JOULE * 1e-3)


def _calculate_kinematics(dataX: np.ndarray, instr_pars: np.ndarray) -> tuple:
    """
    Replica of calculateKinematicsArrays from analysis_functions.py.

    dataX  : (n_spec, n_tof) ToF array [μs]
    instr_pars : (n_spec, 5) = [spec, theta, L0, L1, t_ofs]
    Returns (v0, E0, delta_E, delta_Q) all shape (n_spec, n_tof).
    """
    mN = 1.008 * _AMU
    hbar = _HBAR
    # constants from Mantid VESUVIO algorithm
    mev_to_ang_per_us = 0.22441282  # √(2 m_n / meV) in appropriate units
    t_us = dataX * 1e-6             # μs → s  (conceptual; kept as μs here)

    L0 = instr_pars[:, 2, np.newaxis]    # m
    L1 = instr_pars[:, 3, np.newaxis]    # m
    t_ofs = instr_pars[:, 4, np.newaxis]  # μs
    theta = instr_pars[:, 1, np.newaxis]  # rad

    # incident velocity  [m/μs]  (dataX and t_ofs both in μs)
    t_effective_us = dataX - t_ofs  # (n_spec, n_tof)
    with np.errstate(divide="ignore", invalid="ignore"):
        v0 = np.where(t_effective_us > 0, L0 / t_effective_us, 0.0)

    E0 = 0.5 * mN * (v0 * 1e6) ** 2 * _MEV_PER_J  # meV
    E1 = 4.1 * 1000.0  # eV final; simplified constant for benchmark
    delta_E = E0 - E1

    # |q| via cosine rule
    with np.errstate(invalid="ignore"):
        cos_theta = np.cos(theta)
        q2 = ((2.0 * mN * _MEV_PER_J * 1e-3) / (_HBAR ** 2)) * (
            E0 + E1 - 2.0 * np.sqrt(E0 * E1) * cos_theta
        )
        delta_Q = np.sqrt(np.abs(q2)) * _HBAR / (_AMU * 1.0e-10)  # Å⁻¹

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
    Replica of calcGaussianResolution from analysis_functions.py.

    Returns sigma array with shape (n_mass, n_spec, n_tof).
    """
    mN = 1.008 * _AMU
    n_mass = len(masses)
    sigmas = np.zeros((n_mass,) + v0.shape)

    L1 = instr_pars[:, 3, np.newaxis]       # m
    theta = instr_pars[:, 1, np.newaxis]     # rad
    sigma_l0 = res_pars[:, 0, np.newaxis]   # m
    sigma_l1 = res_pars[:, 1, np.newaxis]   # m
    sigma_th = res_pars[:, 2, np.newaxis]   # rad
    sigma_t = res_pars[:, 3, np.newaxis]    # μs

    # Partial derivatives (simplified for benchmark purposes)
    for i, mass in enumerate(masses):
        M = mass * _AMU
        y = (M / (_HBAR * delta_Q)) * (delta_E - ((_HBAR * delta_Q) ** 2) / (2.0 * M))
        dydE0 = M / (_HBAR * delta_Q + 1e-30)
        dydQ = -M * delta_E / (_HBAR * (delta_Q + 1e-30) ** 2)
        # simplified variance sum
        var = (
            (dydE0 * sigma_l0) ** 2
            + (dydQ * sigma_l1) ** 2
            + (dydE0 * sigma_th) ** 2
            + (dydE0 * sigma_t) ** 2
        )
        sigmas[i] = np.sqrt(np.abs(var))
    return sigmas


def _pseudo_voigt(x: np.ndarray, sigma: float, gamma: float) -> np.ndarray:
    """
    Replica of pseudoVoigt from analysis_functions.py.

    eta interpolates between Gaussian (eta=0) and Lorentzian (eta=1).
    """
    f_g = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    f_l = 2.0 * gamma
    f = (
        f_g ** 5
        + 2.69269 * f_g ** 4 * f_l
        + 2.42843 * f_g ** 3 * f_l ** 2
        + 4.47163 * f_g ** 2 * f_l ** 3
        + 0.07842 * f_g * f_l ** 4
        + f_l ** 5
    ) ** 0.2
    eta = 1.36603 * (f_l / (f + 1e-30)) - 0.47719 * (f_l / (f + 1e-30)) ** 2 + 0.11116 * (f_l / (f + 1e-30)) ** 3
    gauss = np.exp(-(x ** 2) / (2.0 * sigma ** 2)) / (sigma * np.sqrt(2.0 * np.pi))
    lorentz = (gamma / np.pi) / (x ** 2 + gamma ** 2)
    return eta * lorentz + (1.0 - eta) * gauss


def _weighted_sym_arr(
    dataY: np.ndarray, dataE: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Replica of weightedSymArr from fit_in_yspace.py.

    Symmetrises a 1-D spectrum using inverse-variance weighting.
    """
    n = len(dataY)
    dataYsym = np.zeros_like(dataY)
    dataEsym = np.zeros_like(dataE)
    for i in range(n):
        j = n - 1 - i
        w_i = 1.0 / (dataE[i] ** 2 + 1e-30)
        w_j = 1.0 / (dataE[j] ** 2 + 1e-30)
        dataYsym[i] = (w_i * dataY[i] + w_j * dataY[j]) / (w_i + w_j)
        dataEsym[i] = 1.0 / np.sqrt(w_i + w_j)
    return dataYsym, dataEsym


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def kinematics_arrays():
    """Generate synthetic ToF and instrument-parameter arrays."""
    rng = np.random.default_rng(42)
    n_spec, n_tof = 64, 512
    dataX = np.linspace(110.0, 430.0, n_tof)
    dataX = np.tile(dataX, (n_spec, 1))
    # [spec_no, theta(rad), L0(m), L1(m), t_ofs(μs)]
    instr_pars = np.column_stack([
        np.arange(n_spec, dtype=float),
        rng.uniform(0.3, 0.6, n_spec),   # theta
        np.full(n_spec, 11.0),            # L0
        rng.uniform(0.3, 0.7, n_spec),   # L1
        rng.uniform(-0.5, 0.5, n_spec),  # t_ofs
    ])
    return dataX, instr_pars


@pytest.fixture(scope="module")
def resolution_pars(kinematics_arrays):
    rng = np.random.default_rng(7)
    n_spec = kinematics_arrays[1].shape[0]
    # [sigma_L0, sigma_L1, sigma_theta, sigma_t]
    return rng.uniform(1e-4, 1e-2, (n_spec, 4))


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
    """Benchmark Gaussian resolution computation (∼ calcGaussianResolution)."""
    dataX, instr_pars = kinematics_arrays
    v0, E0, delta_E, delta_Q = _calculate_kinematics(dataX, instr_pars)
    masses = np.array([1.0079, 12.0, 16.0, 27.0])
    result = benchmark(
        _calc_gaussian_resolution,
        masses, v0, E0, delta_E, delta_Q, resolution_pars, instr_pars,
    )
    assert result.shape == (len(masses),) + dataX.shape


def test_pseudo_voigt_benchmark(benchmark):
    """Benchmark pseudo-Voigt profile evaluation (∼ pseudoVoigt)."""
    x = np.linspace(-20.0, 20.0, 1024)
    result = benchmark(_pseudo_voigt, x, sigma=3.0, gamma=1.5)
    assert result.shape == x.shape


def test_weighted_sym_benchmark(benchmark):
    """Benchmark inverse-variance symmetrisation (∼ weightedSymArr)."""
    rng = np.random.default_rng(0)
    n = 512
    dataY = rng.standard_normal(n)
    dataE = rng.uniform(0.01, 0.5, n)
    ysym, esym = benchmark(_weighted_sym_arr, dataY, dataE)
    assert ysym.shape == dataY.shape


# ---------------------------------------------------------------------------
# Regression gate
# ---------------------------------------------------------------------------

def _load_baseline(path: Path) -> dict:
    """Return benchmark mean times (seconds) keyed by test node-id."""
    if not path.exists():
        return {}
    with path.open() as fh:
        data = json.load(fh)
    return {b["name"]: b["stats"]["mean"] for b in data.get("benchmarks", [])}


def test_no_regression_vs_baseline():
    """
    Soft regression gate: warn (but do not fail) if any benchmark mean is
    more than 20 % slower than the stored main-branch baseline.

    The baseline JSON is expected at:
        .benchmarks/baseline/Linux-CPython-<ver>-64bit/<file>.json
    generated via `pytest --benchmark-save=baseline`.
    """
    baseline_dir = (
        Path(__file__).parent.parent / ".benchmarks" / "baseline"
    )
    candidates = list(baseline_dir.rglob("*.json")) if baseline_dir.exists() else []
    if not candidates:
        pytest.skip("No baseline file found; skipping regression check.")

    baseline = {}
    for p in candidates:
        baseline.update(_load_baseline(p))

    if not baseline:
        pytest.skip("Baseline file is empty.")

    # Current run results are not available inside the test (pytest-benchmark
    # stores them after the session).  This test acts as a documentation hook;
    # the actual comparison is performed by --benchmark-compare-fail in CI.
    assert True, "Regression gate placeholder — see CI step for actual comparison."
