"""Regression tests comparing legacy NumPy functions vs Numba-accelerated paths.

This script does **not** depend on Mantid and can be run in any standard
Python environment with NumPy and Numba installed::

    python -m pytest tests/test_numba_regression.py -v

For each accelerated function the test:
1. Calls the original (pure-NumPy) implementation.
2. Calls the Numba ``@njit`` implementation.
3. Asserts ``np.allclose(atol=1e-8)``.
4. Prints wall-clock timing for both paths.
"""

import time
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# Re-implement the *original* NumPy functions inline so the test file is
# completely self-contained (no Mantid, no IC objects).
# ---------------------------------------------------------------------------


def _legacy_loadConstants():
    mN = 1.008
    Ef = 4906.0
    en_to_vel = 4.3737e-4
    vf = np.sqrt(Ef) * en_to_vel
    hbar = 2.0445
    return mN, Ef, en_to_vel, vf, hbar


def _legacy_gaussian(x, sigma):
    g = np.exp(-x ** 2 / 2 / sigma ** 2)
    g /= np.sqrt(2.0 * np.pi) * sigma
    return g


def _legacy_lorentizian(x, gamma):
    return gamma / np.pi / (x ** 2 + gamma ** 2)


def _legacy_pseudoVoigt(x, sigma, gamma, normVoigt):
    fg, fl = 2.0 * sigma * np.sqrt(2.0 * np.log(2.0)), 2.0 * gamma
    f = 0.5346 * fl + np.sqrt(0.2166 * fl ** 2 + fg ** 2)
    eta = 1.36603 * fl / f - 0.47719 * (fl / f) ** 2 + 0.11116 * (fl / f) ** 3
    sigma_v = f / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma_v = f / 2.0
    pv = eta * _legacy_lorentizian(x, gamma_v) + (1.0 - eta) * _legacy_gaussian(x, sigma_v)
    if normVoigt:
        norm = np.abs(np.trapz(pv, x, axis=1))[:, np.newaxis]
        pv = pv / norm
    return pv


def _legacy_numericalThirdDerivative(x, fun):
    k6 = (-fun[:, 12:] + fun[:, :-12]) * 1
    k5 = (+fun[:, 11:-1] - fun[:, 1:-11]) * 24
    k4 = (-fun[:, 10:-2] + fun[:, 2:-10]) * 192
    k3 = (+fun[:, 9:-3] - fun[:, 3:-9]) * 488
    k2 = (+fun[:, 8:-4] - fun[:, 4:-8]) * 387
    k1 = (-fun[:, 7:-5] + fun[:, 5:-7]) * 1584
    dev = k1 + k2 + k3 + k4 + k5 + k6
    dev /= np.power(x[:, 7:-5] - x[:, 6:-6], 3)
    dev /= 12 ** 3
    derivative = np.zeros(fun.shape)
    derivative[:, 6:-6] = dev
    return derivative


def _legacy_kinematicsAtYCenters(ySpaces, centers, kinArrays):
    shape = centers.shape
    prox = np.abs(ySpaces - centers)
    yClosest = prox.min(axis=1).reshape(shape)
    mask = prox == yClosest
    v0, E0, dE, dQ = kinArrays
    v0 = (v0 * np.ones(shape))[mask].reshape(shape)
    E0 = (E0 * np.ones(shape))[mask].reshape(shape)
    dE = (dE * np.ones(shape))[mask].reshape(shape)
    dQ = (dQ * np.ones(shape))[mask].reshape(shape)
    return v0, E0, dE, dQ


def _legacy_calcGaussianResolution(masses, v0, E0, delta_E, delta_Q,
                                   resPars, instrPars):
    det, plick, angle, T0, L0, L1 = instrPars
    dE1, dTOF, dTheta, dL0, dL1, dE1_lorz = resPars
    mN, Ef, en_to_vel, vf, hbar = _legacy_loadConstants()
    angle = angle * np.pi / 180
    dWdE1 = 1.0 + (E0 / Ef) ** 1.5 * (L1 / L0)
    dWdTOF = 2.0 * E0 * v0 / L0
    dWdL1 = 2.0 * E0 ** 1.5 / Ef ** 0.5 / L0
    dWdL0 = 2.0 * E0 / L0
    dW2 = (dWdE1 ** 2 * dE1 ** 2 + dWdTOF ** 2 * dTOF ** 2
           + dWdL1 ** 2 * dL1 ** 2 + dWdL0 ** 2 * dL0 ** 2)
    dW2 *= (masses / hbar ** 2 / delta_Q) ** 2
    dQdE1 = (1.0 - (E0 / Ef) ** 1.5 * L1 / L0
             - np.cos(angle) * ((E0 / Ef) ** 0.5 - L1 / L0 * E0 / Ef))
    dQdTOF = 2.0 * E0 * v0 / L0
    dQdL1 = 2.0 * E0 ** 1.5 / L0 / Ef ** 0.5
    dQdL0 = 2.0 * E0 / L0
    dQdTheta = 2.0 * np.sqrt(E0 * Ef) * np.sin(angle)
    dQ2 = (dQdE1 ** 2 * dE1 ** 2
           + (dQdTOF ** 2 * dTOF ** 2 + dQdL1 ** 2 * dL1 ** 2
              + dQdL0 ** 2 * dL0 ** 2)
           * np.abs(Ef / E0 * np.cos(angle) - 1)
           + dQdTheta ** 2 * dTheta ** 2)
    dQ2 *= (mN / hbar ** 2 / delta_Q) ** 2
    return np.sqrt(dW2 + dQ2)


def _legacy_calcLorentzianResolution(masses, v0, E0, delta_E, delta_Q,
                                     resPars, instrPars):
    det, plick, angle, T0, L0, L1 = instrPars
    dE1, dTOF, dTheta, dL0, dL1, dE1_lorz = resPars
    mN, Ef, en_to_vel, vf, hbar = _legacy_loadConstants()
    angle = angle * np.pi / 180
    dWdE1_lor = (1.0 + (E0 / Ef) ** 1.5 * (L1 / L0)) ** 2
    dWdE1_lor *= (masses / hbar ** 2 / delta_Q) ** 2
    dQdE1_lor = (1.0 - (E0 / Ef) ** 1.5 * L1 / L0
                 - np.cos(angle) * ((E0 / Ef) ** 0.5 + L1 / L0 * E0 / Ef)) ** 2
    dQdE1_lor *= (mN / hbar ** 2 / delta_Q) ** 2
    return np.sqrt(dWdE1_lor + dQdE1_lor) * dE1_lorz


def _legacy_calculateNcpSpec(masses_1d, pars, ySpaces, resPars, instrPars,
                             kinArrays, normVoigt):
    n_masses = len(masses_1d)
    masses = masses_1d[:, np.newaxis]
    intensities = pars[::3].reshape(masses.shape)
    widths = pars[1::3].reshape(masses.shape)
    centers = pars[2::3].reshape(masses.shape)
    v0, E0, deltaE, deltaQ = _legacy_kinematicsAtYCenters(ySpaces, centers, kinArrays)
    gaussRes = _legacy_calcGaussianResolution(
        masses, v0, E0, deltaE, deltaQ, resPars, instrPars)
    lorzRes = _legacy_calcLorentzianResolution(
        masses, v0, E0, deltaE, deltaQ, resPars, instrPars)
    totalGaussWidth = np.sqrt(widths ** 2 + gaussRes ** 2)
    JOfY = _legacy_pseudoVoigt(ySpaces - centers, totalGaussWidth, lorzRes, normVoigt)
    FSE = -_legacy_numericalThirdDerivative(ySpaces, JOfY) * widths ** 4 / deltaQ * 0.72
    ncpM = intensities * (JOfY + FSE) * E0 * E0 ** (-0.92) * masses / deltaQ
    ncpT = np.sum(ncpM, axis=0)
    return ncpM, ncpT


# ---------------------------------------------------------------------------
# Numba imports
# ---------------------------------------------------------------------------
from vesuvio_analysis.core_functions.numba_routines import (
    loadConstants as nb_loadConstants,
    gaussian as nb_gaussian,
    lorentizian as nb_lorentizian,
    pseudoVoigt as nb_pseudoVoigt,
    numericalThirdDerivative as nb_numericalThirdDerivative,
    kinematicsAtYCenters as nb_kinematicsAtYCenters,
    calcGaussianResolution as nb_calcGaussianResolution,
    calcLorentzianResolution as nb_calcLorentzianResolution,
    calculateNcpSpec_numba,
)


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

def _make_fixtures(n_masses=3, n_bins=144, seed=42):
    """Create reproducible fixture arrays mimicking VESUVIO DINS data.

    The Neutron Compton Profile (NCP) is a Voigt function centred at
    y ≈ 0 in the West scaling variable.  Accordingly:

    * **y-spaces** are linearly spaced and centred at zero.  The range
      narrows with increasing mass (heavier nuclei have narrower
      momentum distributions under the impulse approximation).
    * **Kinematic arrays** ``[v0, E0, ΔE, ΔQ]`` are smooth,
      monotonically varying, and derived from the VESUVIO TOF→energy
      conversion (Ef = 4906 meV gold-foil analyser, L0 = 11 m,
      L1 = 0.5 m, back-scattering angle 135°).
    * **Fit parameters** use NCP centres near zero and Gaussian widths
      that grow with ``√mass`` (Debye-solid approximation).
    """
    rng = np.random.default_rng(seed)

    # --- Atomic masses (H, C, O by default) ---
    masses_1d = np.array([1.008, 12.0, 16.0])[:n_masses]
    masses_col = masses_1d[:, np.newaxis]

    # --- Y-space grids: linearly spaced, centred at y = 0 ---
    ySpaces = np.zeros((n_masses, n_bins))
    for i, m in enumerate(masses_1d):
        half_range = 30.0 / np.sqrt(m / masses_1d[0])
        ySpaces[i] = np.linspace(-half_range, half_range, n_bins)

    # --- NCP centres: near y = 0 (impulse-approximation origin) ---
    centers = rng.uniform(-0.5, 0.5, (n_masses, 1))

    # --- Kinematic arrays [v0, E0, deltaE, deltaQ] ---
    Ef = 4906.0                          # meV  (gold-foil analyser)
    en_to_vel = 4.3737e-4                # √meV → m/µs
    L0, L1 = 11.0, 0.5                  # primary / secondary flight paths (m)
    vf = np.sqrt(Ef) * en_to_vel
    t1 = L1 / vf                         # secondary flight-path time (µs)
    tof = np.linspace(130, 330, n_bins)   # TOF range where E0 > Ef
    v0_arr = L0 / (tof - t1)
    E0_arr = (v0_arr / en_to_vel) ** 2
    deltaE_arr = E0_arr - Ef
    angle_rad = 135.0 * np.pi / 180
    deltaQ_arr = np.sqrt(
        E0_arr + Ef - 2.0 * np.sqrt(E0_arr * Ef) * np.cos(angle_rad)
    )
    kinArrays = np.array([
        v0_arr,
        E0_arr,
        deltaE_arr,
        deltaQ_arr + 1.0,
    ])

    # --- Instrument parameters [det, plick, angle, T0, L0, L1] ---
    instrPars = np.array([3.0, 1.0, 135.0, 0.0, 11.0, 0.5])

    # --- Resolution parameters [dE1, dTOF, dTheta, dL0, dL1, dE1_lorz] ---
    resPars = np.array([0.1, 0.02, 0.005, 0.01, 0.01, 0.05])

    # --- Fit parameters [I0, W0, C0, I1, W1, C1, ...] ---
    pars = np.zeros(3 * n_masses)
    for i, m in enumerate(masses_1d):
        pars[3 * i + 0] = rng.uniform(1.0, 8.0)          # intensity
        pars[3 * i + 1] = 4.0 + 0.5 * np.sqrt(m)         # width (Å⁻¹)
        pars[3 * i + 2] = rng.uniform(-0.3, 0.3)          # centre ≈ 0

    return dict(
        ySpaces=ySpaces, centers=centers, kinArrays=kinArrays,
        masses_1d=masses_1d, masses_col=masses_col,
        resPars=resPars, instrPars=instrPars, pars=pars,
    )


# ---------------------------------------------------------------------------
# Helper to time a callable
# ---------------------------------------------------------------------------

def _bench(label, fn, n_calls=500):
    """Run *fn* n_calls times, return (result, elapsed_seconds)."""
    fn()  # warm-up / JIT compile
    t0 = time.perf_counter()
    for _ in range(n_calls):
        result = fn()
    elapsed = time.perf_counter() - t0
    print(f"  {label:40s}  {n_calls} calls in {elapsed:.4f}s "
          f"({elapsed / n_calls * 1e6:.1f} µs/call)")
    return result, elapsed


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestLoadConstants(unittest.TestCase):
    def test_values_match(self):
        legacy = _legacy_loadConstants()
        numba = nb_loadConstants()
        for a, b in zip(legacy, numba):
            self.assertAlmostEqual(a, b, places=12)


class TestGaussian(unittest.TestCase):
    def test_regression(self):
        f = _make_fixtures()
        x = f["ySpaces"]
        sigma = np.abs(f["centers"]) + 1.0
        leg = _legacy_gaussian(x, sigma)
        nb = nb_gaussian(x, sigma)
        np.testing.assert_allclose(nb, leg, atol=1e-8)

    def test_benchmark(self):
        f = _make_fixtures()
        x = f["ySpaces"]
        sigma = np.abs(f["centers"]) + 1.0
        print()
        _bench("gaussian (NumPy)", lambda: _legacy_gaussian(x, sigma))
        _bench("gaussian (Numba)", lambda: nb_gaussian(x, sigma))


class TestLorentizian(unittest.TestCase):
    def test_regression(self):
        f = _make_fixtures()
        x = f["ySpaces"]
        gamma = np.abs(f["centers"]) + 1.0
        leg = _legacy_lorentizian(x, gamma)
        nb = nb_lorentizian(x, gamma)
        np.testing.assert_allclose(nb, leg, atol=1e-8)

    def test_benchmark(self):
        f = _make_fixtures()
        x = f["ySpaces"]
        gamma = np.abs(f["centers"]) + 1.0
        print()
        _bench("lorentizian (NumPy)", lambda: _legacy_lorentizian(x, gamma))
        _bench("lorentizian (Numba)", lambda: nb_lorentizian(x, gamma))


class TestPseudoVoigt(unittest.TestCase):
    def test_regression_normalised(self):
        f = _make_fixtures()
        x = f["ySpaces"]
        sigma = np.abs(f["centers"]) + 1.0
        gamma = np.abs(f["centers"]) * 0.5 + 0.5
        leg = _legacy_pseudoVoigt(x, sigma, gamma, True)
        nb = nb_pseudoVoigt(x, sigma, gamma, True)
        np.testing.assert_allclose(nb, leg, atol=1e-8)

    def test_regression_unnormalised(self):
        f = _make_fixtures()
        x = f["ySpaces"]
        sigma = np.abs(f["centers"]) + 1.0
        gamma = np.abs(f["centers"]) * 0.5 + 0.5
        leg = _legacy_pseudoVoigt(x, sigma, gamma, False)
        nb = nb_pseudoVoigt(x, sigma, gamma, False)
        np.testing.assert_allclose(nb, leg, atol=1e-8)

    def test_benchmark(self):
        f = _make_fixtures()
        x = f["ySpaces"]
        sigma = np.abs(f["centers"]) + 1.0
        gamma = np.abs(f["centers"]) * 0.5 + 0.5
        print()
        _bench("pseudoVoigt (NumPy)", lambda: _legacy_pseudoVoigt(x, sigma, gamma, True))
        _bench("pseudoVoigt (Numba)", lambda: nb_pseudoVoigt(x, sigma, gamma, True))


class TestNumericalThirdDerivative(unittest.TestCase):
    def test_regression(self):
        f = _make_fixtures()
        x = f["ySpaces"]
        fun = _legacy_gaussian(x, np.abs(f["centers"]) + 1.0)
        leg = _legacy_numericalThirdDerivative(x, fun)
        nb = nb_numericalThirdDerivative(x, fun)
        np.testing.assert_allclose(nb, leg, atol=1e-8)

    def test_benchmark(self):
        f = _make_fixtures()
        x = f["ySpaces"]
        fun = _legacy_gaussian(x, np.abs(f["centers"]) + 1.0)
        print()
        _bench("thirdDeriv (NumPy)", lambda: _legacy_numericalThirdDerivative(x, fun))
        _bench("thirdDeriv (Numba)", lambda: nb_numericalThirdDerivative(x, fun))


class TestKinematicsAtYCenters(unittest.TestCase):
    def test_regression(self):
        f = _make_fixtures()
        leg = _legacy_kinematicsAtYCenters(f["ySpaces"], f["centers"], f["kinArrays"])
        nb = nb_kinematicsAtYCenters(f["ySpaces"], f["centers"], f["kinArrays"])
        for a, b in zip(leg, nb):
            np.testing.assert_allclose(b, a, atol=1e-8)

    def test_benchmark(self):
        f = _make_fixtures()
        print()
        _bench("kinematicsAtYCenters (NumPy)",
               lambda: _legacy_kinematicsAtYCenters(f["ySpaces"], f["centers"], f["kinArrays"]))
        _bench("kinematicsAtYCenters (Numba)",
               lambda: nb_kinematicsAtYCenters(f["ySpaces"], f["centers"], f["kinArrays"]))


class TestCalcGaussianResolution(unittest.TestCase):
    def test_regression(self):
        f = _make_fixtures()
        v0, E0, dE, dQ = _legacy_kinematicsAtYCenters(
            f["ySpaces"], f["centers"], f["kinArrays"])
        leg = _legacy_calcGaussianResolution(
            f["masses_col"], v0, E0, dE, dQ, f["resPars"], f["instrPars"])
        nb = nb_calcGaussianResolution(
            f["masses_col"], v0, E0, dE, dQ, f["resPars"], f["instrPars"])
        np.testing.assert_allclose(nb, leg, atol=1e-8)


class TestCalcLorentzianResolution(unittest.TestCase):
    def test_regression(self):
        f = _make_fixtures()
        v0, E0, dE, dQ = _legacy_kinematicsAtYCenters(
            f["ySpaces"], f["centers"], f["kinArrays"])
        leg = _legacy_calcLorentzianResolution(
            f["masses_col"], v0, E0, dE, dQ, f["resPars"], f["instrPars"])
        nb = nb_calcLorentzianResolution(
            f["masses_col"], v0, E0, dE, dQ, f["resPars"], f["instrPars"])
        np.testing.assert_allclose(nb, leg, atol=1e-8)


class TestCalculateNcpSpec(unittest.TestCase):
    """End-to-end regression for the full NCP orchestrator."""

    def test_regression_normalised(self):
        f = _make_fixtures()
        legM, legT = _legacy_calculateNcpSpec(
            f["masses_1d"], f["pars"], f["ySpaces"],
            f["resPars"], f["instrPars"], f["kinArrays"], True)
        nbM, nbT = calculateNcpSpec_numba(
            f["masses_1d"], f["pars"], f["ySpaces"],
            f["resPars"], f["instrPars"], f["kinArrays"], True)
        np.testing.assert_allclose(nbM, legM, atol=1e-8)
        np.testing.assert_allclose(nbT, legT, atol=1e-8)

    def test_regression_unnormalised(self):
        f = _make_fixtures()
        legM, legT = _legacy_calculateNcpSpec(
            f["masses_1d"], f["pars"], f["ySpaces"],
            f["resPars"], f["instrPars"], f["kinArrays"], False)
        nbM, nbT = calculateNcpSpec_numba(
            f["masses_1d"], f["pars"], f["ySpaces"],
            f["resPars"], f["instrPars"], f["kinArrays"], False)
        np.testing.assert_allclose(nbM, legM, atol=1e-8)
        np.testing.assert_allclose(nbT, legT, atol=1e-8)

    def test_benchmark(self):
        f = _make_fixtures()
        print()
        _bench("calculateNcpSpec (NumPy)", lambda: _legacy_calculateNcpSpec(
            f["masses_1d"], f["pars"], f["ySpaces"],
            f["resPars"], f["instrPars"], f["kinArrays"], True))
        _bench("calculateNcpSpec (Numba)", lambda: calculateNcpSpec_numba(
            f["masses_1d"], f["pars"], f["ySpaces"],
            f["resPars"], f["instrPars"], f["kinArrays"], True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
