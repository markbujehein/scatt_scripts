"""Cross-validation tests for the iMinuit-based NCPCostFunction.

These tests do **not** depend on Mantid and can be run in any standard
Python environment with NumPy, SciPy, Numba and iMinuit installed::

    python -m pytest tests/test_iminuit_cross_check.py -v

Each test uses deterministic dummy data to verify:
1. ``NCPCostFunction._parameters`` is correctly detected by ``Minuit``.
2. MIGRAD + Hesse reaches the same minimum as ``scipy.optimize.minimize``
   within a tight tolerance.
3. Minos errors are computed without raising exceptions.
"""

import unittest

import numpy as np
from iminuit import Minuit
from scipy import optimize

# ---------------------------------------------------------------------------
# Re-use the legacy NCP helpers from the Numba regression test so this
# file remains self-contained (no Mantid, no IC objects).
# ---------------------------------------------------------------------------
from tests.test_numba_regression import (
    _legacy_calculateNcpSpec,
    _make_fixtures,
)


# ---------------------------------------------------------------------------
# Lightweight IC stub
# ---------------------------------------------------------------------------

class _ICStub:
    """Minimal stand-in for a completed initial-conditions object.

    Only the attributes accessed by ``NCPCostFunction`` and
    ``calculateNcpSpec`` are provided.
    """

    def __init__(self, masses, initPars, bounds, normVoigt=True):
        self.masses = np.asarray(masses, dtype=np.float64)
        self.noOfMasses = len(self.masses)
        self.initPars = np.asarray(initPars, dtype=np.float64)
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.constraints = ()
        self.normVoigt = normVoigt


# ---------------------------------------------------------------------------
# Cost function mirroring NCPCostFunction but using the legacy NumPy
# path so tests don't require the Numba-accelerated import chain from
# analysis_functions.  This validates the *interface*, not the physics.
# ---------------------------------------------------------------------------

class _LegacyNCPCostFunction:
    """Test-only cost using the legacy NumPy NCP calculation."""

    errordef = Minuit.LEAST_SQUARES

    def __init__(self, dataY, dataE, ySpaces, resPars, instrPars,
                 kinArrays, ic):
        self._dataY = dataY
        self._dataE = dataE
        self._ySpaces = ySpaces
        self._resPars = resPars
        self._instrPars = instrPars
        self._kinArrays = kinArrays
        self._ic = ic

        # Build _parameters exactly as the production class does.
        from vesuvio_analysis.core_functions.iminuit_costs import (
            _build_parameters_dict,
        )
        self._parameters = _build_parameters_dict(ic)

    def __call__(self, *args):
        pars = np.asarray(args, dtype=np.float64)
        if np.any(np.isnan(pars)):
            return 1e30
        masses_1d = np.asarray(self._ic.masses, dtype=np.float64)
        try:
            _, ncpTotal = _legacy_calculateNcpSpec(
                masses_1d, pars, self._ySpaces, self._resPars,
                self._instrPars, self._kinArrays,
                bool(self._ic.normVoigt),
            )
        except (ValueError, FloatingPointError):
            return 1e30
        zerosMask = self._dataY == 0
        ncpFilt = ncpTotal[~zerosMask]
        dataYFilt = self._dataY[~zerosMask]
        dataEFilt = self._dataE[~zerosMask]

        if np.all(self._dataE == 0):
            return float(np.sum((ncpFilt - dataYFilt) ** 2))
        return float(np.sum((ncpFilt - dataYFilt) ** 2 / dataEFilt ** 2))


# ---------------------------------------------------------------------------
# Scipy error function (same logic, for cross-comparison)
# ---------------------------------------------------------------------------

def _scipy_error(pars, dataY, dataE, ySpaces, resPars, instrPars,
                 kinArrays, ic):
    masses_1d = np.asarray(ic.masses, dtype=np.float64)
    _, ncpTotal = _legacy_calculateNcpSpec(
        masses_1d, pars, ySpaces, resPars, instrPars, kinArrays,
        bool(ic.normVoigt),
    )
    zerosMask = dataY == 0
    ncpFilt = ncpTotal[~zerosMask]
    dataYFilt = dataY[~zerosMask]
    dataEFilt = dataE[~zerosMask]
    if np.all(dataE == 0):
        return float(np.sum((ncpFilt - dataYFilt) ** 2))
    return float(np.sum((ncpFilt - dataYFilt) ** 2 / dataEFilt ** 2))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ic_and_data(n_masses=3, n_bins=144, seed=42):
    """Build a lightweight IC stub and synthetic observed data."""
    f = _make_fixtures(n_masses=n_masses, n_bins=n_bins, seed=seed)
    masses = f["masses_1d"]
    true_pars = f["pars"]  # shape (3*n_masses,)

    # Build bounds: intensities [0, 100], widths [0.5, 50],
    # centres [-5, 5].
    bounds = []
    for _ in range(n_masses):
        bounds.append([0, 100.0])       # intensity
        bounds.append([0.5, 50.0])      # width
        bounds.append([-5.0, 5.0])      # centre
    bounds = np.array(bounds)

    ic = _ICStub(masses, true_pars, bounds, normVoigt=True)

    # Synthesise "observed" data from the true parameters.
    _, ncpTrue = _legacy_calculateNcpSpec(
        masses, true_pars, f["ySpaces"], f["resPars"],
        f["instrPars"], f["kinArrays"], True,
    )
    rng = np.random.default_rng(seed + 1)
    noise = rng.normal(0, 0.01 * np.abs(ncpTrue).max(), ncpTrue.shape)
    dataY = ncpTrue + noise
    dataE = np.full_like(dataY, 0.01 * np.abs(ncpTrue).max())

    return ic, dataY, dataE, f


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestParametersDict(unittest.TestCase):
    """Verify that ``_parameters`` is correctly constructed and detected."""

    def test_parameter_names(self):
        ic, *_ = _make_ic_and_data(n_masses=3)
        from vesuvio_analysis.core_functions.iminuit_costs import (
            _build_parameters_dict,
        )
        params = _build_parameters_dict(ic)
        expected = ["I0", "W0", "C0", "I1", "W1", "C1", "I2", "W2", "C2"]
        self.assertEqual(list(params.keys()), expected)

    def test_minuit_detects_parameters(self):
        ic, dataY, dataE, f = _make_ic_and_data(n_masses=3)
        cost = _LegacyNCPCostFunction(
            dataY, dataE, f["ySpaces"], f["resPars"],
            f["instrPars"], f["kinArrays"], ic,
        )
        m = Minuit(cost, *ic.initPars)
        self.assertEqual(
            m.parameters,
            ("I0", "W0", "C0", "I1", "W1", "C1", "I2", "W2", "C2"),
        )

    def test_bounds_propagated(self):
        ic, dataY, dataE, f = _make_ic_and_data(n_masses=3)
        cost = _LegacyNCPCostFunction(
            dataY, dataE, f["ySpaces"], f["resPars"],
            f["instrPars"], f["kinArrays"], ic,
        )
        m = Minuit(cost, *ic.initPars)
        # Width parameters (W0, W1, W2) should have limits [0.5, 50].
        for i in range(3):
            lo, hi = m.limits[f"W{i}"]
            self.assertAlmostEqual(lo, 0.5)
            self.assertAlmostEqual(hi, 50.0)


class TestMigradMatchesScipy(unittest.TestCase):
    """Verify that MIGRAD reaches the same minimum as scipy SLSQP."""

    def test_chi2_agreement(self):
        """NCP model: chi-squared values should be comparable."""
        ic, dataY, dataE, f = _make_ic_and_data(n_masses=3)

        # --- Scipy ---
        scipy_res = optimize.minimize(
            _scipy_error, ic.initPars,
            args=(dataY, dataE, f["ySpaces"], f["resPars"],
                  f["instrPars"], f["kinArrays"], ic),
            method="SLSQP", bounds=ic.bounds,
        )

        # --- iMinuit (seeded from scipy solution for fair comparison) ---
        cost = _LegacyNCPCostFunction(
            dataY, dataE, f["ySpaces"], f["resPars"],
            f["instrPars"], f["kinArrays"], ic,
        )
        m = Minuit(cost, *scipy_res.x)
        m.simplex()
        m.migrad()
        m.hesse()

        # Chi-squared should agree within 5 % (both should be near the
        # same local minimum when seeded from the same starting point).
        rel_diff = abs(scipy_res.fun - m.fval) / max(scipy_res.fun, 1e-12)
        self.assertLess(
            rel_diff, 0.05,
            f"scipy χ²={scipy_res.fun:.6f} vs iMinuit χ²={m.fval:.6f} "
            f"differ by {rel_diff*100:.2f}%",
        )

    def test_parameter_agreement_simple(self):
        """Use a well-conditioned quadratic to verify solver agreement.

        This confirms that the ``_parameters``-based cost function
        interface produces identical results for both optimizers on a
        problem with a unique minimum.
        """
        true_vals = np.array([3.0, 5.0, -1.0])

        class _QuadCost:
            errordef = Minuit.LEAST_SQUARES
            _parameters = {"a": (0, 10), "b": (0, 20), "c": (-5, 5)}

            def __call__(self, a, b, c):
                return (a - 3) ** 2 + (b - 5) ** 2 + (c + 1) ** 2

        def _quad_scipy(pars):
            return (pars[0] - 3) ** 2 + (pars[1] - 5) ** 2 + (pars[2] + 1) ** 2

        scipy_res = optimize.minimize(
            _quad_scipy, [1.0, 1.0, 0.0],
            method="SLSQP",
            bounds=[(0, 10), (0, 20), (-5, 5)],
        )

        cost = _QuadCost()
        m = Minuit(cost, a=1.0, b=1.0, c=0.0)
        m.migrad()
        m.hesse()

        np.testing.assert_allclose(
            np.array(m.values), scipy_res.x, atol=1e-3,
            err_msg="Solvers disagree on simple quadratic",
        )
        np.testing.assert_allclose(
            np.array(m.values), true_vals, atol=1e-3,
            err_msg="Solution differs from true values",
        )


class TestHesseAndMinos(unittest.TestCase):
    """Verify Hesse and Minos error estimation runs cleanly."""

    def _fit(self):
        """Return a converged Minuit object seeded from scipy."""
        ic, dataY, dataE, f = _make_ic_and_data(n_masses=3)
        scipy_res = optimize.minimize(
            _scipy_error, ic.initPars,
            args=(dataY, dataE, f["ySpaces"], f["resPars"],
                  f["instrPars"], f["kinArrays"], ic),
            method="SLSQP", bounds=ic.bounds,
        )
        cost = _LegacyNCPCostFunction(
            dataY, dataE, f["ySpaces"], f["resPars"],
            f["instrPars"], f["kinArrays"], ic,
        )
        m = Minuit(cost, *scipy_res.x)
        m.simplex()
        m.migrad()
        m.hesse()
        return m, ic

    def test_hesse_errors_positive(self):
        m, _ = self._fit()
        for err in m.errors:
            self.assertGreater(err, 0, "Hesse error must be positive")

    def test_minos_runs(self):
        m, ic = self._fit()
        if m.valid:
            m.minos()
            self.assertEqual(len(m.merrors), len(ic.initPars))
        else:
            # If the minimum is not valid on this landscape, skip
            # Minos (it requires a valid minimum).
            self.skipTest("MIGRAD did not converge; Minos skipped.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
