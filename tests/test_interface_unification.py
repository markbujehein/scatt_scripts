"""Interface unification tests for Phase 3 cost-function classes.

These tests do **not** depend on Mantid and can be run in any standard
Python environment with NumPy, SciPy and iMinuit installed::

    python -m pytest tests/test_interface_unification.py -v

Each test confirms that:
1. All custom cost classes (NCP, Y-Space, Global) expose the unified
   interface: ``errordef``, ``ndata``, ``_parameters``, ``__call__``.
2. ``MyLeastSquares`` uses the modern ``_parameters`` dict (not the
   deprecated ``func_code``).
3. ``GlobalNCPCostFunction`` correctly integrates with ``CostSum``:
   - Shared parameters are merged into a single variable.
   - Local parameters retain unique suffixed names.
   - ``ndata`` is aggregated across groups.
4. ``NCPCostFunction`` provides a correct ``ndata`` property.
"""

import unittest

import numpy as np
from iminuit import Minuit, cost
from iminuit.util import describe


# ---------------------------------------------------------------------------
# Lightweight stubs — no Mantid dependency
# ---------------------------------------------------------------------------

class _ICStub:
    """Minimal stand-in for a completed initial-conditions object."""

    def __init__(self, masses, initPars, bounds, normVoigt=True):
        self.masses = np.asarray(masses, dtype=np.float64)
        self.noOfMasses = len(self.masses)
        self.initPars = np.asarray(initPars, dtype=np.float64)
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.constraints = ()
        self.normVoigt = normVoigt


# ---------------------------------------------------------------------------
# Standalone replica of MyLeastSquares (fit_in_yspace.py imports Mantid,
# so we replicate the class here identically for zero-Mantid testing).
# ---------------------------------------------------------------------------

class _MyLeastSquaresReplica:
    """Exact replica of ``MyLeastSquares`` from ``fit_in_yspace.py``.

    Validates that the *interface contract* (``_parameters``, ``ndata``,
    ``errordef``, ``__call__``) is satisfied by the current design.
    Uses ``describe(model, annotations=True)`` to propagate model parameter
    limits, following the official iminuit best practice
    (see: scikit-hep.org/iminuit/notebooks/generic_least_squares.html).
    """

    errordef = Minuit.LEAST_SQUARES

    def __init__(self, x, y, model):
        self.model = model
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        # Use annotations=True to propagate any type-annotation limits
        pars = describe(model, annotations=True)
        model_args = iter(pars)
        next(model_args)  # skip the first argument (independent variable x)
        self._parameters = {k: pars[k] for k in model_args}

    def __call__(self, *par):
        ym = self.model(self.x, *par)
        return np.sum((self.y - ym) ** 2)

    @property
    def ndata(self):
        return len(self.x)


# ---------------------------------------------------------------------------
# Test: MyLeastSquares unified interface
# ---------------------------------------------------------------------------

class TestMyLeastSquaresInterface(unittest.TestCase):
    """Verify that ``MyLeastSquares`` uses ``_parameters`` (not ``func_code``).

    Uses a standalone replica of the class to avoid importing Mantid.
    """

    def _make_cost(self):
        def model(x, a, b, c):
            return a * x ** 2 + b * x + c

        x = np.linspace(-5, 5, 20)
        y = 3 * x ** 2 - 2 * x + 1
        return _MyLeastSquaresReplica(x, y, model)

    def test_has_parameters_dict(self):
        c = self._make_cost()
        self.assertIsInstance(c._parameters, dict)
        self.assertEqual(list(c._parameters.keys()), ["a", "b", "c"])

    def test_no_func_code(self):
        c = self._make_cost()
        self.assertFalse(hasattr(c, "func_code"))

    def test_errordef(self):
        c = self._make_cost()
        self.assertEqual(c.errordef, Minuit.LEAST_SQUARES)

    def test_ndata(self):
        c = self._make_cost()
        self.assertEqual(c.ndata, 20)

    def test_callable(self):
        c = self._make_cost()
        val = c(3.0, -2.0, 1.0)
        self.assertAlmostEqual(val, 0.0, places=10)

    def test_minuit_detects_parameters(self):
        c = self._make_cost()
        m = Minuit(c, a=1, b=0, c=0)
        self.assertEqual(m.parameters, ("a", "b", "c"))

    def test_annotations_propagated(self):
        """Verify that type annotations (limits) are propagated through _parameters.

        This follows the official iminuit best practice from
        scikit-hep.org/iminuit/notebooks/generic_least_squares.html
        which uses ``describe(model, annotations=True)`` to capture
        parameter limits declared via ``Annotated`` types.
        """
        from typing import Annotated

        def model(x, a: float, b: Annotated[float, 0:]):
            return a + b * x

        x = np.linspace(-5, 5, 10)
        y = 1 + 2 * x
        c = _MyLeastSquaresReplica(x, y, model)

        # 'a' has no limits → None
        self.assertIsNone(c._parameters["a"])
        # 'b' has lower limit 0 → (0, inf)
        self.assertEqual(c._parameters["b"], (0, np.inf))


# ---------------------------------------------------------------------------
# Test: NCPCostFunction unified interface
# ---------------------------------------------------------------------------

class TestNCPCostFunctionInterface(unittest.TestCase):
    """Verify that ``NCPCostFunction`` exposes ``ndata`` and ``_parameters``."""

    def _make_ic(self):
        bounds = np.array([
            [0, 100.0], [0.5, 50.0], [-3.0, 3.0],
            [0, 100.0], [0.5, 50.0], [-3.0, 3.0],
        ])
        return _ICStub(
            masses=[1.008, 12.0],
            initPars=[5.0, 4.0, 0.0, 3.0, 5.0, 0.1],
            bounds=bounds,
        )

    def test_has_parameters(self):
        from vesuvio_analysis.core_functions.iminuit_costs import (
            NCPCostFunction,
            _build_parameters_dict,
        )
        ic = self._make_ic()
        params = _build_parameters_dict(ic)
        self.assertEqual(list(params.keys()), ["I0", "W0", "C0", "I1", "W1", "C1"])

    def test_ndata(self):
        from vesuvio_analysis.core_functions.iminuit_costs import NCPCostFunction

        ic = self._make_ic()
        # Create a fake dataY with some zeros
        dataY = np.array([0.0, 1.0, 2.0, 3.0, 0.0, 5.0])
        dataE = np.ones_like(dataY)
        fake_ySpaces = np.zeros((2, 6))
        fake_resPars = np.zeros(6)
        fake_instrPars = np.zeros(6)
        fake_kinArrays = np.zeros((4, 6))

        c = NCPCostFunction(dataY, dataE, fake_ySpaces, fake_resPars,
                            fake_instrPars, fake_kinArrays, ic)
        # ndata should count non-zero entries
        self.assertEqual(c.ndata, 4)

    def test_errordef(self):
        from vesuvio_analysis.core_functions.iminuit_costs import NCPCostFunction

        self.assertEqual(NCPCostFunction.errordef, Minuit.LEAST_SQUARES)


# ---------------------------------------------------------------------------
# Test: GlobalNCPCostFunction unified interface and CostSum integration
# ---------------------------------------------------------------------------

class TestGlobalNCPCostFunctionInterface(unittest.TestCase):
    """Verify ``GlobalNCPCostFunction`` interface and ``CostSum`` behaviour."""

    def _make_model_and_data(self, seed=42):
        """Build Gaussian models and synthetic data for two detector groups."""
        rng = np.random.default_rng(seed)
        x = np.linspace(-10, 10, 50)
        sigma_true = 3.0

        def model(x, y0, A, x0, sigma):
            return y0 + A / np.sqrt(2 * np.pi) / sigma * np.exp(
                -(x - x0) ** 2 / 2 / sigma ** 2
            )

        y0 = model(x, 0, 5, 0, sigma_true) + rng.normal(0, 0.05, len(x))
        y1 = model(x, 0, 3, 0, sigma_true) + rng.normal(0, 0.05, len(x))
        e = np.full_like(x, 0.05)
        return x, y0, y1, e, model

    def test_has_parameters(self):
        from vesuvio_analysis.core_functions.iminuit_costs import (
            GlobalNCPCostFunction,
        )
        x, y0, _, e, model = self._make_model_and_data()
        sig = ["x0", "y00", "A0", "x00", "sigma"]
        c = GlobalNCPCostFunction(x, y0, e, model, sig)
        self.assertIsInstance(c._parameters, dict)
        self.assertEqual(
            list(c._parameters.keys()), ["y00", "A0", "x00", "sigma"]
        )

    def test_errordef(self):
        from vesuvio_analysis.core_functions.iminuit_costs import (
            GlobalNCPCostFunction,
        )
        self.assertEqual(GlobalNCPCostFunction.errordef, Minuit.LEAST_SQUARES)

    def test_ndata(self):
        from vesuvio_analysis.core_functions.iminuit_costs import (
            GlobalNCPCostFunction,
        )
        x, y0, _, e, model = self._make_model_and_data()
        sig = ["x0", "y00", "A0", "x00", "sigma"]
        c = GlobalNCPCostFunction(x, y0, e, model, sig)
        self.assertEqual(c.ndata, 50)

    def test_callable(self):
        from vesuvio_analysis.core_functions.iminuit_costs import (
            GlobalNCPCostFunction,
        )
        x, y0, _, e, model = self._make_model_and_data()
        sig = ["x0", "y00", "A0", "x00", "sigma"]
        c = GlobalNCPCostFunction(x, y0, e, model, sig)
        val = c(0, 5, 0, 3)
        self.assertIsInstance(val, float)
        self.assertGreater(val, 0)


class TestCostSumIntegration(unittest.TestCase):
    """Verify that ``CostSum`` correctly identifies shared vs. local parameters."""

    def _make_costs(self):
        """Build two GlobalNCPCostFunction instances with shared 'sigma'."""
        from vesuvio_analysis.core_functions.iminuit_costs import (
            GlobalNCPCostFunction,
        )
        rng = np.random.default_rng(42)
        x = np.linspace(-10, 10, 30)
        sigma_true = 3.0

        def model_a(x, y00, A0, x00, sigma):
            return y00 + A0 / np.sqrt(2 * np.pi) / sigma * np.exp(
                -(x - x00) ** 2 / 2 / sigma ** 2
            )

        def model_b(x, y01, A1, x01, sigma):
            return y01 + A1 / np.sqrt(2 * np.pi) / sigma * np.exp(
                -(x - x01) ** 2 / 2 / sigma ** 2
            )

        y0 = model_a(x, 0, 5, 0, sigma_true) + rng.normal(0, 0.05, len(x))
        y1 = model_b(x, 0, 3, 0, sigma_true) + rng.normal(0, 0.05, len(x))
        e = np.full_like(x, 0.05)

        sig0 = ["x0", "y00", "A0", "x00", "sigma"]
        sig1 = ["x1", "y01", "A1", "x01", "sigma"]

        c0 = GlobalNCPCostFunction(x, y0, e, model_a, sig0)
        c1 = GlobalNCPCostFunction(x, y1, e, model_b, sig1)
        return c0, c1

    def test_costsum_type(self):
        c0, c1 = self._make_costs()
        total = c0 + c1
        self.assertIsInstance(total, cost.CostSum)

    def test_shared_params_merged(self):
        c0, c1 = self._make_costs()
        total = c0 + c1
        params = describe(total)
        # 'sigma' appears once (shared), others are unique
        self.assertEqual(params.count("sigma"), 1)
        self.assertIn("y00", params)
        self.assertIn("y01", params)
        self.assertIn("A0", params)
        self.assertIn("A1", params)

    def test_ndata_aggregated(self):
        c0, c1 = self._make_costs()
        total = c0 + c1
        self.assertEqual(total.ndata, c0.ndata + c1.ndata)
        self.assertEqual(total.ndata, 60)

    def test_len_counts_groups(self):
        c0, c1 = self._make_costs()
        total = c0 + c1
        self.assertEqual(len(total), 2)

    def test_costsum_parameter_count(self):
        c0, c1 = self._make_costs()
        total = c0 + c1
        # c0: y00, A0, x00, sigma (4 params)
        # c1: y01, A1, x01, sigma (4 params)
        # Merged: y00, A0, x00, sigma, y01, A1, x01 -> 7 unique
        params = describe(total)
        self.assertEqual(len(params), 7)

    def test_incremental_sum(self):
        """Verify ``totCost = 0; totCost += c`` works (as in global fit)."""
        c0, c1 = self._make_costs()
        totCost = 0
        totCost += c0
        totCost += c1
        self.assertIsInstance(totCost, cost.CostSum)
        self.assertEqual(len(totCost), 2)
        self.assertEqual(totCost.ndata, 60)

    def test_minuit_fit_converges(self):
        """Verify Minuit can fit with the summed cost and find the shared sigma."""
        c0, c1 = self._make_costs()
        total = c0 + c1

        m = Minuit(total, y00=0, A0=1, x00=0, y01=0, A1=1, x01=0, sigma=5)
        m.limits["A0"] = (0, None)
        m.limits["A1"] = (0, None)
        m.simplex()
        m.migrad()
        m.hesse()

        # Sigma should be close to true value 3.0
        self.assertAlmostEqual(m.values["sigma"], 3.0, delta=0.5)
        # Both amplitudes should be recovered
        self.assertAlmostEqual(m.values["A0"], 5.0, delta=1.0)
        self.assertAlmostEqual(m.values["A1"], 3.0, delta=1.0)

    def test_ndof_gof_metric(self):
        """Verify Minuit.ndof uses ndata for GoF (χ²/ndof) reporting.

        Per the iminuit docs: 'To support this feature, the cost function
        has to report the number of data points with a property called ndata.'
        (scikit-hep.org/iminuit/reference.html — Minuit.ndof)
        """
        c0, c1 = self._make_costs()
        total = c0 + c1

        m = Minuit(total, y00=0, A0=1, x00=0, y01=0, A1=1, x01=0, sigma=5)
        m.limits["A0"] = (0, None)
        m.limits["A1"] = (0, None)
        m.simplex()
        m.migrad()

        # ndof = ndata - npar = 60 - 7 = 53
        self.assertEqual(m.ndof, total.ndata - m.npar)
        self.assertEqual(m.ndof, 53)


class TestMultipleSharedParams(unittest.TestCase):
    """Verify CostSum with multiple shared parameters (GC_C4_C6 scenario)."""

    def test_gc_c4_c6_shared(self):
        from vesuvio_analysis.core_functions.iminuit_costs import (
            GlobalNCPCostFunction,
        )
        x = np.linspace(-10, 10, 30)

        def model_a(x, y00, A0, x00, sigma1, c4, c6):
            return y00 + A0 * np.exp(-(x - x00) ** 2 / 2 / sigma1 ** 2)

        def model_b(x, y01, A1, x01, sigma1, c4, c6):
            return y01 + A1 * np.exp(-(x - x01) ** 2 / 2 / sigma1 ** 2)

        y = np.ones_like(x)
        e = np.full_like(x, 0.1)

        # Shared: sigma1, c4, c6
        sig0 = ["x0", "y00", "A0", "x00", "sigma1", "c4", "c6"]
        sig1 = ["x1", "y01", "A1", "x01", "sigma1", "c4", "c6"]

        c0 = GlobalNCPCostFunction(x, y, e, model_a, sig0)
        c1 = GlobalNCPCostFunction(x, y, e, model_b, sig1)
        total = c0 + c1

        params = describe(total)
        # sigma1, c4, c6 each appear once
        self.assertEqual(params.count("sigma1"), 1)
        self.assertEqual(params.count("c4"), 1)
        self.assertEqual(params.count("c6"), 1)
        # Local params
        self.assertIn("y00", params)
        self.assertIn("y01", params)
        self.assertIn("A0", params)
        self.assertIn("A1", params)
        # 3 local per group (y0, A, x0) + 3 shared = 6 + 3 = 9
        self.assertEqual(len(params), 9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
