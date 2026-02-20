"""Namespace verification tests for the vesuvio_analysis package.

These tests confirm that:
1. The top-level package entry points are importable via
   ``from vesuvio_analysis import <symbol>``.
2. All core sub-modules are reachable via absolute import paths starting
   from the ``vesuvio_analysis`` namespace.
3. Importing a Mantid-free sub-module (e.g. numba_routines) does NOT
   trigger the Mantid import chain — lazy loading must be in effect.

Runnable from the repository root *or* any parent directory once the
package is installed with ``pip install -e .``::

    python -m pytest tests/test_namespace.py -v
"""

from __future__ import annotations

import importlib
import sys
import unittest


class TestNoMantidOnPackageImport(unittest.TestCase):
    """Importing the top-level package must NOT pull in mantid prematurely."""

    def test_numba_routines_importable_without_mantid_mock(self) -> None:
        """numba_routines is Mantid-free; it must import cleanly before the mock."""
        # Remove any previously cached vesuvio_analysis modules to get a
        # clean slate for this specific check.
        to_drop = [k for k in sys.modules if k.startswith("vesuvio_analysis")]
        for key in to_drop:
            del sys.modules[key]

        # This must not raise ModuleNotFoundError for mantid
        mod = importlib.import_module(
            "vesuvio_analysis.core_functions.numba_routines"
        )
        self.assertIsNotNone(mod)


# Install the Mantid stub so Mantid-dependent modules can be imported without
# a full Mantid installation (mirrors the pattern used in all other CI tests).
from tests.mock_mantid import install as _install_mantid_mock  # noqa: E402

_install_mantid_mock()


class TestTopLevelEntryPoints(unittest.TestCase):
    """All __all__ symbols must be directly importable from the top-level package."""

    ENTRY_POINTS = [
        "runScript",
        "iterativeFitForDataReduction",
        "fitInYSpaceProcedure",
        "completeICFromInputs",
        "completeBootIC",
        "completeYFitIC",
        "buildFinalWSName",
        "runIndependentIterativeProcedure",
        "runJointBackAndForwardProcedure",
    ]

    def test_all_entry_points_in_package_all(self) -> None:
        import vesuvio_analysis

        for name in self.ENTRY_POINTS:
            with self.subTest(symbol=name):
                self.assertIn(name, vesuvio_analysis.__all__)

    def test_all_entry_points_callable(self) -> None:
        import vesuvio_analysis

        for name in self.ENTRY_POINTS:
            with self.subTest(symbol=name):
                obj = getattr(vesuvio_analysis, name)
                self.assertTrue(callable(obj))


class TestAbsoluteImports(unittest.TestCase):
    """Every core sub-module must be importable via its absolute path."""

    MODULES = [
        "vesuvio_analysis.core_functions.analysis_functions",
        "vesuvio_analysis.core_functions.fit_in_yspace",
        "vesuvio_analysis.core_functions.procedures",
        "vesuvio_analysis.core_functions.iminuit_costs",
        "vesuvio_analysis.core_functions.numba_routines",
        "vesuvio_analysis.core_functions.plot_style",
        "vesuvio_analysis.core_functions.run_script",
        "vesuvio_analysis.core_functions.ICHelpers",
        "vesuvio_analysis.core_functions.statistical_plugins",
        "vesuvio_analysis.core_functions.correction_plots",
        "vesuvio_analysis.core_functions.log_manager",
        "vesuvio_analysis.core_functions.stream_manager",
    ]

    def test_all_core_modules_importable(self) -> None:
        for module_name in self.MODULES:
            with self.subTest(module=module_name):
                mod = importlib.import_module(module_name)
                self.assertIsNotNone(mod)


if __name__ == "__main__":
    unittest.main(verbosity=2)
