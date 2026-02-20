"""Namespace verification tests for the vesuvio_analysis package.

These tests confirm that:
1. The top-level package entry point ``runScript`` is importable via
   ``from vesuvio_analysis import runScript``.
2. All core sub-modules are reachable via absolute import paths starting
   from the ``vesuvio_analysis`` namespace.

Runnable from the repository root *or* any parent directory once the
package is installed with ``pip install -e .``::

    python -m pytest tests/test_namespace.py -v
"""

from __future__ import annotations

import importlib
import unittest

# Install the Mantid stub so Mantid-dependent modules can be imported without
# a full Mantid installation (mirrors the pattern used in all other CI tests).
from tests.mock_mantid import install as _install_mantid_mock

_install_mantid_mock()


class TestTopLevelEntryPoint(unittest.TestCase):
    """runScript must be directly importable from the top-level package."""

    def test_run_script_importable_from_package(self) -> None:
        from vesuvio_analysis import runScript  # noqa: F401

        self.assertTrue(callable(runScript))

    def test_package_all_exports_run_script(self) -> None:
        import vesuvio_analysis

        self.assertIn("runScript", dir(vesuvio_analysis))


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
