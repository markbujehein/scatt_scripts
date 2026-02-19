"""Tests for Console UX refinements introduced in the 'refine-console-ux' task.

These tests do **not** require a real Mantid installation; the lightweight
mock from ``tests/mock_mantid.py`` is installed before any vesuvio import.

Run with::

    python -m pytest tests/test_console_ux.py -v

Coverage:
1. ``_optimizer_check_log`` module-level accumulator is populated and
   cleared correctly.
2. ``_print_optimizer_agreement_summary`` prints PASS/FAIL with the
   correct difference and tolerance fields.
3. Separator constants (``_SEP_DOUBLE``, ``_SEP_SINGLE``) in
   ``run_script`` match the expected format.
4. ``verbose`` attribute is present on ``UserScriptControls`` in the
   BaH2_500C submission script.
"""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Install Mantid mock *before* any vesuvio import
# ---------------------------------------------------------------------------
from tests.mock_mantid import install as _install_mantid_mock
_install_mantid_mock()

# Stub out mantid.kernel so run_script's try/except import succeeds cleanly
import types as _types
if "mantid.kernel" not in sys.modules:
    _kernel_mod = _types.ModuleType("mantid.kernel")
    _kernel_mod.ConfigService = type("ConfigService", (), {"setLogLevel": staticmethod(lambda lvl: None)})()
    sys.modules["mantid.kernel"] = _kernel_mod

from vesuvio_analysis.core_functions import analysis_functions as _af
from vesuvio_analysis.core_functions import run_script as _rs


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _capture(fn, *args, **kwargs) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOptimizerCheckLog(unittest.TestCase):
    """Verify the module-level _optimizer_check_log accumulator."""

    def setUp(self):
        _af._optimizer_check_log.clear()

    def tearDown(self):
        _af._optimizer_check_log.clear()

    def test_log_empty_after_clear(self):
        _af._optimizer_check_log.append((0.1, 0.2))
        _af._optimizer_check_log.clear()
        self.assertEqual(len(_af._optimizer_check_log), 0)

    def test_log_accepts_tuples(self):
        _af._optimizer_check_log.append((0.005, 0.003))
        _af._optimizer_check_log.append((0.001, 0.002))
        self.assertEqual(len(_af._optimizer_check_log), 2)

    def test_summary_pass(self):
        """All spectra within threshold → PASS in output."""
        _af._optimizer_check_log.extend([(0.005, 0.003), (0.001, 0.002)])
        output = _capture(_af._print_optimizer_agreement_summary)
        self.assertIn("PASS", output)
        self.assertIn("iMinuit", output)
        self.assertIn("Tolerance: 1.00%", output)

    def test_summary_fail(self):
        """At least one spectrum over threshold → FAIL in output."""
        _af._optimizer_check_log.extend([(0.02, 0.005), (0.001, 0.002)])
        output = _capture(_af._print_optimizer_agreement_summary)
        self.assertIn("FAIL", output)

    def test_summary_empty_prints_nothing(self):
        """Empty log → nothing printed."""
        output = _capture(_af._print_optimizer_agreement_summary)
        self.assertEqual(output, "")

    def test_summary_shows_spectra_count(self):
        _af._optimizer_check_log.extend([(0.005, 0.0), (0.001, 0.0), (0.002, 0.0)])
        output = _capture(_af._print_optimizer_agreement_summary)
        self.assertIn("Spectra checked: 3", output)

    def test_summary_shows_failed_count(self):
        """Failed count == number of spectra that exceeded threshold."""
        _af._optimizer_check_log.extend([(0.02, 0.0), (0.005, 0.0)])
        output = _capture(_af._print_optimizer_agreement_summary)
        self.assertIn("Failed: 1", output)

    def test_summary_shows_max_difference(self):
        """Max difference percentage appears in the output."""
        _af._optimizer_check_log.extend([(0.005, 0.003)])
        output = _capture(_af._print_optimizer_agreement_summary)
        # max diff = 0.5%, formatted as XX.XX%
        self.assertIn("0.50%", output)


class TestSeparatorConstants(unittest.TestCase):
    """Verify ASCII separator constants in run_script."""

    def test_sep_double_is_60_equals(self):
        self.assertEqual(_rs._SEP_DOUBLE, "=" * 60)

    def test_sep_single_is_60_dashes(self):
        self.assertEqual(_rs._SEP_SINGLE, "-" * 60)


class TestVerboseFlagInSubmissionScript(unittest.TestCase):
    """Verify that UserScriptControls in BaH2_500C.py declares verbose."""

    def test_verbose_attribute_present(self):
        src_path = Path(__file__).parent.parent / "BaH2_500C.py"
        src = src_path.read_text(encoding="utf-8")
        self.assertIn("verbose", src)

    def test_verbose_default_true(self):
        src_path = Path(__file__).parent.parent / "BaH2_500C.py"
        src = src_path.read_text(encoding="utf-8")
        self.assertIn("verbose: bool = True", src)


class TestSavedNotificationInSource(unittest.TestCase):
    """Verify Saved: print statements are present in analysis_functions."""

    def test_saved_notification_in_plot_sum(self):
        src_path = (
            Path(__file__).parent.parent
            / "vesuvio_analysis" / "core_functions" / "analysis_functions.py"
        )
        src = src_path.read_text(encoding="utf-8")
        self.assertIn('print(f"Saved:', src)

    def test_mantid_log_suppression_in_run_script(self):
        src_path = (
            Path(__file__).parent.parent
            / "vesuvio_analysis" / "core_functions" / "run_script.py"
        )
        src = src_path.read_text(encoding="utf-8")
        self.assertIn("ConfigService", src)
        self.assertIn("setLogLevel", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
