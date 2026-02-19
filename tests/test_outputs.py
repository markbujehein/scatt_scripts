"""Tests for the RunLogger metadata-logging system.

These tests do **not** depend on Mantid and can be run in any standard
Python environment with NumPy installed::

    python -m pytest tests/test_outputs.py -v

Each test verifies a distinct aspect of the RunLogger class:

1. A log file is created in the requested output directory.
2. All seven InitialConditions class headers appear in the log.
3. The environment block records Python, NumPy, and a Mantid entry.
4. The iMinuit–Scipy numerical agreement check is correctly recorded,
   including the ``overall_gate_passed`` status.
5. Final fit results (mean_widths / mean_intensities) are serialized.
6. GoF metrics (chi2, reduced_chi2, ndata) are captured.
7. Named timestamps (ncp_start, ncp_end, yspace_start, yspace_end) are
   written.
8. Errors are captured with a traceback when a fit step fails.
9. Boolean flags appear under the ``flags:`` block.
"""

import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np

from vesuvio_analysis.core_functions.log_manager import RunLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_logger(tmp_dir: Path, direction: str = "FORWARD") -> RunLogger:
    """Return a fresh RunLogger writing to *tmp_dir*."""
    return RunLogger(
        scriptName="thymol_10K_Gauss1D",
        direction=direction,
        output_dir=tmp_dir,
        timestamp="20260101_000000",
    )


def _read_log(logger: RunLogger) -> str:
    """Write the logger and return the full log text."""
    logger.write()
    return logger.logfile.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Stubs that mimic the seven IC classes used in thymol_10K_Gauss1D.py
# ---------------------------------------------------------------------------

class _UserScriptControls:
    runRoutine = True
    procedure = "JOINT"
    fitInYSpace = "FORWARD"


class _LoadVesuvioBackParameters:
    runs = "50888-50900"
    empty_runs = "51382-51415"
    spectra = "3-134"
    mode = "DoubleDifference"
    ipfile = Path("/fake/IP.par")


class _LoadVesuvioFrontParameters:
    runs = "50888-50900"
    empty_runs = "51382-51415"
    spectra = "135-182"
    mode = "SingleDifference"
    ipfile = Path("/fake/IP.par")


class _BackwardInitialConditions:
    masses = np.array([12.0, 16.0, 27.0])
    initPars = np.array([1, 4.9, 0.0, 1, 4.9, 0.0, 1, 9.27, 0.0])
    noOfMSIterations = 2
    firstSpec = 3
    lastSpec = 134
    MSCorrectionFlag = True
    GammaCorrectionFlag = False
    tofBinning = "110,1.,500"
    transmission_guess = 0.6
    HToMassIdxRatio = None


class _ForwardInitialConditions:
    masses = np.array([1.0079, 12.0, 16.0, 27.0])
    initPars = np.array([1, 5, 0.0, 1, 4.9, 0.0, 1, 4.9, 0.0, 1, 9.27, 0.0])
    noOfMSIterations = 2
    firstSpec = 135
    lastSpec = 182
    MSCorrectionFlag = True
    GammaCorrectionFlag = True
    tofBinning = "110,1,430"
    transmission_guess = 0.87


class _YSpaceFitInitialConditions:
    showPlots = False
    symmetrisationFlag = True
    rebinParametersForYSpaceFit = "-25, 0.5, 25"
    fitModel = "SINGLE_GAUSSIAN"
    runMinos = True
    globalFit = True
    nGlobalFitGroups = 3
    maskTypeProcedure = "NAN"


class _BootstrapInitialConditions:
    runBootstrap = False
    procedure = "BACKWARD"
    fitInYSpace = None
    bootstrapType = "BOOT_RESIDUALS"
    nSamples = 650
    skipMSIterations = False
    userConfirmation = True
    runningTest = False


# Fake results stub with all_mean_widths / all_mean_intensities
class _FakeResults:
    all_mean_widths = np.array([[4.9, 4.9, 9.27], [4.8, 4.8, 9.27]])
    all_mean_intensities = np.array([[0.5, 0.3, 0.2], [0.51, 0.31, 0.18]])


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

_SEVEN_IC_LABELS = [
    "UserScriptControls:",
    "LoadVesuvioBackParameters:",
    "LoadVesuvioFrontParameters:",
    "BackwardInitialConditions:",
    "ForwardInitialConditions:",
    "YSpaceFitInitialConditions:",
    "BootstrapInitialConditions:",
]


class TestLogFileCreation(unittest.TestCase):
    """Verify that the log file is created in the expected location."""

    def test_file_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(Path(tmp))
            path = _read_log(logger)
            self.assertTrue(
                logger.logfile.is_file(),
                f"Log file not found at {logger.logfile}",
            )

    def test_filename_convention(self):
        """Log file must follow the {scriptName}_{direction}_{timestamp}.log convention."""
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(Path(tmp), direction="BACKWARD")
            logger.write()
            self.assertEqual(
                logger.logfile.name,
                "thymol_10K_Gauss1D_BACKWARD_20260101_000000.log",
            )

    def test_output_dir_created_if_missing(self):
        """RunLogger must create the output directory when it does not exist."""
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "new_subdir" / "logs"
            logger = _make_logger(nested)
            logger.write()
            self.assertTrue(nested.is_dir())
            self.assertTrue(logger.logfile.is_file())


class TestAllSevenICsLogged(unittest.TestCase):
    """Verify that all seven IC class headers appear in the log."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        logger = _make_logger(Path(self._tmp.name))
        for label, ic in zip(
            _SEVEN_IC_LABELS,
            [
                _UserScriptControls,
                _LoadVesuvioBackParameters,
                _LoadVesuvioFrontParameters,
                _BackwardInitialConditions,
                _ForwardInitialConditions,
                _YSpaceFitInitialConditions,
                _BootstrapInitialConditions,
            ],
        ):
            # log_ic takes a name (without the colon)
            logger.log_ic(label.rstrip(":"), ic)
        self._content = _read_log(logger)

    def tearDown(self):
        self._tmp.cleanup()

    def test_all_seven_headers_present(self):
        for label in _SEVEN_IC_LABELS:
            self.assertIn(
                label, self._content,
                f"Expected IC header '{label}' not found in log",
            )

    def test_backward_masses_logged(self):
        self.assertIn("masses:", self._content)

    def test_mscorrection_flag_logged(self):
        self.assertIn("MSCorrectionFlag:", self._content)


class TestEnvironmentBlock(unittest.TestCase):
    """Verify that the environment block captures required version strings."""

    def _get_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(Path(tmp))
            logger.log_environment()
            return _read_log(logger)

    def test_environment_section_present(self):
        self.assertIn("environment:", self._get_content())

    def test_python_version_logged(self):
        content = self._get_content()
        self.assertIn("python_version:", content)
        # The recorded version must match the running interpreter
        self.assertIn(sys.version.split()[0], content)

    def test_numpy_version_logged(self):
        content = self._get_content()
        self.assertIn("numpy_version:", content)
        self.assertIn(np.__version__, content)

    def test_mantid_entry_present(self):
        """mantid_version key must appear even when Mantid is not installed."""
        content = self._get_content()
        self.assertIn("mantid_version:", content)


class TestAgreementGate(unittest.TestCase):
    """Verify iMinuit–Scipy numerical agreement check recording."""

    def _run(self, scipy_chi2, iminuit_chi2, scipy_pars, iminuit_pars):
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(Path(tmp))
            logger.log_agreement_gate(
                scipy_chi2, iminuit_chi2,
                np.asarray(scipy_pars), np.asarray(iminuit_pars),
            )
            return _read_log(logger)

    def test_gate_section_present(self):
        content = self._run(1.0, 1.0, [1.0, 2.0], [1.0, 2.0])
        self.assertIn("optimizer_agreement_check:", content)

    def test_overall_gate_passed_when_within_threshold(self):
        content = self._run(1.0, 1.005, [1.0, 2.0], [1.005, 2.01])
        self.assertIn("overall_gate_passed: True", content)

    def test_overall_gate_fails_when_chi2_exceeds_threshold(self):
        content = self._run(1.0, 1.05, [1.0, 2.0], [1.0, 2.0])
        self.assertIn("overall_gate_passed: False", content)

    def test_overall_gate_fails_when_pars_exceed_threshold(self):
        content = self._run(1.0, 1.0, [1.0, 2.0], [1.0, 2.5])
        self.assertIn("overall_gate_passed: False", content)

    def test_threshold_recorded(self):
        content = self._run(1.0, 1.0, [1.0], [1.0])
        self.assertIn("threshold: 0.01", content)

    def test_scipy_and_iminuit_chi2_recorded(self):
        content = self._run(2.5, 2.48, [1.0], [1.0])
        self.assertIn("scipy_chi2: 2.5", content)
        self.assertIn("iminuit_chi2: 2.48", content)


class TestFinalResultsLogging(unittest.TestCase):
    """Verify that final fit results are serialized correctly."""

    def _get_content(self, results):
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(Path(tmp))
            logger.log_final_results(results)
            return _read_log(logger)

    def test_final_results_section_present(self):
        content = self._get_content(_FakeResults())
        self.assertIn("final_results:", content)

    def test_mean_widths_logged(self):
        content = self._get_content(_FakeResults())
        self.assertIn("mean_widths:", content)

    def test_mean_intensities_logged(self):
        content = self._get_content(_FakeResults())
        self.assertIn("mean_intensities:", content)

    def test_null_results_handled(self):
        content = self._get_content(None)
        self.assertIn("final_results: null", content)

    def test_two_iterations_recorded(self):
        content = self._get_content(_FakeResults())
        self.assertIn("iteration_0:", content)
        self.assertIn("iteration_1:", content)


class TestGoFLogging(unittest.TestCase):
    """Verify GoF metric logging."""

    def _get_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(Path(tmp))
            logger.log_gof(chi2=12.34, reduced_chi2=0.987, ndata=100)
            return _read_log(logger)

    def test_gof_section(self):
        self.assertIn("goodness_of_fit:", self._get_content())

    def test_chi2_logged(self):
        self.assertIn("chi2: 12.34", self._get_content())

    def test_reduced_chi2_logged(self):
        self.assertIn("reduced_chi2: 0.987", self._get_content())

    def test_ndata_logged(self):
        self.assertIn("ndata: 100", self._get_content())


class TestTimestampLogging(unittest.TestCase):
    """Verify that named timestamps appear in the log."""

    def test_ncp_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(Path(tmp))
            logger.log_timestamp("ncp_start")
            logger.log_timestamp("ncp_end")
            content = _read_log(logger)
        self.assertIn("timestamp_ncp_start:", content)
        self.assertIn("timestamp_ncp_end:", content)

    def test_yspace_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(Path(tmp))
            logger.log_timestamp("yspace_start")
            logger.log_timestamp("yspace_end")
            content = _read_log(logger)
        self.assertIn("timestamp_yspace_start:", content)
        self.assertIn("timestamp_yspace_end:", content)


class TestErrorCapture(unittest.TestCase):
    """Verify that exceptions are captured with a traceback."""

    def test_error_section_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(Path(tmp))
            try:
                raise ValueError("test failure")
            except ValueError as exc:
                logger.log_error(exc)
            content = _read_log(logger)
        self.assertIn("error:", content)
        self.assertIn("ValueError", content)
        self.assertIn("test failure", content)


class TestFlagsLogging(unittest.TestCase):
    """Verify that boolean flags appear under the flags: block."""

    def test_flags_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(Path(tmp))
            logger.log_flags(
                MSCorrectionFlag=True,
                GammaCorrectionFlag=False,
                runRoutine=True,
            )
            content = _read_log(logger)
        self.assertIn("flags:", content)
        self.assertIn("MSCorrectionFlag:", content)
        self.assertIn("GammaCorrectionFlag:", content)
        self.assertIn("runRoutine:", content)


class TestCovarianceMatrixLogging(unittest.TestCase):
    """Verify covariance matrix logging for fit engines."""

    def _get_content(self, engine: str, matrix: np.ndarray) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(Path(tmp))
            logger.log_covariance_matrix(engine, matrix)
            return _read_log(logger)

    def test_iminuit_section_present(self):
        cov = np.eye(3)
        content = self._get_content("iminuit", cov)
        self.assertIn("covariance_matrix_iminuit:", content)

    def test_scipy_section_present(self):
        cov = np.eye(2)
        content = self._get_content("scipy", cov)
        self.assertIn("covariance_matrix_scipy:", content)

    def test_matrix_rows_logged(self):
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        content = self._get_content("iminuit", cov)
        self.assertIn("row_0:", content)
        self.assertIn("row_1:", content)

    def test_diagonal_values_present(self):
        cov = np.diag([3.14, 2.71])
        content = self._get_content("scipy", cov)
        self.assertIn("3.14", content)
        self.assertIn("2.71", content)


class TestBayesianPercentilesLogging(unittest.TestCase):
    """Verify Bayesian posterior percentile logging."""

    def _get_content(self, names, samples) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(Path(tmp))
            logger.log_bayesian_percentiles(names, samples)
            return _read_log(logger)

    def test_section_present(self):
        rng = np.random.default_rng(42)
        samples = rng.normal(5.0, 0.3, (200, 1))
        content = self._get_content(["width_H"], samples)
        self.assertIn("bayesian_percentiles:", content)

    def test_all_three_percentiles_present(self):
        rng = np.random.default_rng(0)
        samples = rng.normal(4.9, 0.2, (300, 1))
        content = self._get_content(["width_C"], samples)
        self.assertIn("p5:", content)
        self.assertIn("p50:", content)
        self.assertIn("p95:", content)

    def test_param_name_logged(self):
        rng = np.random.default_rng(1)
        samples = rng.normal(size=(100, 2))
        content = self._get_content(["intensity_H", "intensity_C"], samples)
        self.assertIn("intensity_H:", content)
        self.assertIn("intensity_C:", content)

    def test_multicolumn_samples(self):
        """Two-parameter samples should each have their percentiles logged."""
        rng = np.random.default_rng(7)
        samples = rng.normal(size=(500, 2))
        content = self._get_content(["w1", "w2"], samples)
        self.assertIn("w1:", content)
        self.assertIn("w2:", content)


class TestClusterMetadataLogging(unittest.TestCase):
    """Verify cluster metadata logging for DBSCAN output."""

    def _get_content(self, groups, noise, reason=None) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            logger = _make_logger(Path(tmp))
            kwargs = {} if reason is None else {"noise_reason": reason}
            logger.log_cluster_metadata(groups, noise, **kwargs)
            return _read_log(logger)

    def test_section_present(self):
        content = self._get_content({0: [1, 2, 3], 1: [4, 5]}, [])
        self.assertIn("cluster_metadata:", content)

    def test_cluster_ids_logged(self):
        content = self._get_content({0: [1, 2], 1: [3, 4]}, [])
        self.assertIn("cluster_0:", content)
        self.assertIn("cluster_1:", content)

    def test_noise_indices_logged(self):
        content = self._get_content({0: [1, 2]}, [10, 11])
        self.assertIn("noise_indices:", content)
        self.assertIn("10", content)
        self.assertIn("11", content)

    def test_empty_noise_logged(self):
        content = self._get_content({0: [1, 2, 3]}, [])
        self.assertIn("noise_indices: []", content)

    def test_custom_noise_reason(self):
        content = self._get_content(
            {0: [0, 1]}, [5],
            reason="Low signal-to-noise ratio",
        )
        self.assertIn("Low signal-to-noise ratio", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
