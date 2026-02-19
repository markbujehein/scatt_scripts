"""Persistent metadata logging for the VESUVIO analysis pipeline.

Provides ``RunLogger``, a class that captures and writes a timestamped,
human-readable (YAML-like) log of every analysis run, recording the full
experimental state (Initial Conditions, Flags, Environment) and numerical
outcomes (Fit Results, GoF metrics, Agreement Gate status).

Log files are stored in the run's output directory and named::

    {scriptName}_{DIRECTION}_{timestamp}.log

Usage::

    logger = RunLogger(scriptName="thymol_10K_Gauss1D",
                       direction="FORWARD",
                       output_dir=Path("experiments/thymol_10K_Gauss1D/output_npz_for_testing"))
    logger.log_environment()
    logger.log_ic("UserScriptControls", userCtr)
    logger.log_ic("BackwardInitialConditions", bckwdIC)
    # ... log other ICs ...
    logger.log_timestamp("ncp_start")
    # ... run procedure ...
    logger.log_timestamp("ncp_end")
    logger.log_final_results(results)
    logger.log_gof(chi2=1.23, reduced_chi2=0.98, ndata=100)
    logger.log_agreement_gate(scipy_chi2=1.23, iminuit_chi2=1.22,
                              scipy_pars=np.array([...]),
                              iminuit_pars=np.array([...]))
    logger.write()
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np


# Threshold for the Sieve-3 numerical agreement gate
_AGREEMENT_THRESHOLD: float = 0.01


class RunLogger:
    """Captures and writes a structured log of a VESUVIO analysis run.

    The log is buffered in memory and flushed to disk via :meth:`write`.
    Each section is delimited by a YAML-like header so the file can be
    parsed programmatically with a simple line-by-line reader.

    Args:
        scriptName: Base name of the submission script (without ``.py``).
        direction: Scattering direction string – ``"BACKWARD"``,
            ``"FORWARD"``, ``"JOINT"``, or ``"NONE"``.
        output_dir: Directory where the log file is written.  Created
            if it does not exist.
        timestamp: Optional ISO-format timestamp string (``YYYYMMDD_HHMMSS``).
            When ``None`` (default), the current wall-clock time is used.
            Providing a fixed value is useful for deterministic testing.
    """

    def __init__(
        self,
        scriptName: str,
        direction: str,
        output_dir: Any,
        timestamp: Optional[str] = None,
    ) -> None:
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._timestamp = timestamp
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logfile: Path = output_dir / f"{scriptName}_{direction}_{timestamp}.log"
        self._lines: list[str] = []
        self._lines.append(f"# VESUVIO Run Log")
        self._lines.append(f"# Script   : {scriptName}")
        self._lines.append(f"# Direction: {direction}")
        self._lines.append(f"# Timestamp: {timestamp}")
        self._lines.append("")

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def log_environment(self) -> None:
        """Record Python, NumPy, and Mantid version strings."""
        self._lines.append("environment:")
        self._lines.append(f"  python_version: {sys.version.split()[0]}")
        self._lines.append(f"  numpy_version: {np.__version__}")
        try:
            import mantid
            mantid_ver = mantid.__version__
        except ImportError:
            mantid_ver: str = "unavailable"
        self._lines.append(f"  mantid_version: {mantid_ver}")
        self._lines.append("")

    # ------------------------------------------------------------------
    # Initial Conditions
    # ------------------------------------------------------------------

    def log_ic(self, name: str, ic: Any) -> None:
        """Serialize one InitialConditions class as a YAML-like block.

        Only non-callable, non-dunder attributes are recorded.
        ``numpy.ndarray`` values are formatted as compact inline lists.
        Works with both class objects (``type``) and instances.

        Args:
            name: Human-readable class label (e.g. ``"BackwardInitialConditions"``).
            ic: The class (or instance) to introspect.
        """
        self._lines.append(f"{name}:")
        # Collect attributes from the object's __dict__ (and its MRO for classes)
        seen: set[str] = set()
        for klass in (ic.__mro__ if isinstance(ic, type) else [type(ic)]):
            for attr, val in vars(klass).items():
                if attr.startswith("__") or callable(val) or attr in seen:
                    continue
                seen.add(attr)
                self._lines.append(f"  {attr}: {_format_value(val)}")
        # Also capture instance-level attributes
        if not isinstance(ic, type):
            for attr, val in vars(ic).items():
                if attr.startswith("__") or callable(val) or attr in seen:
                    continue
                seen.add(attr)
                self._lines.append(f"  {attr}: {_format_value(val)}")
        self._lines.append("")

    # ------------------------------------------------------------------
    # Boolean Flags
    # ------------------------------------------------------------------

    def log_flags(self, **flags: Any) -> None:
        """Record named boolean (or any scalar) flags.

        Args:
            **flags: Keyword arguments whose names are flag names and
                values are the flag values (typically ``True``/``False``).

        Example::

            logger.log_flags(MSCorrectionFlag=True,
                             GammaCorrectionFlag=False,
                             runRoutine=True)
        """
        self._lines.append("flags:")
        for k, v in sorted(flags.items()):
            self._lines.append(f"  {k}: {_format_value(v)}")
        self._lines.append("")

    # ------------------------------------------------------------------
    # Timestamps
    # ------------------------------------------------------------------

    def log_timestamp(self, label: str) -> None:
        """Record a named wall-clock timestamp.

        Args:
            label: Identifier for the event, e.g. ``"ncp_start"``,
                ``"yspace_end"``.
        """
        self._lines.append(
            f"timestamp_{label}: {datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')}"
        )

    # ------------------------------------------------------------------
    # Intermediate / Final Fit Results
    # ------------------------------------------------------------------

    def log_iteration(self, iteration: int, mean_widths: Any,
                      mean_intensities: Any) -> None:
        """Log per-iteration mean widths and intensities from the MS/GC loop.

        Args:
            iteration: Zero-based iteration index.
            mean_widths: 1-D array of mean widths for each mass.
            mean_intensities: 1-D array of mean intensities for each mass.
        """
        self._lines.append(f"iteration_{iteration}:")
        self._lines.append(f"  mean_widths: {_format_value(mean_widths)}")
        self._lines.append(f"  mean_intensities: {_format_value(mean_intensities)}")
        self._lines.append("")

    def log_final_results(self, results: Any) -> None:
        """Log the final fit parameters for all masses.

        Extracts ``all_mean_widths`` and ``all_mean_intensities`` from the
        results object (standard ``resultsObject`` from
        ``iterativeFitForDataReduction``).  Logs each iteration entry.

        Args:
            results: Results object with ``all_mean_widths`` and
                ``all_mean_intensities`` arrays (shape: ``(n_iterations,
                n_masses)``).
        """
        if results is None:
            self._lines.append("final_results: null")
            self._lines.append("")
            return

        self._lines.append("final_results:")
        try:
            for i, (w, a) in enumerate(
                zip(results.all_mean_widths, results.all_mean_intensities)
            ):
                self._lines.append(f"  iteration_{i}:")
                self._lines.append(f"    mean_widths: {_format_value(w)}")
                self._lines.append(f"    mean_intensities: {_format_value(a)}")
        except AttributeError:
            self._lines.append("  note: result object has unexpected structure")
        self._lines.append("")

    # ------------------------------------------------------------------
    # GoF Metrics
    # ------------------------------------------------------------------

    def log_gof(self, chi2: float, reduced_chi2: float, ndata: int) -> None:
        """Log Goodness-of-Fit metrics.

        Args:
            chi2: Final chi-squared value.
            reduced_chi2: Reduced chi-squared (chi2 / ndof).
            ndata: Number of data points used in the fit.
        """
        self._lines.append("goodness_of_fit:")
        self._lines.append(f"  chi2: {chi2}")
        self._lines.append(f"  reduced_chi2: {reduced_chi2}")
        self._lines.append(f"  ndata: {ndata}")
        self._lines.append("")

    # ------------------------------------------------------------------
    # Sieve-3 Agreement Gate
    # ------------------------------------------------------------------

    def log_agreement_gate(
        self,
        scipy_chi2: float,
        iminuit_chi2: float,
        scipy_pars: np.ndarray,
        iminuit_pars: np.ndarray,
    ) -> None:
        """Record the Sieve-3 numerical agreement gate between Scipy and iMinuit.

        Computes the relative difference for chi-squared and for each
        parameter.  Flags whether the agreement gate is passed or failed
        (threshold: 1 %).

        Args:
            scipy_chi2: Chi-squared from ``scipy.optimize.minimize``.
            iminuit_chi2: Chi-squared from ``iMinuit.migrad()``.
            scipy_pars: Best-fit parameter vector from Scipy.
            iminuit_pars: Best-fit parameter vector from iMinuit.
        """
        scipy_pars = np.asarray(scipy_pars, dtype=float)
        iminuit_pars = np.asarray(iminuit_pars, dtype=float)

        # Chi-squared relative difference
        if scipy_chi2 > 0:
            chi2_rel = abs(scipy_chi2 - iminuit_chi2) / abs(scipy_chi2)
        else:
            chi2_rel = 0.0

        # Parameter-wise relative difference (guarded against zero)
        par_rel = np.where(
            np.abs(scipy_pars) > 1e-12,
            np.abs(scipy_pars - iminuit_pars) / np.abs(scipy_pars),
            0.0,
        )
        max_par_rel = float(np.max(par_rel)) if par_rel.size > 0 else 0.0

        chi2_pass = chi2_rel <= _AGREEMENT_THRESHOLD
        par_pass = max_par_rel <= _AGREEMENT_THRESHOLD
        gate_pass = chi2_pass and par_pass

        self._lines.append("sieve3_agreement_gate:")
        self._lines.append(f"  threshold: {_AGREEMENT_THRESHOLD}")
        self._lines.append(f"  scipy_chi2: {scipy_chi2}")
        self._lines.append(f"  iminuit_chi2: {iminuit_chi2}")
        self._lines.append(f"  chi2_rel_diff: {chi2_rel:.6f}")
        self._lines.append(f"  chi2_gate_passed: {chi2_pass}")
        self._lines.append(f"  scipy_pars: {_format_value(scipy_pars)}")
        self._lines.append(f"  iminuit_pars: {_format_value(iminuit_pars)}")
        self._lines.append(f"  max_par_rel_diff: {max_par_rel:.6f}")
        self._lines.append(f"  par_gate_passed: {par_pass}")
        self._lines.append(f"  overall_gate_passed: {gate_pass}")
        self._lines.append("")

    # ------------------------------------------------------------------
    # Error capture
    # ------------------------------------------------------------------

    def log_error(self, exc: BaseException) -> None:
        """Record a traceback when a fit step fails.

        Args:
            exc: The exception that was raised.
        """
        self._lines.append("error:")
        tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        for line in "".join(tb_lines).splitlines():
            self._lines.append(f"  {line}")
        self._lines.append("")

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def write(self) -> Path:
        """Flush all buffered log lines to the log file.

        Returns:
            Path to the written log file.
        """
        with open(self.logfile, "w", encoding="utf-8") as fh:
            fh.write("\n".join(self._lines))
            fh.write("\n")
        return self.logfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_value(val: Any) -> str:
    """Return a compact, human-readable string for a log entry value.

    ``numpy.ndarray`` objects are rendered as ``[v0, v1, …]`` inline
    lists.  ``Path`` objects are rendered as POSIX strings.  All other
    types use ``repr()``.

    Args:
        val: Value to format.

    Returns:
        Formatted string.
    """
    if isinstance(val, np.ndarray):
        flat = val.flatten()
        if flat.size <= 20:
            items = ", ".join(f"{v}" for v in flat.tolist())
        else:
            items = ", ".join(f"{v}" for v in flat[:20].tolist()) + ", ..."
        return f"[{items}]"
    if isinstance(val, Path):
        return str(val)
    if val is None:
        return "null"
    if isinstance(val, bool):
        return str(val)
    return repr(val)
