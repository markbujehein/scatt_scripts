"""Shared constants for core functions.

This module centralises tunable numerical thresholds and other values
that are referenced by multiple submodules.  Having a single source of
truth prevents accidental divergence when the thresholds change and
simplifies testing.

Current constants:

* ``_AGREEMENT_THRESHOLD`` -- relative difference tolerance used by the
  iMinuit–SciPy agreement check (default 2.5%).  Both
  ``analysis_functions`` and ``log_manager`` import this value, and the
  unit tests refer to it as well.
"""

# Relative tolerance for optimizer cross-validation (2.5 %)
#
# Previously this value appeared as hardcoded literals scattered across
# the codebase (and 1 % in the original specification).  Changing it in a
# single place avoids mismatches.
_AGREEMENT_THRESHOLD: float = 0.025
