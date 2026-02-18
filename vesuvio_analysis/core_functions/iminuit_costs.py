"""Class-based cost function for iMinuit NCP fitting.

Provides ``NCPCostFunction``, a callable class compatible with
``iminuit.Minuit`` that wraps the same chi-squared calculation used by
``scipy.optimize.minimize`` in ``fitNcpToSingleSpec``.

The class uses the modern iMinuit v2.x ``_parameters`` dict to
dynamically define parameter names and bounds.  Each parameter triplet
*(I_m, W_m, C_m)* corresponds to the intensity, width, and centre of
mass *m* in the NCP model.

Usage
-----
``_parameters`` is a ``dict[str, tuple[float, float] | None]`` mapping
parameter names to ``(lower, upper)`` limit tuples.  ``None`` means the
parameter is unbounded.  When present on the cost object, ``Minuit``
reads it to discover the parameter signature **and** automatically
applies the corresponding limits — no separate ``m.limits[…] = …``
call is needed.

Example::

    cost = NCPCostFunction(dataY, dataE, ySpaces, resPars, instrPars,
                           kinArrays, ic)
    m = Minuit(cost, *ic.initPars)
    m.migrad()
    m.hesse()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from iminuit import Minuit

logger = logging.getLogger(__name__)


class NCPCostFunction:
    """Chi-squared cost function for the NCP model, compatible with iMinuit.

    Attributes:
        errordef: Tells ``Minuit`` the statistical meaning of the cost
            value.  ``Minuit.LEAST_SQUARES`` (= 1.0) is used because the
            cost is a chi-squared.
    """

    errordef: float = Minuit.LEAST_SQUARES

    def __init__(
        self,
        dataY: np.ndarray,
        dataE: np.ndarray,
        ySpacesForEachMass: np.ndarray,
        resolutionPars: np.ndarray,
        instrPars: np.ndarray,
        kinematicArrays: np.ndarray,
        ic: Any,
    ) -> None:
        """Initialise with fixed experimental data and IC metadata.

        Args:
            dataY: Observed counts for one spectrum, shape ``(n_bins,)``.
            dataE: Errors for one spectrum, shape ``(n_bins,)``.
            ySpacesForEachMass: y-spaces, shape ``(n_masses, n_bins)``.
            resolutionPars: Resolution parameters, shape ``(6,)``.
            instrPars: Instrument parameters, shape ``(6,)``.
            kinematicArrays: ``[v0, E0, deltaE, deltaQ]``, shape
                ``(4, n_bins)``.
            ic: Completed initial-conditions object with ``masses``,
                ``bounds``, and ``normVoigt``.
        """
        self._dataY = dataY
        self._dataE = dataE
        self._ySpacesForEachMass = ySpacesForEachMass
        self._resolutionPars = resolutionPars
        self._instrPars = instrPars
        self._kinematicArrays = kinematicArrays
        self._ic = ic

        # Build the _parameters dict for Minuit signature discovery.
        self._parameters = _build_parameters_dict(ic)

    # ------------------------------------------------------------------
    def __call__(self, *args: float) -> float:
        """Evaluate the chi-squared cost.

        Args:
            *args: Current fit parameters in the order
                ``[I0, W0, C0, I1, W1, C1, …]``.

        Returns:
            Scalar chi-squared value.
        """
        # Import here to avoid circular imports at module level.
        from .analysis_functions import calculateNcpSpec

        pars = np.asarray(args, dtype=np.float64)

        _, ncpTotal = calculateNcpSpec(
            self._ic,
            pars,
            self._ySpacesForEachMass,
            self._resolutionPars,
            self._instrPars,
            self._kinematicArrays,
        )

        # Mask zeros (same logic as errorFunction).
        zerosMask = self._dataY == 0
        ncpFilt = ncpTotal[~zerosMask]
        dataYFilt = self._dataY[~zerosMask]
        dataEFilt = self._dataE[~zerosMask]

        if np.all(self._dataE == 0):
            return float(np.sum((ncpFilt - dataYFilt) ** 2))

        return float(np.sum((ncpFilt - dataYFilt) ** 2 / dataEFilt ** 2))


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _build_parameters_dict(ic: Any) -> Dict[str, Optional[Tuple[float, float]]]:
    """Construct the ``_parameters`` dict from the IC bounds array.

    Parameters are named ``I0, W0, C0, I1, W1, C1, …`` following the
    interleaved convention of ``ic.initPars``.  Bounds of ``np.nan`` in
    ``ic.bounds`` are translated to ``None`` (unbounded) for Minuit.

    Args:
        ic: Completed initial-conditions object with ``masses`` and
            ``bounds``.

    Returns:
        Ordered dict of ``{name: (lo, hi) | None}``.
    """
    n_masses = len(ic.masses)
    labels = ("I", "W", "C")
    params: Dict[str, Optional[Tuple[float, float]]] = {}
    for m_idx in range(n_masses):
        for p_idx, lbl in enumerate(labels):
            flat_idx = 3 * m_idx + p_idx
            name = f"{lbl}{m_idx}"
            lo, hi = ic.bounds[flat_idx]
            lo_val = None if np.isnan(lo) else float(lo)
            hi_val = None if np.isnan(hi) else float(hi)
            if lo_val is None and hi_val is None:
                params[name] = None
            else:
                params[name] = (
                    lo_val if lo_val is not None else -np.inf,
                    hi_val if hi_val is not None else np.inf,
                )
    return params
