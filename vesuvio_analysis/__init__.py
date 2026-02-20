"""vesuvio_analysis — DINS data reduction and analysis package.

Primary entry points are exposed at the top level so users can write::

    from vesuvio_analysis import runScript
    from vesuvio_analysis import iterativeFitForDataReduction
    from vesuvio_analysis import fitInYSpaceProcedure
    from vesuvio_analysis import completeICFromInputs

All symbols are loaded *lazily* so that importing a Mantid-free sub-module
(e.g. ``vesuvio_analysis.core_functions.numba_routines``) does not
inadvertently trigger the Mantid import chain at collection time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # only for type-checkers; never executed at runtime
    from vesuvio_analysis.core_functions.run_script import runScript
    from vesuvio_analysis.core_functions.analysis_functions import (
        iterativeFitForDataReduction,
    )
    from vesuvio_analysis.core_functions.fit_in_yspace import fitInYSpaceProcedure
    from vesuvio_analysis.core_functions.ICHelpers import (
        completeICFromInputs,
        completeBootIC,
        completeYFitIC,
        buildFinalWSName,
    )
    from vesuvio_analysis.core_functions.procedures import (
        runIndependentIterativeProcedure,
        runJointBackAndForwardProcedure,
    )

__all__ = [
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

# Module-level __getattr__ provides lazy loading: the symbol is imported only
# when first accessed, which means Mantid is *not* required unless the caller
# actually uses a Mantid-dependent entry point.
_LAZY_MAP: dict[str, tuple[str, str]] = {
    "runScript": (
        "vesuvio_analysis.core_functions.run_script",
        "runScript",
    ),
    "iterativeFitForDataReduction": (
        "vesuvio_analysis.core_functions.analysis_functions",
        "iterativeFitForDataReduction",
    ),
    "fitInYSpaceProcedure": (
        "vesuvio_analysis.core_functions.fit_in_yspace",
        "fitInYSpaceProcedure",
    ),
    "completeICFromInputs": (
        "vesuvio_analysis.core_functions.ICHelpers",
        "completeICFromInputs",
    ),
    "completeBootIC": (
        "vesuvio_analysis.core_functions.ICHelpers",
        "completeBootIC",
    ),
    "completeYFitIC": (
        "vesuvio_analysis.core_functions.ICHelpers",
        "completeYFitIC",
    ),
    "buildFinalWSName": (
        "vesuvio_analysis.core_functions.ICHelpers",
        "buildFinalWSName",
    ),
    "runIndependentIterativeProcedure": (
        "vesuvio_analysis.core_functions.procedures",
        "runIndependentIterativeProcedure",
    ),
    "runJointBackAndForwardProcedure": (
        "vesuvio_analysis.core_functions.procedures",
        "runJointBackAndForwardProcedure",
    ),
}


def __getattr__(name: str):  # noqa: ANN001, ANN202
    if name in _LAZY_MAP:
        import importlib

        module_path, attr = _LAZY_MAP[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module 'vesuvio_analysis' has no attribute {name!r}")
