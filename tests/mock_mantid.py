"""Lightweight Mantid mock for use in the test suite.

This module patches ``sys.modules`` so that test files can import
``vesuvio_analysis`` modules that contain top-level ``import mantid``
statements without requiring a real Mantid installation.

Usage
-----
Call ``mock_mantid.install()`` **before** importing any
``vesuvio_analysis`` module that depends on Mantid::

    import tests.mock_mantid as mock_mantid
    mock_mantid.install()

    from vesuvio_analysis.core_functions import some_module  # safe now

The mock covers the minimal Mantid surface used by the analysis pipeline:

- ``mantid.simpleapi``           — ``LoadVesuvio``, ``SaveNexus``,
  ``Rebin``, ``Integration``, ``SumSpectra`` as no-op ``MagicMock``\\s.
- ``mantid.api``                 — ``AnalysisDataService``, ``mtd``
  (backed by ``_MockADS``), ``WorkspaceGroup``.
- ``mantid.kernel``              — ``logger``, ``V3D``.
- ``mantid.plots``               — silenced.
- ``mantid``                     — top-level ``__version__`` stub.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Mock AnalysisDataService
# ---------------------------------------------------------------------------

class _MockADS(dict):
    """Dictionary-backed stand-in for Mantid's ``AnalysisDataService``.

    Supports the subset of the ADS interface exercised by the pipeline:
    ``__contains__``, ``__getitem__``, ``__setitem__``, ``__delitem__``,
    ``clear()``, and ``addOrReplace()``.
    """

    def addOrReplace(self, name: str, workspace: Any) -> None:
        self[name] = workspace

    def remove(self, name: str) -> None:
        self.pop(name, None)

    def doesExist(self, name: str) -> bool:
        return name in self

    def getObjectNames(self) -> list[str]:
        return list(self.keys())


_ads_instance = _MockADS()


# ---------------------------------------------------------------------------
# Public install() entry point
# ---------------------------------------------------------------------------

def install() -> None:
    """Patch ``sys.modules`` with lightweight Mantid stubs.

    Idempotent — calling ``install()`` multiple times is safe.
    """
    if "mantid" in sys.modules:
        return  # already installed (real or mock)

    # --- mantid (top-level) -------------------------------------------------
    mantid_mod = types.ModuleType("mantid")
    mantid_mod.__version__ = "mock-6.0.0"  # type: ignore[attr-defined]
    sys.modules["mantid"] = mantid_mod

    # --- mantid.kernel -------------------------------------------------------
    kernel_mod = types.ModuleType("mantid.kernel")
    kernel_mod.logger = MagicMock()  # type: ignore[attr-defined]
    kernel_mod.V3D = MagicMock()  # type: ignore[attr-defined]
    sys.modules["mantid.kernel"] = kernel_mod
    mantid_mod.kernel = kernel_mod  # type: ignore[attr-defined]

    # --- mantid.api ----------------------------------------------------------
    api_mod = types.ModuleType("mantid.api")
    api_mod.AnalysisDataService = _ads_instance  # type: ignore[attr-defined]
    api_mod.mtd = _ads_instance  # type: ignore[attr-defined]
    api_mod.WorkspaceGroup = MagicMock()  # type: ignore[attr-defined]
    api_mod.MatrixWorkspace = MagicMock()  # type: ignore[attr-defined]
    sys.modules["mantid.api"] = api_mod
    mantid_mod.api = api_mod  # type: ignore[attr-defined]

    # Expose mtd at the top-level ``mantid`` namespace (common usage pattern)
    mantid_mod.mtd = _ads_instance  # type: ignore[attr-defined]

    # --- mantid.simpleapi ----------------------------------------------------
    simpleapi_mod = types.ModuleType("mantid.simpleapi")
    for _fn in (
        "LoadVesuvio",
        "LoadNexus",
        "SaveNexus",
        "Rebin",
        "Integration",
        "SumSpectra",
        "Scale",
        "Minus",
        "Plus",
        "Divide",
        "Multiply",
        "CloneWorkspace",
        "DeleteWorkspace",
        "GroupWorkspaces",
        "UnGroupWorkspace",
        "ConvertUnits",
        "CropWorkspace",
        "ExtractSingleSpectrum",
        "AppendSpectra",
        "RenameWorkspace",
        "mtd",
    ):
        setattr(simpleapi_mod, _fn, MagicMock(name=_fn))
    simpleapi_mod.mtd = _ads_instance  # type: ignore[attr-defined]
    sys.modules["mantid.simpleapi"] = simpleapi_mod
    mantid_mod.simpleapi = simpleapi_mod  # type: ignore[attr-defined]

    # --- mantid.plots --------------------------------------------------------
    plots_mod = types.ModuleType("mantid.plots")
    sys.modules["mantid.plots"] = plots_mod
    mantid_mod.plots = plots_mod  # type: ignore[attr-defined]

    # --- mantid.dataobjects --------------------------------------------------
    dataobjects_mod = types.ModuleType("mantid.dataobjects")
    dataobjects_mod.Workspace2D = MagicMock()  # type: ignore[attr-defined]
    sys.modules["mantid.dataobjects"] = dataobjects_mod
    mantid_mod.dataobjects = dataobjects_mod  # type: ignore[attr-defined]
