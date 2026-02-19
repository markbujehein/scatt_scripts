"""Lightweight Mantid mock for CI testing without the full Mantid framework.

This module installs stub implementations of ``mantid.api`` and
``mantid.simpleapi`` into ``sys.modules`` so that the core fitting and
workspace lifecycle logic in ``vesuvio_analysis/core_functions/`` can be
imported and tested without requiring a full Mantid installation.

Usage — call ``install()`` *before* importing any vesuvio module::

    from tests.mock_mantid import install
    install()

    from vesuvio_analysis.core_functions import procedures  # works without Mantid

The mock covers every Mantid symbol used in the pipeline:

* ``mantid.api``: ``AnalysisDataService``, ``mtd``
* ``mantid.simpleapi``: ``LoadVesuvio``, ``SaveNexus``, ``Load``,
  ``CloneWorkspace``, ``SumSpectra``, ``CreateEmptyTableWorkspace``,
  plus a wildcard catch-all ``__getattr__`` for the star imports.

Workspace objects implement the ``extractX`` / ``extractY`` / ``extractE``
interface expected by the NCP pipeline; the ``TableWorkspace`` stub supports
``addColumn`` / ``addRow`` / ``column`` as used by ``procedures.py``.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# MockWorkspace
# ---------------------------------------------------------------------------

class MockWorkspace:
    """Stand-in for a Mantid ``MatrixWorkspace``.

    Supports:
    * ``name()``
    * ``extractX()``, ``extractY()``, ``extractE()``
    * ``getNumberHistograms()``, ``blocksize()``
    """

    def __init__(self, name: str = "mock_ws", n_spectra: int = 3,
                 n_bins: int = 144):
        self._name = name
        self._n_spectra = n_spectra
        self._n_bins = n_bins
        rng = np.random.default_rng(0)
        self._Y = rng.random((n_spectra, n_bins))
        self._E = np.ones((n_spectra, n_bins)) * 0.01
        self._X = np.tile(
            np.linspace(100.0, 600.0, n_bins), (n_spectra, 1)
        )

    def name(self) -> str:
        return self._name

    def extractX(self) -> np.ndarray:
        return self._X.copy()

    def extractY(self) -> np.ndarray:
        return self._Y.copy()

    def extractE(self) -> np.ndarray:
        return self._E.copy()

    def getNumberHistograms(self) -> int:
        return self._n_spectra

    def blocksize(self) -> int:
        return self._n_bins

    def getSpectrumNumbers(self) -> np.ndarray:
        """Return 1-based spectrum numbers for each histogram."""
        return np.arange(self._n_spectra) + 1

    def __repr__(self) -> str:  # pragma: no cover
        return f"MockWorkspace('{self._name}', {self._n_spectra}×{self._n_bins})"


# ---------------------------------------------------------------------------
# MockTableWorkspace
# ---------------------------------------------------------------------------

class MockTableWorkspace:
    """Stand-in for a Mantid ``TableWorkspace``."""

    def __init__(self, name: str = "mock_table"):
        self._name = name
        self._columns: dict[str, list] = {}
        self._col_types: dict[str, str] = {}

    def name(self) -> str:
        return self._name

    def addColumn(self, col_type: str, col_name: str) -> None:
        self._col_types[col_name] = col_type
        self._columns[col_name] = []

    def addRow(self, row: list | dict) -> None:
        cols = list(self._columns.keys())
        if isinstance(row, dict):
            for k, v in row.items():
                self._columns[k].append(v)
        else:
            for col, val in zip(cols, row):
                self._columns[col].append(val)

    def column(self, col_name: str) -> list:
        return self._columns.get(col_name, [])

    def rowCount(self) -> int:
        if not self._columns:
            return 0
        return len(next(iter(self._columns.values())))

    def columnCount(self) -> int:
        return len(self._columns)


# ---------------------------------------------------------------------------
# MockAnalysisDataService — mirrors Mantid's ``mtd`` / ``AnalysisDataService``
# ---------------------------------------------------------------------------

class MockAnalysisDataService(dict):
    """Dictionary-like stand-in for Mantid's ``AnalysisDataService`` (``mtd``).

    Supports:
    * ``name in ads`` — membership test
    * ``ads[name]`` — workspace retrieval
    * ``ads.add(name, ws)`` — register a workspace
    * ``ads.remove(name)`` — deregister
    * ``ads.clear()`` — remove all entries
    """

    def add(self, name: str, ws: Any) -> None:
        self[name] = ws

    def remove(self, name: str) -> None:
        self.pop(name, None)


# ---------------------------------------------------------------------------
# Singleton ADS instance (analogous to Mantid's global ``mtd``)
# ---------------------------------------------------------------------------

_ads = MockAnalysisDataService()


# ---------------------------------------------------------------------------
# simpleapi function stubs
# ---------------------------------------------------------------------------

def LoadVesuvio(**kwargs: Any) -> MockWorkspace:
    """Return a dummy workspace; no real data is loaded."""
    ws_name = kwargs.get("OutputWorkspace", "LoadVesuvio_out")
    ws = MockWorkspace(name=ws_name)
    _ads.add(ws_name, ws)
    return ws


def SaveNexus(**kwargs: Any) -> None:
    """No-op stub."""
    pass


def Load(**kwargs: Any) -> MockWorkspace:
    """Return a dummy workspace."""
    ws_name = kwargs.get("OutputWorkspace", "Load_out")
    ws = MockWorkspace(name=ws_name)
    _ads.add(ws_name, ws)
    return ws


def CloneWorkspace(**kwargs: Any) -> MockWorkspace:
    """Return a copy of the input workspace (or a fresh stub if absent)."""
    input_ws = kwargs.get("InputWorkspace")
    out_name = kwargs.get("OutputWorkspace", "Clone_out")
    if isinstance(input_ws, str):
        input_ws = _ads.get(input_ws)
    if isinstance(input_ws, MockWorkspace):
        ws = MockWorkspace(name=out_name,
                           n_spectra=input_ws._n_spectra,
                           n_bins=input_ws._n_bins)
    else:
        ws = MockWorkspace(name=out_name)
    _ads.add(out_name, ws)
    return ws


def SumSpectra(**kwargs: Any) -> MockWorkspace:
    """Return a single-spectrum workspace."""
    out_name = kwargs.get("OutputWorkspace", "SumSpectra_out")
    ws = MockWorkspace(name=out_name, n_spectra=1)
    _ads.add(out_name, ws)
    return ws


def CreateEmptyTableWorkspace(**kwargs: Any) -> MockTableWorkspace:
    """Return a dummy table workspace."""
    out_name = kwargs.get("OutputWorkspace", "Table_out")
    tws = MockTableWorkspace(name=out_name)
    _ads.add(out_name, tws)
    return tws


# ---------------------------------------------------------------------------
# Module assembly
# ---------------------------------------------------------------------------

def _make_mantid_api_module() -> types.ModuleType:
    """Build a ``mantid.api`` stub module."""
    mod = types.ModuleType("mantid.api")
    mod.AnalysisDataService = _ads
    mod.mtd = _ads
    return mod


def _make_mantid_simpleapi_module() -> types.ModuleType:
    """Build a ``mantid.simpleapi`` stub module."""
    mod = types.ModuleType("mantid.simpleapi")
    mod.LoadVesuvio = LoadVesuvio
    mod.SaveNexus = SaveNexus
    mod.Load = Load
    mod.CloneWorkspace = CloneWorkspace
    mod.SumSpectra = SumSpectra
    mod.CreateEmptyTableWorkspace = CreateEmptyTableWorkspace

    # Wildcard catch-all: any unrecognised name returns a no-op callable.
    def __getattr__(name: str) -> Callable[..., MockWorkspace]:
        def _noop(*args: Any, **kwargs: Any) -> MockWorkspace:
            ws_name = kwargs.get("OutputWorkspace", f"{name}_out")
            ws = MockWorkspace(name=ws_name)
            _ads.add(ws_name, ws)
            return ws
        return _noop

    mod.__getattr__ = __getattr__  # type: ignore[assignment]
    return mod


def _make_mantid_module() -> types.ModuleType:
    """Build the top-level ``mantid`` stub module."""
    mod = types.ModuleType("mantid")
    mod.__version__ = "0.0.0+mock"
    return mod


# ---------------------------------------------------------------------------
# Public installation entry point
# ---------------------------------------------------------------------------

def install() -> None:
    """Register all ``mantid.*`` stubs in ``sys.modules``.

    Safe to call multiple times — subsequent calls are no-ops if the stubs
    are already installed.
    """
    if "mantid" in sys.modules:
        return  # already installed (real or mock)

    mantid_mod = _make_mantid_module()
    api_mod = _make_mantid_api_module()
    simpleapi_mod = _make_mantid_simpleapi_module()

    sys.modules["mantid"] = mantid_mod
    sys.modules["mantid.api"] = api_mod
    sys.modules["mantid.simpleapi"] = simpleapi_mod

    # Attach sub-modules to the parent so attribute access works too.
    mantid_mod.api = api_mod
    mantid_mod.simpleapi = simpleapi_mod


def uninstall() -> None:
    """Remove the mock stubs from ``sys.modules`` (useful in test teardowns)."""
    for key in ("mantid", "mantid.api", "mantid.simpleapi"):
        sys.modules.pop(key, None)
