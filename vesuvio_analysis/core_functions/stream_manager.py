"""Universal data persistence and multi-domain serialization for VESUVIO.

Provides :class:`StreamManager`, a non-intrusive capture layer that
serializes every data stream produced during NCP fitting and y-space
analysis across all physical domains (TOF, *Q*, *y*) and all levels of
correction.

**Schema**::

    Level 0 (Raw)                   — Full spectrum per detector S(t).
    Level 1 (Correction Components) — MS(t) and Gamma(t) arrays.
    Level 2 (Intermediate Corrected)— (Raw − MS) and (Raw − Gamma).
    Level 3 (Final Physics)         — Corrected signal in t, Q, y domains.
    Metadata                        — InitialConditions, MassIndex, fit results.

Data is stored in compressed ``.npz`` files with hierarchical key
conventions (dot-separated) that encode the level, domain, iteration,
and mass index so that downstream consumers (e.g. a Streamlit dashboard)
can reconstruct plots without any physics calculations.

Architecture Audit alignment (Section 6.4):
  * StreamManager **never** calls Mantid algorithms.
  * It operates only on NumPy arrays extracted from workspaces.
  * Workspace deletion must occur **after** :meth:`save` completes.

Usage::

    sm = StreamManager(output_dir=Path("outputs"), script_name="BaH2",
                       direction="BACKWARD")
    sm.capture("counts", dataY, DataLevel.RAW, domain="tof")
    sm.capture("ms_correction", ms_arr, DataLevel.CORRECTION_COMPONENTS,
               domain="tof", iteration=0)
    sm.set_metadata("masses", np.array([1.008, 12.0, 16.0]))
    path = sm.save()
"""

from __future__ import annotations

import enum
import logging
import weakref
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data-level taxonomy
# ---------------------------------------------------------------------------

class DataLevel(enum.IntEnum):
    """Hierarchical correction level for a captured data stream.

    * ``RAW`` — uncorrected detector counts in TOF.
    * ``CORRECTION_COMPONENTS`` — individual correction profiles (MS, Gamma).
    * ``INTERMEDIATE_CORRECTED`` — partially corrected signal (e.g. Raw − MS).
    * ``FINAL_PHYSICS`` — fully corrected signal in any domain (TOF, *Q*, *y*).
    """

    RAW = 0
    CORRECTION_COMPONENTS = 1
    INTERMEDIATE_CORRECTED = 2
    FINAL_PHYSICS = 3


# Human-readable labels for serialization keys
_LEVEL_LABELS: Dict[DataLevel, str] = {
    DataLevel.RAW: "L0_raw",
    DataLevel.CORRECTION_COMPONENTS: "L1_corrections",
    DataLevel.INTERMEDIATE_CORRECTED: "L2_intermediate",
    DataLevel.FINAL_PHYSICS: "L3_final",
}


# ---------------------------------------------------------------------------
# StreamManager
# ---------------------------------------------------------------------------

class StreamManager:
    """Non-intrusive capture and serialization of analysis data streams.

    All captured arrays are stored in an internal dictionary keyed by a
    hierarchical dot-separated string (e.g.
    ``L0_raw.tof.counts``, ``L1_corrections.tof.iter0.ms``).

    Two capture modes are supported:

    * :meth:`capture` — makes a copy of the array (safe, default).
    * :meth:`capture_weak` — stores a :class:`weakref.ref` to avoid
      doubling memory.  The array is only included in the final
      ``.npz`` if it has not been garbage-collected by save-time.

    Scalar and small metadata values (masses, fit parameters, IC fields)
    are stored separately via :meth:`set_metadata`.

    Args:
        output_dir: Directory for the output ``.npz`` file.
        script_name: Base name of the submission script (e.g. ``"BaH2_500C"``).
        direction: Scattering direction — ``"BACKWARD"``, ``"FORWARD"``,
            ``"JOINT"``, or ``"NONE"``.
    """

    def __init__(
        self,
        output_dir: Path,
        script_name: str,
        direction: str,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.script_name = script_name
        self.direction = direction

        self._streams: Dict[str, np.ndarray] = {}
        self._weak_refs: Dict[str, weakref.ref] = {}
        self._metadata: Dict[str, Any] = {}

    # -- key helpers --------------------------------------------------------

    @staticmethod
    def _build_key(
        level: DataLevel,
        domain: str,
        name: str,
        *,
        iteration: Optional[int] = None,
        mass_index: Optional[int] = None,
    ) -> str:
        """Build a hierarchical dot-separated key.

        Examples::

            >>> StreamManager._build_key(DataLevel.RAW, "tof", "counts")
            'L0_raw.tof.counts'
            >>> StreamManager._build_key(
            ...     DataLevel.CORRECTION_COMPONENTS, "tof", "ms",
            ...     iteration=0, mass_index=1)
            'L1_corrections.tof.iter0.mass1.ms'
        """
        parts: List[str] = [_LEVEL_LABELS[level], domain]
        if iteration is not None:
            parts.append(f"iter{iteration}")
        if mass_index is not None:
            parts.append(f"mass{mass_index}")
        parts.append(name)
        return ".".join(parts)

    # -- capture methods ----------------------------------------------------

    def capture(
        self,
        name: str,
        data: np.ndarray,
        level: DataLevel,
        *,
        domain: str = "tof",
        iteration: Optional[int] = None,
        mass_index: Optional[int] = None,
    ) -> None:
        """Capture a copy of *data* into the internal store.

        A full copy is made so that subsequent in-place modifications to
        the source array do not corrupt the captured snapshot.

        Args:
            name: Short identifier (e.g. ``"counts"``, ``"ms"``, ``"ncp_total"``).
            data: NumPy array to capture.
            level: Correction level (:class:`DataLevel`).
            domain: Physical domain — ``"tof"``, ``"q"``, or ``"y"``.
            iteration: MS/GC iteration index (0-based), if applicable.
            mass_index: Mass index (0-based), if applicable.
        """
        key = self._build_key(
            level, domain, name, iteration=iteration, mass_index=mass_index,
        )
        self._streams[key] = np.array(data, copy=True)

    def capture_weak(
        self,
        name: str,
        data: np.ndarray,
        level: DataLevel,
        *,
        domain: str = "tof",
        iteration: Optional[int] = None,
        mass_index: Optional[int] = None,
    ) -> None:
        """Store a weak reference to *data* (zero-copy, memory-safe).

        The array is included in the ``.npz`` output only if it has not
        been garbage-collected by the time :meth:`save` is called.

        Args:
            name: Short identifier.
            data: NumPy array to reference (must support ``weakref``).
            level: Correction level.
            domain: Physical domain.
            iteration: Iteration index, if applicable.
            mass_index: Mass index, if applicable.
        """
        key = self._build_key(
            level, domain, name, iteration=iteration, mass_index=mass_index,
        )
        self._weak_refs[key] = weakref.ref(data)

    # -- metadata -----------------------------------------------------------

    def set_metadata(self, key: str, value: Any) -> None:
        """Store a scalar or small metadata value.

        Supported types: ``np.ndarray``, ``int``, ``float``, ``str``,
        ``list`` (converted to array), ``bool``.

        Args:
            key: Metadata identifier (e.g. ``"masses"``, ``"n_iterations"``).
            value: Value to store.
        """
        self._metadata[key] = value

    # -- serialization ------------------------------------------------------

    @property
    def save_path(self) -> Path:
        """The resolved output file path.

        Note: the parent directory may not exist until :meth:`save` is
        called, which creates it via ``mkdir(parents=True)``.
        """
        filename = f"{self.script_name}_{self.direction}_streams.npz"
        return self.output_dir / filename

    def save(self) -> Path:
        """Serialize all captured streams and metadata to a compressed ``.npz``.

        Weak references that have expired are silently skipped.

        Returns:
            The :class:`Path` to the written file.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        all_data: Dict[str, np.ndarray] = dict(self._streams)

        # Resolve weak references
        for key, ref in self._weak_refs.items():
            obj = ref()
            if obj is not None:
                all_data[key] = obj
            else:
                logger.debug("StreamManager: weakref expired for key '%s'", key)

        # Encode metadata as arrays (NPZ stores only arrays)
        for mkey, mval in self._metadata.items():
            store_key = f"metadata.{mkey}"
            if isinstance(mval, np.ndarray):
                all_data[store_key] = mval
            elif isinstance(mval, (list, tuple)):
                all_data[store_key] = np.asarray(mval)
            elif isinstance(mval, (int, float, bool)):
                all_data[store_key] = np.array(mval)
            elif isinstance(mval, str):
                all_data[store_key] = np.array(mval)
            else:
                logger.warning(
                    "StreamManager: skipping non-serializable metadata '%s' "
                    "(type %s)", mkey, type(mval).__name__,
                )

        path = self.save_path
        np.savez_compressed(path, **all_data)
        logger.info("StreamManager: saved %d streams to %s", len(all_data), path)
        return path

    # -- loading ------------------------------------------------------------

    @staticmethod
    def load(path: Path) -> Dict[str, np.ndarray]:
        """Load a previously saved stream container.

        Args:
            path: Path to the ``.npz`` file.

        Returns:
            A dictionary mapping hierarchical keys to NumPy arrays.
        """
        return dict(np.load(path, allow_pickle=False))

    # -- introspection ------------------------------------------------------

    def keys(self) -> List[str]:
        """Return all currently registered stream keys (strong + weak)."""
        return sorted(set(self._streams) | set(self._weak_refs))

    @property
    def stream_count(self) -> int:
        """Number of captured streams (strong references only)."""
        return len(self._streams)

    def __repr__(self) -> str:
        return (
            f"StreamManager(script={self.script_name!r}, "
            f"direction={self.direction!r}, "
            f"streams={self.stream_count}, "
            f"weak={len(self._weak_refs)}, "
            f"metadata={len(self._metadata)})"
        )
