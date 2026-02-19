"""Tests for the StreamManager data persistence system.

These tests do **not** depend on Mantid and can be run in any standard
Python environment with NumPy installed::

    python -m pytest tests/test_stream_manager.py -v

Each test verifies a distinct aspect of the StreamManager class:

1. Capture and serialization of arrays at different data levels.
2. Hierarchical key construction (dot-separated, level/domain/iter/mass).
3. Weak-reference capture (zero-copy, memory-safe).
4. Metadata storage and retrieval.
5. Round-trip save/load fidelity.
6. Correct handling of expired weak references.
7. Introspection (keys, stream_count, repr).
8. Output directory creation.
"""

import gc
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vesuvio_analysis.core_functions.stream_manager import (
    DataLevel,
    StreamManager,
)


class TestDataLevel(unittest.TestCase):
    """DataLevel enum semantics."""

    def test_level_values(self):
        self.assertEqual(DataLevel.RAW, 0)
        self.assertEqual(DataLevel.CORRECTION_COMPONENTS, 1)
        self.assertEqual(DataLevel.INTERMEDIATE_CORRECTED, 2)
        self.assertEqual(DataLevel.FINAL_PHYSICS, 3)

    def test_level_ordering(self):
        self.assertLess(DataLevel.RAW, DataLevel.CORRECTION_COMPONENTS)
        self.assertLess(DataLevel.CORRECTION_COMPONENTS, DataLevel.INTERMEDIATE_CORRECTED)
        self.assertLess(DataLevel.INTERMEDIATE_CORRECTED, DataLevel.FINAL_PHYSICS)


class TestKeyConstruction(unittest.TestCase):
    """StreamManager._build_key produces correct hierarchical keys."""

    def test_minimal_key(self):
        key = StreamManager._build_key(DataLevel.RAW, "tof", "counts")
        self.assertEqual(key, "L0_raw.tof.counts")

    def test_key_with_iteration(self):
        key = StreamManager._build_key(
            DataLevel.CORRECTION_COMPONENTS, "tof", "ms", iteration=2,
        )
        self.assertEqual(key, "L1_corrections.tof.iter2.ms")

    def test_key_with_mass_index(self):
        key = StreamManager._build_key(
            DataLevel.FINAL_PHYSICS, "y", "ncp", mass_index=0,
        )
        self.assertEqual(key, "L3_final.y.mass0.ncp")

    def test_key_with_iteration_and_mass(self):
        key = StreamManager._build_key(
            DataLevel.CORRECTION_COMPONENTS, "tof", "gamma",
            iteration=1, mass_index=3,
        )
        self.assertEqual(key, "L1_corrections.tof.iter1.mass3.gamma")


class TestCapture(unittest.TestCase):
    """StreamManager.capture stores data correctly."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.sm = StreamManager(
            output_dir=Path(self.tmp),
            script_name="test_script",
            direction="BACKWARD",
        )

    def test_capture_stores_copy(self):
        """Modifying the source array must not affect captured data."""
        original = np.array([1.0, 2.0, 3.0])
        self.sm.capture("counts", original, DataLevel.RAW)
        original[0] = 999.0
        key = "L0_raw.tof.counts"
        np.testing.assert_array_equal(self.sm._streams[key], [1.0, 2.0, 3.0])

    def test_capture_2d_array(self):
        data = np.random.rand(5, 100)
        self.sm.capture("dataY", data, DataLevel.RAW, domain="tof")
        key = "L0_raw.tof.dataY"
        self.assertIn(key, self.sm._streams)
        self.assertEqual(self.sm._streams[key].shape, (5, 100))

    def test_capture_with_iteration(self):
        data = np.ones((3, 50))
        self.sm.capture("ms", data, DataLevel.CORRECTION_COMPONENTS,
                        domain="tof", iteration=0)
        key = "L1_corrections.tof.iter0.ms"
        self.assertIn(key, self.sm._streams)

    def test_capture_with_mass_index(self):
        data = np.ones(50)
        self.sm.capture("ncp", data, DataLevel.FINAL_PHYSICS,
                        domain="y", mass_index=1)
        key = "L3_final.y.mass1.ncp"
        self.assertIn(key, self.sm._streams)

    def test_stream_count(self):
        self.assertEqual(self.sm.stream_count, 0)
        self.sm.capture("a", np.array([1]), DataLevel.RAW)
        self.sm.capture("b", np.array([2]), DataLevel.RAW)
        self.assertEqual(self.sm.stream_count, 2)


class TestCaptureWeak(unittest.TestCase):
    """StreamManager.capture_weak stores weak references."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.sm = StreamManager(
            output_dir=Path(self.tmp),
            script_name="test_script",
            direction="FORWARD",
        )

    def test_weak_reference_stored(self):
        data = np.array([10.0, 20.0, 30.0])
        self.sm.capture_weak("resolution", data, DataLevel.FINAL_PHYSICS,
                             domain="y")
        key = "L3_final.y.resolution"
        self.assertIn(key, self.sm._weak_refs)

    def test_weak_ref_resolved_on_save(self):
        """Weak reference should be included in saved file while alive."""
        data = np.array([1.0, 2.0, 3.0])
        self.sm.capture_weak("res", data, DataLevel.FINAL_PHYSICS, domain="y")
        path = self.sm.save()
        loaded = StreamManager.load(path)
        np.testing.assert_array_equal(loaded["L3_final.y.res"], data)

    def test_expired_weak_ref_skipped(self):
        """Expired weak references are silently skipped during save."""
        data = np.array([1.0, 2.0, 3.0])
        self.sm.capture_weak("temp", data, DataLevel.RAW, domain="tof")
        # Delete the array to expire the weakref
        del data
        gc.collect()
        path = self.sm.save()
        loaded = StreamManager.load(path)
        self.assertNotIn("L0_raw.tof.temp", loaded)


class TestMetadata(unittest.TestCase):
    """StreamManager.set_metadata stores and serializes metadata."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.sm = StreamManager(
            output_dir=Path(self.tmp),
            script_name="test_meta",
            direction="BACKWARD",
        )

    def test_metadata_array(self):
        masses = np.array([1.008, 12.0, 16.0])
        self.sm.set_metadata("masses", masses)
        path = self.sm.save()
        loaded = StreamManager.load(path)
        np.testing.assert_array_almost_equal(
            loaded["metadata.masses"], masses,
        )

    def test_metadata_scalar(self):
        self.sm.set_metadata("n_iterations", 4)
        path = self.sm.save()
        loaded = StreamManager.load(path)
        self.assertEqual(int(loaded["metadata.n_iterations"]), 4)

    def test_metadata_string(self):
        self.sm.set_metadata("fit_model", "ANSIO_GAUSSIAN")
        path = self.sm.save()
        loaded = StreamManager.load(path)
        self.assertEqual(str(loaded["metadata.fit_model"]), "ANSIO_GAUSSIAN")

    def test_metadata_list(self):
        self.sm.set_metadata("widths", [1.0, 2.0, 3.0])
        path = self.sm.save()
        loaded = StreamManager.load(path)
        np.testing.assert_array_almost_equal(
            loaded["metadata.widths"], [1.0, 2.0, 3.0],
        )


class TestRoundTrip(unittest.TestCase):
    """Full save/load round-trip preserves data fidelity."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.sm = StreamManager(
            output_dir=Path(self.tmp),
            script_name="BaH2_500C",
            direction="BACKWARD",
        )

    def test_full_round_trip(self):
        """Multi-level capture with metadata survives round-trip."""
        raw_Y = np.random.rand(10, 200)
        raw_E = np.random.rand(10, 200)
        ms_arr = np.random.rand(10, 200)
        corrected = np.random.rand(10, 200)
        joy = np.random.rand(50)

        self.sm.capture("dataY", raw_Y, DataLevel.RAW, domain="tof")
        self.sm.capture("dataE", raw_E, DataLevel.RAW, domain="tof")
        self.sm.capture("ms", ms_arr, DataLevel.CORRECTION_COMPONENTS,
                        domain="tof", iteration=0)
        self.sm.capture("corrected", corrected,
                        DataLevel.INTERMEDIATE_CORRECTED,
                        domain="tof", iteration=0)
        self.sm.capture("joy_avg", joy, DataLevel.FINAL_PHYSICS, domain="y")
        self.sm.set_metadata("masses", np.array([1.008, 12.0, 16.0]))
        self.sm.set_metadata("n_iterations", 3)

        path = self.sm.save()
        self.assertTrue(path.exists())
        self.assertTrue(str(path).endswith(".npz"))

        loaded = StreamManager.load(path)

        np.testing.assert_array_equal(loaded["L0_raw.tof.dataY"], raw_Y)
        np.testing.assert_array_equal(loaded["L0_raw.tof.dataE"], raw_E)
        np.testing.assert_array_equal(
            loaded["L1_corrections.tof.iter0.ms"], ms_arr,
        )
        np.testing.assert_array_equal(
            loaded["L2_intermediate.tof.iter0.corrected"], corrected,
        )
        np.testing.assert_array_equal(
            loaded["L3_final.y.joy_avg"], joy,
        )
        np.testing.assert_array_almost_equal(
            loaded["metadata.masses"], [1.008, 12.0, 16.0],
        )

    def test_save_creates_directory(self):
        """Output directory is created if it does not exist."""
        nested = Path(self.tmp) / "sub" / "dir"
        sm = StreamManager(output_dir=nested, script_name="x", direction="NONE")
        sm.capture("a", np.array([1]), DataLevel.RAW)
        path = sm.save()
        self.assertTrue(nested.exists())
        self.assertTrue(path.exists())

    def test_save_path_property(self):
        expected = Path(self.tmp) / "BaH2_500C_BACKWARD_streams.npz"
        self.assertEqual(self.sm.save_path, expected)


class TestIntrospection(unittest.TestCase):
    """StreamManager introspection methods."""

    def setUp(self):
        self.sm = StreamManager(
            output_dir=Path("/tmp/test_sm"),
            script_name="test",
            direction="FORWARD",
        )

    def test_keys_empty(self):
        self.assertEqual(self.sm.keys(), [])

    def test_keys_combined(self):
        self.sm.capture("a", np.array([1]), DataLevel.RAW)
        data = np.array([2.0])
        self.sm.capture_weak("b", data, DataLevel.RAW)
        keys = self.sm.keys()
        self.assertEqual(len(keys), 2)
        self.assertIn("L0_raw.tof.a", keys)
        self.assertIn("L0_raw.tof.b", keys)

    def test_repr(self):
        r = repr(self.sm)
        self.assertIn("test", r)
        self.assertIn("FORWARD", r)
        self.assertIn("streams=0", r)


class TestEmptySave(unittest.TestCase):
    """Edge case: saving with no captured data."""

    def test_empty_save_creates_file(self):
        tmp = tempfile.mkdtemp()
        sm = StreamManager(
            output_dir=Path(tmp), script_name="empty", direction="NONE",
        )
        path = sm.save()
        self.assertTrue(path.exists())
        loaded = StreamManager.load(path)
        self.assertEqual(len(loaded), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
