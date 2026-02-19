import unittest
import warnings
from pathlib import Path
import re
import importlib
import sys
import tempfile
import types
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from vesuvio_analysis.core_functions.ic_validation import (
    BackwardInitialConditionsModel,
    shadow_validate_backward_initial_conditions,
)


class _BackwardICStub:
    def __init__(self, masses, noOfMSIterations, HToMassIdxRatio=None):
        self.masses = np.array(masses, dtype=float)
        self.noOfMSIterations = noOfMSIterations
        self.HToMassIdxRatio = HToMassIdxRatio


class TestBackwardInitialConditionsModel(unittest.TestCase):
    def test_accepts_valid_backward_conditions(self):
        model = BackwardInitialConditionsModel(
            masses=np.array([1.0079, 16.0]),
            noOfMSIterations=0,
            HToMassIdxRatio=12.0,
        )
        self.assertEqual(model.noOfMSIterations, 0)

    def test_rejects_non_positive_masses(self):
        with self.assertRaises(ValidationError):
            BackwardInitialConditionsModel(
                masses=np.array([0.0, 16.0]),
                noOfMSIterations=0,
                HToMassIdxRatio=None,
            )

    def test_rejects_negative_ms_iterations(self):
        with self.assertRaises(ValidationError):
            BackwardInitialConditionsModel(
                masses=np.array([16.0, 27.0]),
                noOfMSIterations=-1,
                HToMassIdxRatio=None,
            )

    def test_rejects_h_ratio_without_hydrogen(self):
        with self.assertRaises(ValidationError):
            BackwardInitialConditionsModel(
                masses=np.array([16.0, 27.0]),
                noOfMSIterations=0,
                HToMassIdxRatio=2.0,
            )


class TestShadowValidation(unittest.TestCase):
    def test_shadow_validation_warns_instead_of_raising(self):
        ic = _BackwardICStub(masses=[16.0, 27.0], noOfMSIterations=1, HToMassIdxRatio=2.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            shadow_validate_backward_initial_conditions(ic)
        self.assertGreaterEqual(len(caught), 1)
        self.assertIn("Pydantic shadow validation", str(caught[0].message))

    def test_ichelpers_applies_shadow_validation_only_in_backward_mode(self):
        ichelpers_path = (
            Path(__file__).resolve().parent.parent
            / "vesuvio_analysis"
            / "core_functions"
            / "ICHelpers.py"
        )
        content = ichelpers_path.read_text(encoding="utf-8")
        pattern = (
            r'if IC\.modeRunning == "BACKWARD":\n'
            r"\s+shadow_validate_backward_initial_conditions\(IC\)"
        )
        self.assertRegex(content, re.compile(pattern))


class TestICHelpersIntegration(unittest.TestCase):
    def _import_ichelpers_with_mantid_stub(self):
        simpleapi = types.ModuleType("mantid.simpleapi")
        simpleapi.LoadVesuvio = lambda **kwargs: None
        simpleapi.SaveNexus = lambda **kwargs: None
        mantid = types.ModuleType("mantid")
        mantid.simpleapi = simpleapi
        with patch.dict(
            sys.modules,
            {"mantid": mantid, "mantid.simpleapi": simpleapi},
        ):
            module = importlib.import_module("vesuvio_analysis.core_functions.ICHelpers")
            return importlib.reload(module)

    def test_complete_ic_calls_shadow_validation_for_backward(self):
        ichelpers = self._import_ichelpers_with_mantid_stub()

        class _BackwardIC:
            firstSpec = 3
            lastSpec = 4
            masses = np.array([16.0, 27.0])
            noOfMSIterations = 0
            HToMassIdxRatio = None
            maskedSpecAllNo = np.array([], dtype=int)

        class _WSIC:
            mode = "DoubleDifference"
            ipfile = "ip.par"

        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.nxs"
            empty = Path(tmp) / "empty.nxs"
            raw.write_text("raw", encoding="utf-8")
            empty.write_text("empty", encoding="utf-8")

            with (
                patch.object(ichelpers, "inputDirsForSample", return_value=(raw, empty)),
                patch.object(ichelpers, "setOutputDirsForSample", return_value=None),
                patch.object(ichelpers, "experimentsPath", Path(tmp) / "experiments"),
                patch.object(
                    ichelpers,
                    "shadow_validate_backward_initial_conditions",
                ) as mock_shadow,
            ):
                (Path(tmp) / "experiments" / "thymol_10K_Gauss1D").mkdir(
                    parents=True, exist_ok=True
                )
                ichelpers.completeICFromInputs(
                    _BackwardIC, "thymol_10K_Gauss1D", _WSIC
                )
                mock_shadow.assert_called_once_with(_BackwardIC)

    def test_complete_ic_emits_shadow_warning_for_invalid_backward_ic(self):
        ichelpers = self._import_ichelpers_with_mantid_stub()

        class _BackwardICInvalid:
            firstSpec = 3
            lastSpec = 4
            masses = np.array([16.0, 27.0])
            noOfMSIterations = 0
            HToMassIdxRatio = 2.0
            maskedSpecAllNo = np.array([], dtype=int)

        class _WSIC:
            mode = "DoubleDifference"
            ipfile = "ip.par"

        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.nxs"
            empty = Path(tmp) / "empty.nxs"
            raw.write_text("raw", encoding="utf-8")
            empty.write_text("empty", encoding="utf-8")
            with (
                patch.object(ichelpers, "inputDirsForSample", return_value=(raw, empty)),
                patch.object(ichelpers, "setOutputDirsForSample", return_value=None),
                patch.object(ichelpers, "experimentsPath", Path(tmp) / "experiments"),
                warnings.catch_warnings(record=True) as caught,
            ):
                warnings.simplefilter("always")
                (Path(tmp) / "experiments" / "thymol_10K_Gauss1D").mkdir(
                    parents=True, exist_ok=True
                )
                ichelpers.completeICFromInputs(
                    _BackwardICInvalid, "thymol_10K_Gauss1D", _WSIC
                )
                self.assertGreaterEqual(len(caught), 1)
                self.assertIn("Pydantic shadow validation", str(caught[0].message))
