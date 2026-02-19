"""Shadow-validation tests for Pydantic-based InitialConditions checks.

These tests run without Mantid by stubbing ``mantid.simpleapi`` where needed::

    python -m unittest tests.test_ic_validation_shadow -v

Coverage:
1. ``BackwardInitialConditionsModel`` field and cross-field validation.
2. ``ForwardInitialConditionsModel`` field validation.
3. ``YSpaceFitInitialConditionsModel`` field validation.
4. ``BootstrapInitialConditionsModel`` field validation.
5. Warning-only shadow behavior (non-breaking mode) for all IC classes.
6. Integration with ``ICHelpers.completeICFromInputs``:
   backward path calls backward validator; forward path calls forward validator.
7. Integration with ``ICHelpers.completeYFitIC`` and ``completeBootIC``.
"""

import unittest
import warnings
from pathlib import Path
import importlib
import sys
import tempfile
import types
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from vesuvio_analysis.core_functions.ic_validation import (
    BackwardInitialConditionsModel,
    BootstrapInitialConditionsModel,
    ForwardInitialConditionsModel,
    YSpaceFitInitialConditionsModel,
    shadow_validate_backward_initial_conditions,
    shadow_validate_bootstrap_initial_conditions,
    shadow_validate_forward_initial_conditions,
    shadow_validate_yspace_fit_initial_conditions,
)


class _BackwardICStub:
    def __init__(self, masses, noOfMSIterations, HToMassIdxRatio=None):
        self.masses = np.array(masses, dtype=float)
        self.noOfMSIterations = noOfMSIterations
        self.HToMassIdxRatio = HToMassIdxRatio


class _ForwardICStub:
    def __init__(self, masses, noOfMSIterations):
        self.masses = np.array(masses, dtype=float)
        self.noOfMSIterations = noOfMSIterations


class _YFitICStub:
    def __init__(self, fitModel="SINGLE_GAUSSIAN", nGlobalFitGroups=4, maskTypeProcedure="NAN"):
        self.fitModel = fitModel
        self.nGlobalFitGroups = nGlobalFitGroups
        self.maskTypeProcedure = maskTypeProcedure


class _BootICStub:
    def __init__(self, procedure="BACKWARD", fitInYSpace=None,
                 bootstrapType="BOOT_GAUSS_ERRS", nSamples=10):
        self.procedure = procedure
        self.fitInYSpace = fitInYSpace
        self.bootstrapType = bootstrapType
        self.nSamples = nSamples


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

    def test_rejects_empty_masses(self):
        with self.assertRaises(ValidationError):
            BackwardInitialConditionsModel(
                masses=np.array([]),
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
    def test_backward_shadow_warns_instead_of_raising(self):
        ic = _BackwardICStub(masses=[16.0, 27.0], noOfMSIterations=1, HToMassIdxRatio=2.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            shadow_validate_backward_initial_conditions(ic)
        self.assertGreaterEqual(len(caught), 1)
        self.assertIn("Pydantic shadow validation", str(caught[0].message))

    def test_forward_shadow_warns_instead_of_raising(self):
        ic = _ForwardICStub(masses=[16.0, 27.0], noOfMSIterations=-1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            shadow_validate_forward_initial_conditions(ic)
        self.assertGreaterEqual(len(caught), 1)
        self.assertIn("Pydantic shadow validation", str(caught[0].message))

    def test_yfit_shadow_warns_instead_of_raising(self):
        ic = _YFitICStub(fitModel="INVALID_MODEL")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            shadow_validate_yspace_fit_initial_conditions(ic)
        self.assertGreaterEqual(len(caught), 1)
        self.assertIn("Pydantic shadow validation", str(caught[0].message))

    def test_bootstrap_shadow_warns_instead_of_raising(self):
        ic = _BootICStub(nSamples=-1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            shadow_validate_bootstrap_initial_conditions(ic)
        self.assertGreaterEqual(len(caught), 1)
        self.assertIn("Pydantic shadow validation", str(caught[0].message))


class TestForwardInitialConditionsModel(unittest.TestCase):
    def test_accepts_valid_forward_conditions(self):
        model = ForwardInitialConditionsModel(
            masses=np.array([1.0079, 16.0]),
            noOfMSIterations=0,
        )
        self.assertEqual(model.noOfMSIterations, 0)

    def test_rejects_non_positive_masses(self):
        with self.assertRaises(ValidationError):
            ForwardInitialConditionsModel(
                masses=np.array([-1.0, 16.0]),
                noOfMSIterations=0,
            )

    def test_rejects_empty_masses(self):
        with self.assertRaises(ValidationError):
            ForwardInitialConditionsModel(masses=np.array([]), noOfMSIterations=0)

    def test_rejects_negative_ms_iterations(self):
        with self.assertRaises(ValidationError):
            ForwardInitialConditionsModel(
                masses=np.array([1.0079, 16.0]),
                noOfMSIterations=-2,
            )


class TestYSpaceFitInitialConditionsModel(unittest.TestCase):
    def test_accepts_valid_yfit_conditions(self):
        model = YSpaceFitInitialConditionsModel(
            fitModel="SINGLE_GAUSSIAN",
            nGlobalFitGroups=4,
            maskTypeProcedure="NAN",
        )
        self.assertEqual(model.fitModel, "SINGLE_GAUSSIAN")

    def test_accepts_all_as_n_global_fit_groups(self):
        model = YSpaceFitInitialConditionsModel(
            fitModel="GC_C4",
            nGlobalFitGroups="ALL",
            maskTypeProcedure=None,
        )
        self.assertEqual(model.nGlobalFitGroups, "ALL")

    def test_rejects_invalid_fit_model(self):
        with self.assertRaises(ValidationError):
            YSpaceFitInitialConditionsModel(
                fitModel="NOT_A_MODEL",
                nGlobalFitGroups=4,
                maskTypeProcedure=None,
            )

    def test_rejects_invalid_n_global_fit_groups_string(self):
        with self.assertRaises(ValidationError):
            YSpaceFitInitialConditionsModel(
                fitModel="SINGLE_GAUSSIAN",
                nGlobalFitGroups="SOME",
                maskTypeProcedure=None,
            )

    def test_rejects_non_positive_n_global_fit_groups(self):
        with self.assertRaises(ValidationError):
            YSpaceFitInitialConditionsModel(
                fitModel="SINGLE_GAUSSIAN",
                nGlobalFitGroups=0,
                maskTypeProcedure=None,
            )

    def test_rejects_invalid_mask_type(self):
        with self.assertRaises(ValidationError):
            YSpaceFitInitialConditionsModel(
                fitModel="SINGLE_GAUSSIAN",
                nGlobalFitGroups=4,
                maskTypeProcedure="INVALID",
            )

    def test_accepts_none_mask_type(self):
        model = YSpaceFitInitialConditionsModel(
            fitModel="GC_C6",
            nGlobalFitGroups=2,
            maskTypeProcedure=None,
        )
        self.assertIsNone(model.maskTypeProcedure)


class TestBootstrapInitialConditionsModel(unittest.TestCase):
    def test_accepts_valid_bootstrap_conditions(self):
        model = BootstrapInitialConditionsModel(
            procedure="BACKWARD",
            fitInYSpace="FORWARD",
            bootstrapType="BOOT_RESIDUALS",
            nSamples=100,
        )
        self.assertEqual(model.nSamples, 100)

    def test_accepts_none_procedure(self):
        model = BootstrapInitialConditionsModel(
            procedure=None,
            fitInYSpace=None,
            bootstrapType="JACKKNIFE",
            nSamples=50,
        )
        self.assertIsNone(model.procedure)

    def test_rejects_invalid_procedure(self):
        with self.assertRaises(ValidationError):
            BootstrapInitialConditionsModel(
                procedure="INVALID",
                fitInYSpace=None,
                bootstrapType="JACKKNIFE",
                nSamples=50,
            )

    def test_rejects_invalid_bootstrap_type(self):
        with self.assertRaises(ValidationError):
            BootstrapInitialConditionsModel(
                procedure="BACKWARD",
                fitInYSpace=None,
                bootstrapType="NOT_A_TYPE",
                nSamples=50,
            )

    def test_rejects_zero_n_samples(self):
        with self.assertRaises(ValidationError):
            BootstrapInitialConditionsModel(
                procedure="BACKWARD",
                fitInYSpace=None,
                bootstrapType="JACKKNIFE",
                nSamples=0,
            )

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

    def test_complete_ic_does_not_call_shadow_validation_for_forward(self):
        ichelpers = self._import_ichelpers_with_mantid_stub()

        class _ForwardIC:
            firstSpec = 135
            lastSpec = 136
            masses = np.array([1.0079, 16.0])
            noOfMSIterations = 0
            maskedSpecAllNo = np.array([], dtype=int)

        class _WSIC:
            mode = "SingleDifference"
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
                    _ForwardIC, "thymol_10K_Gauss1D", _WSIC
                )
                mock_shadow.assert_not_called()

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


    def test_complete_ic_calls_forward_validator_for_forward(self):
        ichelpers = self._import_ichelpers_with_mantid_stub()

        class _ForwardICComplete:
            firstSpec = 135
            lastSpec = 136
            masses = np.array([1.0079, 16.0])
            noOfMSIterations = 0
            maskedSpecAllNo = np.array([], dtype=int)

        class _WSIC:
            mode = "SingleDifference"
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
                    "shadow_validate_forward_initial_conditions",
                ) as mock_fwd,
                patch.object(
                    ichelpers,
                    "shadow_validate_backward_initial_conditions",
                ) as mock_bwd,
            ):
                (Path(tmp) / "experiments" / "thymol_10K_Gauss1D").mkdir(
                    parents=True, exist_ok=True
                )
                ichelpers.completeICFromInputs(_ForwardICComplete, "thymol_10K_Gauss1D", _WSIC)
                mock_fwd.assert_called_once_with(_ForwardICComplete)
                mock_bwd.assert_not_called()

    def test_complete_yfit_ic_calls_yfit_validator(self):
        ichelpers = self._import_ichelpers_with_mantid_stub()

        class _YFitICComplete:
            fitModel = "SINGLE_GAUSSIAN"
            nGlobalFitGroups = 4
            maskTypeProcedure = "NAN"

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(ichelpers, "experimentsPath", Path(tmp) / "experiments"),
                patch.object(
                    ichelpers,
                    "shadow_validate_yspace_fit_initial_conditions",
                ) as mock_yfit,
            ):
                (Path(tmp) / "experiments" / "thymol_10K_Gauss1D" / "figures").mkdir(
                    parents=True, exist_ok=True
                )
                ichelpers.completeYFitIC(_YFitICComplete, "thymol_10K_Gauss1D")
                mock_yfit.assert_called_once_with(_YFitICComplete)

    def test_complete_boot_ic_calls_boot_validator(self):
        ichelpers = self._import_ichelpers_with_mantid_stub()

        class _BootICComplete:
            runBootstrap = False
            procedure = "BACKWARD"
            fitInYSpace = None
            bootstrapType = "BOOT_GAUSS_ERRS"
            nSamples = 10

        class _DummyIC:
            pass

        with (
            patch.object(
                ichelpers,
                "shadow_validate_bootstrap_initial_conditions",
            ) as mock_boot,
        ):
            ichelpers.completeBootIC(_BootICComplete, _DummyIC, _DummyIC, _DummyIC)
            mock_boot.assert_called_once_with(_BootICComplete)


if __name__ == "__main__":
    unittest.main(verbosity=2)
