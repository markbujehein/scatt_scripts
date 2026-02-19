import unittest
import warnings

import numpy as np
from pydantic import ValidationError

from vesuvio_analysis.core_functions.ic_validation import (
    BackwardInitialConditionsModel,
    shadowValidateBackwardInitialConditions,
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
            shadowValidateBackwardInitialConditions(ic)
        self.assertGreaterEqual(len(caught), 1)
        self.assertIn("Pydantic shadow validation", str(caught[0].message))
