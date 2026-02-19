from __future__ import annotations

from typing import Any, List, Optional
import warnings

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator


class BackwardInitialConditionsModel(BaseModel):
    """Pydantic shadow model for backward initial conditions.

    This model is intentionally narrow and non-breaking: it validates
    core physical constraints while still accepting the existing class-based
    initial-condition objects used throughout the Mantid pipeline.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    masses: List[float]
    noOfMSIterations: int
    HToMassIdxRatio: Optional[float] = None

    @field_validator("masses", mode="before")
    @classmethod
    def _coerce_masses(cls, value: Any) -> List[float]:
        if isinstance(value, np.ndarray):
            return value.astype(float).tolist()
        return value

    @field_validator("masses")
    @classmethod
    def _validate_masses_positive(cls, masses: List[float]) -> List[float]:
        if len(masses) == 0:
            raise ValueError("masses must not be empty.")
        if any(mass <= 0 for mass in masses):
            raise ValueError("masses must all be positive.")
        return masses

    @field_validator("noOfMSIterations")
    @classmethod
    def _validate_ms_iterations(cls, value: int) -> int:
        if value < 0:
            raise ValueError("noOfMSIterations must be a non-negative integer.")
        return value

    @model_validator(mode="after")
    def _validate_h_ratio_dependency(self) -> "BackwardInitialConditionsModel":
        has_hydrogen = any(abs(mass - 1.0) / 1.0 < 0.1 for mass in self.masses)
        if (self.HToMassIdxRatio is not None) and (not has_hydrogen):
            raise ValueError(
                "HToMassIdxRatio can only be provided when hydrogen is present in masses."
            )
        return self


def shadowValidateBackwardInitialConditions(IC: Any, stage: str = "completeICFromInputs") -> None:
    """Run non-breaking validation and emit warnings instead of errors."""

    try:
        BackwardInitialConditionsModel.model_validate(
            {
                "masses": getattr(IC, "masses", []),
                "noOfMSIterations": getattr(IC, "noOfMSIterations", 0),
                "HToMassIdxRatio": getattr(IC, "HToMassIdxRatio", None),
            }
        )
    except ValidationError as err:
        warnings.warn(
            f"[Pydantic shadow validation][{stage}] {err}",
            RuntimeWarning,
            stacklevel=2,
        )
