from __future__ import annotations

from typing import Any
import warnings

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

HYDROGEN_MASS_TOLERANCE = 0.1


class BackwardInitialConditionsModel(BaseModel):
    """Pydantic shadow model for backward initial conditions.

    This model is intentionally narrow and non-breaking: it validates
    core physical constraints while still accepting the existing class-based
    initial-condition objects used throughout the Mantid pipeline.
    Field names intentionally mirror legacy IC attribute names.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    masses: list[float]
    noOfMSIterations: int
    HToMassIdxRatio: float | None = None

    @field_validator("masses", mode="before")
    @classmethod
    def _coerce_masses(cls, value: Any) -> list[float]:
        if isinstance(value, np.ndarray):
            return value.astype(float).tolist()
        return value

    @field_validator("masses")
    @classmethod
    def _validate_masses_positive(cls, masses: list[float]) -> list[float]:
        if not masses:
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
        has_hydrogen = any(
            abs(mass - 1.0) < HYDROGEN_MASS_TOLERANCE for mass in self.masses
        )
        if (self.HToMassIdxRatio is not None) and (not has_hydrogen):
            raise ValueError(
                "HToMassIdxRatio can only be provided when hydrogen is present in masses."
            )
        return self


def _format_validation_error(err: ValidationError) -> str:
    details = []
    for item in err.errors():
        field = ".".join(str(part) for part in item.get("loc", ())) or "validation"
        details.append(f"{field}: {item.get('msg', 'invalid value')}")
    return "; ".join(details)


def shadow_validate_backward_initial_conditions(
    IC: Any, stage: str = "completeICFromInputs"
) -> None:
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
            f"[Pydantic shadow validation][{stage}] {_format_validation_error(err)}",
            RuntimeWarning,
            stacklevel=2,
        )
