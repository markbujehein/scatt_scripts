from __future__ import annotations

from typing import Any
import warnings

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

HYDROGEN_MASS_TOLERANCE = 0.1

_VALID_PROCEDURES = frozenset({"BACKWARD", "FORWARD", "JOINT"})
_VALID_FIT_MODELS = frozenset({
    "SINGLE_GAUSSIAN", "GC_C4", "GC_C6", "GC_C4_C6",
    "DOUBLE_WELL", "ANSIO_GAUSSIAN", "MULTIVARIATE_GAUSSIAN",
})
_VALID_MASK_TYPES = frozenset({"NCP", "NAN"})
_VALID_BOOTSTRAP_TYPES = frozenset({"JACKKNIFE", "BOOT_RESIDUALS", "BOOT_GAUSS_ERRS"})


class _NcpICModel(BaseModel):
    """Shared validators for NCP-based scattering initial conditions.

    Provides common validation of ``masses`` and ``noOfMSIterations`` for both
    backward and forward scattering IC classes.  Field names mirror legacy IC
    attribute names.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    masses: list[float]
    noOfMSIterations: int

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


class BackwardInitialConditionsModel(_NcpICModel):
    """Pydantic shadow model for backward initial conditions.

    Extends ``_NcpICModel`` with the hydrogen-ratio cross-field constraint.
    Field names intentionally mirror legacy IC attribute names.
    """

    HToMassIdxRatio: float | None = None

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


class ForwardInitialConditionsModel(_NcpICModel):
    """Pydantic shadow model for forward initial conditions.

    Validates the same physical constraints as backward scattering
    (positive masses, non-negative MS iterations) but does not enforce
    hydrogen-ratio dependency, since forward detectors see all masses.
    """


class YSpaceFitInitialConditionsModel(BaseModel):
    """Pydantic shadow model for y-space fit initial conditions.

    Validates ``fitModel`` against known line-shape identifiers,
    ``nGlobalFitGroups`` as a positive integer or the literal ``"ALL"``,
    and ``maskTypeProcedure`` against the allowed masking strategies.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fitModel: str
    nGlobalFitGroups: int | str
    maskTypeProcedure: str | None = None

    @field_validator("fitModel")
    @classmethod
    def _validate_fit_model(cls, value: str) -> str:
        if value not in _VALID_FIT_MODELS:
            raise ValueError(
                f"fitModel '{value}' is not recognised. "
                f"Allowed values: {sorted(_VALID_FIT_MODELS)}."
            )
        return value

    @field_validator("nGlobalFitGroups", mode="before")
    @classmethod
    def _validate_n_global_fit_groups(cls, value: Any) -> Any:
        if isinstance(value, str):
            if value != "ALL":
                raise ValueError("nGlobalFitGroups string must be 'ALL'.")
        elif isinstance(value, int):
            if value < 1:
                raise ValueError("nGlobalFitGroups must be a positive integer or 'ALL'.")
        return value

    @field_validator("maskTypeProcedure")
    @classmethod
    def _validate_mask_type(cls, value: str | None) -> str | None:
        if value is not None and value not in _VALID_MASK_TYPES:
            raise ValueError(
                f"maskTypeProcedure '{value}' is not recognised. "
                f"Allowed values: {sorted(_VALID_MASK_TYPES)} or None."
            )
        return value


class BootstrapInitialConditionsModel(BaseModel):
    """Pydantic shadow model for bootstrap/jackknife initial conditions.

    Validates ``procedure`` and ``fitInYSpace`` against the allowed
    scattering-direction strings, ``bootstrapType`` against known
    resampling strategies, and ``nSamples`` as a positive integer.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    procedure: str | None = None
    fitInYSpace: str | None = None
    bootstrapType: str
    nSamples: int

    @field_validator("procedure", "fitInYSpace")
    @classmethod
    def _validate_procedure_flags(cls, value: str | None) -> str | None:
        if value is not None and value not in _VALID_PROCEDURES:
            raise ValueError(
                f"procedure/fitInYSpace '{value}' is not recognised. "
                f"Allowed values: {sorted(_VALID_PROCEDURES)} or None."
            )
        return value

    @field_validator("bootstrapType")
    @classmethod
    def _validate_bootstrap_type(cls, value: str) -> str:
        if value not in _VALID_BOOTSTRAP_TYPES:
            raise ValueError(
                f"bootstrapType '{value}' is not recognised. "
                f"Allowed values: {sorted(_VALID_BOOTSTRAP_TYPES)}."
            )
        return value

    @field_validator("nSamples")
    @classmethod
    def _validate_n_samples(cls, value: int) -> int:
        if value < 1:
            raise ValueError("nSamples must be a positive integer.")
        return value


def _format_validation_error(err: ValidationError) -> str:
    details = []
    for item in err.errors():
        field = ".".join(str(part) for part in item.get("loc", ())) or "validation"
        details.append(f"{field}: {item.get('msg', 'invalid value')}")
    return "; ".join(details)


def _shadow_warn(model_cls: type, data: dict, stage: str) -> None:
    """Validate *data* against *model_cls* and emit a warning on failure."""
    try:
        model_cls.model_validate(data)
    except ValidationError as err:
        warnings.warn(
            f"[Pydantic shadow validation][{stage}] {_format_validation_error(err)}",
            RuntimeWarning,
            stacklevel=3,
        )


def shadow_validate_backward_initial_conditions(
    IC: Any, stage: str = "completeICFromInputs"
) -> None:
    """Run non-breaking shadow validation for backward initial conditions."""
    _shadow_warn(
        BackwardInitialConditionsModel,
        {
            "masses": getattr(IC, "masses", []),
            "noOfMSIterations": getattr(IC, "noOfMSIterations", 0),
            "HToMassIdxRatio": getattr(IC, "HToMassIdxRatio", None),
        },
        stage,
    )


def shadow_validate_forward_initial_conditions(
    IC: Any, stage: str = "completeICFromInputs"
) -> None:
    """Run non-breaking shadow validation for forward initial conditions."""
    _shadow_warn(
        ForwardInitialConditionsModel,
        {
            "masses": getattr(IC, "masses", []),
            "noOfMSIterations": getattr(IC, "noOfMSIterations", 0),
        },
        stage,
    )


def shadow_validate_yspace_fit_initial_conditions(
    IC: Any, stage: str = "completeYFitIC"
) -> None:
    """Run non-breaking shadow validation for y-space fit initial conditions."""
    _shadow_warn(
        YSpaceFitInitialConditionsModel,
        {
            "fitModel": getattr(IC, "fitModel", "SINGLE_GAUSSIAN"),
            "nGlobalFitGroups": getattr(IC, "nGlobalFitGroups", 1),
            "maskTypeProcedure": getattr(IC, "maskTypeProcedure", None),
        },
        stage,
    )


def shadow_validate_bootstrap_initial_conditions(
    IC: Any, stage: str = "completeBootIC"
) -> None:
    """Run non-breaking shadow validation for bootstrap initial conditions."""
    _shadow_warn(
        BootstrapInitialConditionsModel,
        {
            "procedure": getattr(IC, "procedure", None),
            "fitInYSpace": getattr(IC, "fitInYSpace", None),
            "bootstrapType": getattr(IC, "bootstrapType", "BOOT_GAUSS_ERRS"),
            "nSamples": getattr(IC, "nSamples", 1),
        },
        stage,
    )
