from pathlib import Path
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class VesuvioBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class LoadVesuvioParameters(VesuvioBaseModel):
    runs: str
    empty_runs: Optional[str] = None
    spectra: str
    mode: str
    ipfile: Optional[str] = None
    scriptName: Optional[str] = None


class InitialConditions(VesuvioBaseModel):
    firstSpec: int
    lastSpec: int
    masses: np.ndarray
    initPars: np.ndarray
    bounds: np.ndarray
    maskedSpecAllNo: np.ndarray
    tofBinning: str
    maskTOFRange: Optional[str] = None
    MSCorrectionFlag: bool = False
    GammaCorrectionFlag: bool = False
    noOfMSIterations: int = 0
    scaleRaw: float = 1.0
    scaleEmpty: float = 1.0
    subEmptyFromRaw: bool = True
    modeRunning: Optional[str] = None
    name: Optional[str] = None
    scriptName: Optional[str] = None
    ipfile: Optional[str] = None
    InstrParsPath: Optional[str] = None
    mode: Optional[str] = None
    resultsSavePath: Optional[Path] = None
    ySpaceFitSavePath: Optional[Path] = None
    figSavePath: Optional[Path] = None
    userWsRawPath: Optional[Path] = None
    userWsEmptyPath: Optional[Path] = None
    runningSampleWS: bool = False
    runningPreliminary: bool = False
    runHistData: bool = False
    normVoigt: bool = True
    HToMassIdxRatio: Optional[float] = None
    massIdx: Optional[int] = None
    constraint_config: Optional["ConstraintConfig"] = None
    constraints: List[dict] = Field(default_factory=list)
    bootSavePath: Optional[Path] = None
    bootYFitSavePath: Optional[Path] = None
    logFilePath: Optional[Path] = None
    bootSavePathLog: Optional[str] = None
    bootYFitSavePathLog: Optional[str] = None
    maskedDetectorIdx: Optional[np.ndarray] = None
    # General conditions merged from YAML
    vertical_width: float = 0.0
    horizontal_width: float = 0.0
    thickness: float = 0.0
    transmission_guess: Optional[float] = None
    multiple_scattering_order: int = 2
    number_of_events: float = 1.0e5

    @computed_field
    @property
    def noOfMasses(self) -> int:
        return len(self.masses)

    @field_validator("masses", "initPars", "bounds", "maskedSpecAllNo", mode="before")
    @classmethod
    def convert_to_numpy(cls, v):
        if isinstance(v, list):
            return np.array(v)
        return v


class ConstraintConfig(VesuvioBaseModel):
    active: bool = False
    ratio: Optional[float] = None


class YSpaceFitInitialConditions(VesuvioBaseModel):
    scriptName: Optional[str] = None
    symmetrisationFlag: bool = False
    rebinParametersForYSpaceFit: str = ""
    fitModel: str = ""
    maskTypeProcedure: Optional[str] = None
    figSavePath: Optional[Path] = None
    showPlots: bool = False
    runMinos: bool = False
    globalFit: bool = False
    nGlobalFitGroups: int = 0


class BootstrapInitialConditions(VesuvioBaseModel):
    runBootstrap: bool = False
    procedure: Optional[str] = None
    fitInYSpace: Optional[str] = None
    bootstrapType: str = "JACKKNIFE"
    nSamples: int = 0
    skipMSIterations: bool = False
    userConfirmation: bool = True
    runningTest: bool = False
    allowOverwrite: bool = False
    runTimesPath: Optional[Path] = None
    scriptName: Optional[str] = None


class UserScriptControls(VesuvioBaseModel):
    procedure: Optional[str] = None
    fitInYSpace: Optional[str] = None
    runRoutine: bool = False


class BootstrapAnalysis(VesuvioBaseModel):
    runAnalysis: bool = False
    filterAvg: bool = False
    plotRawWidthsIntensities: bool = False
    plotMeanWidthsIntensities: bool = False
    plotMeansEvolution: bool = False
    plotYFitHists: bool = False
    plot2DHists: bool = False
    allowOverwrite: bool = False


class StoichiometryConfig(VesuvioBaseModel):
    molecule_name: str
    stoich_map: dict[str, int]
    formula: Optional[str] = None
    reference_atom: Optional[str] = None


class MetaConfig(VesuvioBaseModel):
    ip_folder: str
    ip_filename: str
