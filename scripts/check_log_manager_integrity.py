"""
LogManager Integrity Check
==========================
Validates that the LogManager implementation (Phase 5) correctly captures
all 7 mandatory parameter-class categories and records environment metadata.

Run standalone:
    python scripts/check_log_manager_integrity.py

Exit codes:
    0 – all checks passed
    1 – one or more checks failed
    2 – LogManager module not found (implementation pending)
"""

from __future__ import annotations

import importlib
import inspect
import sys
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# The 7 mandatory parameter-class categories defined in the VESUVIO analysis
# entry-point scripts (e.g. BaH2_500C.py).
# ---------------------------------------------------------------------------
REQUIRED_PARAM_CLASSES: list[str] = [
    "LoadVesuvioBackParameters",
    "LoadVesuvioFrontParameters",
    "BackwardInitialConditions",
    "ForwardInitialConditions",
    "YSpaceFitInitialConditions",
    "BootstrapInitialConditions",
    "UserScriptControls",
]

# Environment metadata keys that every log entry must contain.
REQUIRED_ENV_KEYS: list[str] = [
    "python_version",
    "platform",
    "numpy_version",
    "scipy_version",
    "iminuit_version",
    "timestamp",
    "git_commit",
]

# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str = ""


def _check_module_importable() -> CheckResult:
    """LogManager module must be importable from vesuvio_analysis."""
    try:
        importlib.import_module("vesuvio_analysis.core_functions.log_manager")
        return CheckResult("module_importable", True)
    except ModuleNotFoundError:
        return CheckResult(
            "module_importable",
            False,
            "vesuvio_analysis.core_functions.log_manager not found. "
            "Implement Phase 5 LogManager before merging to main.",
        )
    except (ImportError, AttributeError) as exc:
        return CheckResult("module_importable", False, f"Import error: {exc}")


def _check_class_exists(mod: Any) -> CheckResult:
    """Module must expose a 'LogManager' class."""
    if not hasattr(mod, "LogManager"):
        return CheckResult(
            "class_exists",
            False,
            "LogManager class not found in log_manager module.",
        )
    if not inspect.isclass(mod.LogManager):
        return CheckResult("class_exists", False, "'LogManager' is not a class.")
    return CheckResult("class_exists", True)


def _check_captures_param_classes(mod: Any) -> list[CheckResult]:
    """
    LogManager must define (or document) capture support for all 7 parameter
    classes.  We look for a class-level attribute 'TRACKED_PARAM_CLASSES' or
    equivalent that lists them.
    """
    results: list[CheckResult] = []
    cls = mod.LogManager

    # Strategy 1: explicit list attribute
    tracked: list[str] | None = None
    for attr in ("TRACKED_PARAM_CLASSES", "PARAM_CLASS_NAMES", "param_classes"):
        if hasattr(cls, attr):
            val = getattr(cls, attr)
            if isinstance(val, (list, tuple, set)):
                tracked = list(val)
                break

    if tracked is None:
        # Strategy 2: check instance method / signature for each class name
        instance_methods = [
            name for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
        ]
        source = inspect.getsource(cls)
        tracked = [pc for pc in REQUIRED_PARAM_CLASSES if pc in source]

    for param_class in REQUIRED_PARAM_CLASSES:
        found = param_class in tracked if tracked else False
        results.append(CheckResult(
            f"captures_{param_class}",
            found,
            "" if found else f"'{param_class}' not tracked by LogManager.",
        ))
    return results


def _check_env_metadata(mod: Any) -> list[CheckResult]:
    """
    LogManager must capture the required environment metadata keys.
    We check either a 'REQUIRED_ENV_KEYS' attribute or source-level presence.
    """
    results: list[CheckResult] = []
    cls = mod.LogManager
    source = inspect.getsource(cls)

    env_keys: list[str] | None = None
    for attr in ("REQUIRED_ENV_KEYS", "ENV_KEYS", "env_keys"):
        if hasattr(cls, attr):
            val = getattr(cls, attr)
            if isinstance(val, (list, tuple, set)):
                env_keys = list(val)
                break

    if env_keys is None:
        env_keys = [k for k in REQUIRED_ENV_KEYS if k in source]

    for key in REQUIRED_ENV_KEYS:
        found = key in (env_keys or []) or key in source
        results.append(CheckResult(
            f"env_key_{key}",
            found,
            "" if found else f"Environment metadata key '{key}' not captured by LogManager.",
        ))
    return results


def _check_log_method(mod: Any) -> CheckResult:
    """LogManager must expose a callable 'log' or 'record' method."""
    cls = mod.LogManager
    for method_name in ("log", "record", "capture", "write"):
        if hasattr(cls, method_name) and callable(getattr(cls, method_name)):
            return CheckResult("log_method_exists", True)
    return CheckResult(
        "log_method_exists",
        False,
        "No log/record/capture/write method found on LogManager.",
    )


def _check_serialisation(mod: Any) -> CheckResult:
    """LogManager must support serialisation (to_dict / to_json / save)."""
    cls = mod.LogManager
    for method_name in ("to_dict", "to_json", "save", "dump", "as_dict"):
        if hasattr(cls, method_name) and callable(getattr(cls, method_name)):
            return CheckResult("serialisation_method", True)
    return CheckResult(
        "serialisation_method",
        False,
        "No serialisation method (to_dict/to_json/save/dump) found on LogManager.",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_checks() -> int:
    """Execute all checks; return exit code."""
    all_results: list[CheckResult] = []

    # Step 1 — import
    import_result = _check_module_importable()
    all_results.append(import_result)

    if not import_result.passed:
        _report(all_results)
        return 2  # Implementation pending

    mod = importlib.import_module("vesuvio_analysis.core_functions.log_manager")

    # Step 2 — class existence
    class_result = _check_class_exists(mod)
    all_results.append(class_result)

    if not class_result.passed:
        _report(all_results)
        return 1

    # Step 3 — parameter classes
    all_results.extend(_check_captures_param_classes(mod))

    # Step 4 — environment metadata
    all_results.extend(_check_env_metadata(mod))

    # Step 5 — log method
    all_results.append(_check_log_method(mod))

    # Step 6 — serialisation
    all_results.append(_check_serialisation(mod))

    _report(all_results)

    failures = [r for r in all_results if not r.passed]
    return 0 if not failures else 1


def _report(results: list[CheckResult]) -> None:
    """Print a structured report to stdout."""
    print("\n" + "=" * 60)
    print("  LogManager Integrity Report")
    print("=" * 60)
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    for r in results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        msg = f"  {r.message}" if r.message else ""
        print(f"  [{status}] {r.name}{msg}")
    print("-" * 60)
    print(f"  Result: {passed}/{total} checks passed")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    sys.exit(run_checks())
