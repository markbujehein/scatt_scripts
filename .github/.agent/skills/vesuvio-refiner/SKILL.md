---
name: vesuvio-refiner
description: Iteratively refines the VESUVIO analysis pipeline. Updates legacy code to Pydantic 2.0 and manages experimental trial runs, with specific handling for long-running Monte Carlo simulations.
---

# Vesuvio Evolution Skill

This skill manages the iterative modernization of a 10+ year old Python codebase (legacy C++ port). It focuses on the transition to Pydantic 2.0 and ensures efficient use of time during expensive neutron scattering simulations.

## Core Files & Context
- **Entry Point**: `./run_vesuvio_analysis.py`
- **Internal Library**: `scatt_scripts/`
- **Experiment Config**: `experiments/CYM_10K.yaml`
- **Execution**: `pixi run python run_vesuvio_analysis.py experiments/CYM_10K.yaml`

## Decision Tree
1. **Does the pipeline execute successfully?**
   - **No (Traceback/Error)**: Analyze output -> Identify fault in entry script or library -> Propose Pydantic 2.0-compliant fix.
   - **Yes**: Proceed to Step 2.
2. **Is "VesuvioCalculateMS" detected in terminal output?**
   - **Yes**: **BREAKPOINT REACHED.** The library has started a Monte Carlo simulation (Multiple Scattering correction). 
     - **Action**: Stop execution or pause and ask the user if they wish to wait (10+ minutes) or terminate to continue refactoring other modules.
   - **No**: Proceed to Step 3.
3. **Is further verification required?**
   - **Yes**: Select an untested boolean flag in `experiments/CYM_10K.yaml` -> Toggle state -> Propose new trial.

## Implementation Standards
- **Modernization**: Replace physicists' "manual" data parsing with **Pydantic 2.0 Models**.
- **Efficiency**: Avoid unnecessary Monte Carlo runs during pure code-style refactoring. If a change only affects UI or logging, suggest disabling MS flags in the YAML first.
- **Scientific Integrity**: Maintain all physical constants and coordinate systems for TOSCA/VESUVIO.

## Operational Workflow
1. **Pre-Run**: Sync state with `experiments/CYM_10K.yaml`.
2. **Mandatory User Gate**: Present code diffs or YAML changes. **Wait for explicit user approval.**
3. **Execution & Monitoring**: Stream terminal output. If `VesuvioCalculateMS` appears, immediately trigger the Breakpoint protocol.
4. **Iterate**: Use results to inform the next refactoring step.