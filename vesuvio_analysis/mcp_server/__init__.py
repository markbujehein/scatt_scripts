"""MCP Server package for the VESUVIO analysis pipeline.

This package provides Model Context Protocol (MCP) server stubs that
bridge the LLM agent and the physical Mantid C++ backend, as specified
in the Tier 2 Tools proposal in ``AGENTIC_ENVIRONMENT.md``.

Modules
-------
mantid_ads_server
    Exposes the Mantid ``AnalysisDataService`` (``mtd``) state as an MCP
    tool so that agents can query workspace names, shapes, and statistics
    without running the full pipeline.

environment_server
    Exposes Pixi/Conda environment metadata (Python version, package
    versions, platform) as an MCP resource, enabling self-correcting
    agents to diagnose ``AttributeError`` / ``RuntimeError`` failures
    caused by version mismatches.
"""
