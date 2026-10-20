"""
LUNA XAI — Data Subpackage
==========================

Provides unified interfaces for data preprocessing, transformation,
and feature/label management across all LUNA domains.

Components
----------
DataInterface
    High-level wrapper around sklearn-style preprocessors and pipelines,
    ensuring consistent feature handling, transformations, and inverse mappings.

Design Goals
------------
- Make explainers agnostic to the underlying data pipeline:
    • Full sklearn Pipelines
    • Custom feature/label preprocessors
    • Minimal/no preprocessing (passthrough mode)

- Provide a stable and predictable interface for domains to request
  transformations without needing to know pipeline internals.

This module exposes only the public-facing API of the data subsystem.
"""

from luna_xai.data.base_interface import DataInterface

__all__ = [
    "DataInterface",
]
