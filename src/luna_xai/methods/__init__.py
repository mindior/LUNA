"""
Methods Subpackage for LUNA XAI
===============================

Contains all sensitivity analysis algorithms used to generate
local or neighborhood-level sensitivity data.

Currently includes
------------------
- DSAMethod: Dual-Space Deterministic Sensitivity Analysis.

Notes
-----
All methods automatically register themselves in the global
LUNA registry upon import, enabling discovery through
`auto_discover_core_components()` in `luna_xai.core`.
"""

from luna_xai.methods.dsa_method import DSAMethod
from luna_xai.core import registry  # ensures registry is active

__all__ = [
    "DSAMethod",
]
