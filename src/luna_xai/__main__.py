"""
LUNA XAI Command-Line Interface (CLI)
=====================================

LUNA stands for **Local Understanding through Neighborhood Analysis**.

This module provides a minimal entry point to verify that the
package is installed and importable:

    python -m luna_xai --help
    python -m luna_xai --version
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

# Centralized version string (to be kept in sync with the package)
LUNA_VERSION = "1.0.0"


def main(argv: Optional[list[str]] = None) -> None:
    """
    Entry point for the LUNA XAI CLI.

    Parameters
    ----------
    argv : list of str, optional
        Optional list of arguments. If None, ``sys.argv[1:]`` is used.
    """
    parser = argparse.ArgumentParser(
        prog="luna_xai",
        description=(
            "LUNA XAI — Local Understanding through Neighborhood Analysis\n"
            "Explainable AI framework for tabular models."
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show LUNA XAI version information and exit.",
    )

    args = parser.parse_args(argv)

    if args.version:
        print(f"LUNA XAI (Local Understanding through Neighborhood Analysis) version {LUNA_VERSION}")
        sys.exit(0)

    # Default: print help text
    parser.print_help()


if __name__ == "__main__":
    main()
