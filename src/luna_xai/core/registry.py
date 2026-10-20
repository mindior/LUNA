"""
LUNA XAI Global Component Registry
==================================

Provides a unified, extensible registry for registering, discovering,
and retrieving core components such as:
- Domains
- Methods
- Metrics
- Renderers
- Explainers

This registry acts as the central lookup table for all component
implementations across the framework (Strategy / Factory pattern).
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable, Dict, Type

from luna_xai.utils.logging_utils import get_luna_logger

# Module-level logger for registry operations.
# By default, application code is expected to configure handlers so that
# these logs are written to a file (e.g., via a FileHandler attached to
# the "luna_xai" logger). If you prefer console output instead, you can
# configure the "luna_xai" logger with a StreamHandler or via
# `configure_luna_logging`.
LOGGER = get_luna_logger("registry")

# ---------------------------------------------------------------------
# Global registry container (namespace → name → class)
# ---------------------------------------------------------------------
REGISTRY: Dict[str, Dict[str, Type[Any]]] = {}


# ---------------------------------------------------------------------
# Component registration
# ---------------------------------------------------------------------
def register_component(
    namespace: str,
    name: str,
    cls: Type[Any] | None = None,
    *,
    overwrite: bool = False,
) -> Callable[[Type[Any]], Type[Any]]:
    """
    Register a component under a given namespace and name.

    Supports both decorator and direct invocation usage.

    Parameters
    ----------
    namespace : str
        Logical group of components (e.g., 'metric', 'method', 'renderer').
    name : str
        Component identifier within the namespace.
    cls : type, optional
        If provided, the registration happens immediately.
        If omitted, the function returns a decorator.
    overwrite : bool, default=False
        Whether existing entries can be replaced. Useful in rapid prototyping
        but should be False in production for safety.

    Returns
    -------
    Callable
        A decorator if `cls` is None, otherwise returns the registered class.

    Raises
    ------
    ValueError
        If attempting to register a duplicate name without overwrite=True.
    """

    if not isinstance(namespace, str) or not isinstance(name, str):
        raise TypeError("Both 'namespace' and 'name' must be strings.")

    namespace = namespace.strip().lower()
    name = name.strip().lower()

    if not namespace or not name:
        raise ValueError("Both 'namespace' and 'name' must be non-empty strings.")

    def decorator(inner_cls: Type[Any]) -> Type[Any]:
        REGISTRY.setdefault(namespace, {})

        # Prevent accidental overriding unless explicitly allowed
        if not overwrite and name in REGISTRY[namespace]:
            raise ValueError(
                f"Component '{namespace}.{name}' is already registered "
                f"as {REGISTRY[namespace][name].__name__}. "
                f"Use overwrite=True to replace."
            )

        REGISTRY[namespace][name] = inner_cls

        # Structured debug logging for registry actions
        LOGGER.debug(
            "Registered component '%s.%s' → %s",
            namespace,
            name,
            inner_cls.__name__,
        )

        return inner_cls

    return decorator if cls is None else decorator(cls)


# ---------------------------------------------------------------------
# Component lookup
# ---------------------------------------------------------------------
def get_component(full_name: str) -> Type[Any]:
    """
    Retrieve a component by its fully qualified name: 'namespace.name'.

    Parameters
    ----------
    full_name : str
        Identifier of the component ('method.dsa', 'metric.ch', etc.)

    Returns
    -------
    type
        The class registered under the requested name.

    Raises
    ------
    KeyError
        If the component or namespace is missing.
    ValueError
        If the format is invalid.
    """
    if "." not in full_name:
        raise ValueError(
            f"Invalid component identifier '{full_name}'. Expected format 'namespace.name'."
        )

    namespace, name = full_name.split(".", maxsplit=1)
    namespace = namespace.strip().lower()
    name = name.strip().lower()

    try:
        cls = REGISTRY[namespace][name]
        return cls
    except KeyError as exc:
        available = list_registered(namespace).keys()
        LOGGER.error(
            "Component '%s' not found in registry. Available in namespace '%s': %s",
            full_name,
            namespace,
            list(available),
        )
        raise KeyError(
            f"Component '{full_name}' not found in registry. "
            f"Available: {available}"
        ) from exc


# ---------------------------------------------------------------------
# Auto-discovery of LUNA core components
# ---------------------------------------------------------------------
CORE_MODULES = [
    "luna_xai.domains",
    "luna_xai.methods",
    "luna_xai.metrics",
    "luna_xai.renderers",
    "luna_xai.explainers",
]


def auto_discover_core_components(verbose: bool = False) -> None:
    """
    Import all submodules under known core namespaces to trigger
    auto-registration via decorator pattern.

    Parameters
    ----------
    verbose : bool, default=False
        Whether to log warnings during discovery.
    """
    for base_pkg in CORE_MODULES:
        try:
            pkg = importlib.import_module(base_pkg)

            if not hasattr(pkg, "__path__"):
                # Not a package (should not happen in normal usage)
                continue

            for _, mod_name, _ in pkgutil.iter_modules(pkg.__path__):
                importlib.import_module(f"{base_pkg}.{mod_name}")

            if verbose:
                LOGGER.debug("Auto-discovered components under '%s'.", base_pkg)

        except Exception as exc:
            if verbose:
                LOGGER.warning(
                    "Could not load core package '%s' during auto-discovery: %s",
                    base_pkg,
                    exc,
                )


# ---------------------------------------------------------------------
# Registry introspection
# ---------------------------------------------------------------------
def list_registered(namespace: str | None = None) -> Dict[str, Any]:
    """
    Return dictionary of registered components.

    Parameters
    ----------
    namespace : str, optional
        If provided, return only entries under that namespace.

    Returns
    -------
    dict
        Mapping of registered classes.
    """
    if namespace is None:
        return REGISTRY
    return REGISTRY.get(namespace.lower(), {})
