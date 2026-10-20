"""
ExplainerFactory
================

Factory responsible for assembling LUNA XAI explainers.

Supported scenarios
-------------------
1) Classic mode:
   - You pass the model, X_train, and the factory:
     • creates the TabularDomain
     • builds the HNSW index
     • instantiates method, metric, and renderer

2) Pre-loaded domain:
   - You load a TabularDomain via TabularDomainManager.load(...)
   - Pass this domain in `domain=...`
   - The factory does not recreate the index or the domain.

3) bypass_index=True:
   - Creates the domain but does not call build_index()
   - Useful for tests/unit tests or domains that do not rely on neighborhood.
"""

from __future__ import annotations

from typing import Any, Optional

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.pipeline import Pipeline

from luna_xai.data import DataInterface
from luna_xai.core.registry import get_component
from luna_xai.utils.normalization import arctan_normalize
from luna_xai.utils.logging_utils import get_luna_logger

# Module-level logger for factory operations.
# Application code is expected to configure handlers so that
# these logs are written to a log file (e.g., via a FileHandler attached
# to the "luna_xai" logger). If you prefer console output instead, you can
# configure the "luna_xai" logger with a StreamHandler or via
# `configure_luna_logging`.
LOGGER = get_luna_logger("factory.explainer")


class ExplainerFactory:
    """
    Flexible factory to assemble explainers from registered components.

    Supported scenarios
    -------------------
    1) Classic mode:
       - You pass the model and X_train, and the factory:
         • creates the TabularDomain
         • builds the HNSW index
         • instantiates method, metric, and renderer

    2) Pre-loaded domain:
       - You load a TabularDomain via TabularDomainManager.load(...)
       - Pass this domain in `domain=...`
       - The factory does not recreate the index or the domain.

    3) bypass_index=True:
       - Creates the domain but does not call build_index()
       - Useful for tests/unit tests or domains that do not rely on neighborhood.
    """

    @staticmethod
    def create(
        model: Any,
        X_train: Optional[Any] = None,
        categorical_features: Optional[list] = None,
        output_dir: str = "results",
        task: Optional[str] = None,
        domain_name: str = "tabular",
        method_name: str = "dsa",
        metric_name: str = "ch",
        renderer_name: str = "plot",
        normalizer: Optional[Any] = arctan_normalize,
        data_interface: Optional[DataInterface] = None,
        *,
        # Pre-constructed domain (e.g., from TabularDomainManager.load)
        domain: Optional[Any] = None,
        # Disables HNSW index creation even if X_train is provided
        bypass_index: bool = False,
        # Enables debug mode on created components (when supported)
        debug: bool = False,
        # Flag propagated to TabularDomain (when applicable)
        use_cane: bool = True,
    ) -> Any:
        """
        Create a fully configured explainer.

        Parameters
        ----------
        model : Any
            Trained model (or sklearn Pipeline).
        X_train : Any, optional
            Training data used to build the index when the domain is created here.
        categorical_features : list, optional
            Categorical feature names for TabularDomain.
        output_dir : str
            Directory where artifacts (plots, etc.) will be saved.
        task : {"classification", "regression"}, optional
            Task type. If None, it will be inferred from the model.
        domain_name, method_name, metric_name, renderer_name : str
            Component names in the registry (e.g., "tabular", "dsa", "ch", "plot").
        normalizer : callable, optional
            Normalization function applied to importances (e.g., arctan_normalize).
        data_interface : DataInterface, optional
            Preprocessing interface. If None and an sklearn Pipeline is provided,
            it will be created automatically from the pipeline transformer.

        domain : object, optional
            If provided, the factory WILL NOT create a new domain or call build_index().
        bypass_index : bool
            If True, never calls build_index(), even if X_train is provided.
        debug : bool
            If True, enables debug logging for created components (when supported).
        use_cane : bool
            Flag forwarded to TabularDomain (hybrid preprocessor choice).
        """

        LOGGER.info("[FACTORY] Initializing LUNA XAI explainer creation...")

        # --------------------------------------------------------------
        # 0) Use a pre-loaded domain if provided
        # --------------------------------------------------------------
        if domain is not None:
            LOGGER.info("[FACTORY] Using existing pre-loaded domain (TabularDomainManager).")
            DomainCls = type(domain)
        else:
            DomainCls = get_component(f"domain.{domain_name}")

        # --------------------------------------------------------------
        # 1) Extract internal model and transformer from sklearn Pipeline
        # --------------------------------------------------------------
        feature_transformer = None

        if isinstance(model, Pipeline):
            LOGGER.info("[FACTORY] Detected sklearn Pipeline — extracting internal model.")
            steps = dict(model.named_steps)
            feature_transformer = steps.get("preprocessor") or steps.get("transformer")
            model = model.steps[-1][1]

        # --------------------------------------------------------------
        # 2) Infer task type ("classification" or "regression")
        # --------------------------------------------------------------
        if task is None:
            if isinstance(model, ClassifierMixin) or hasattr(model, "predict_proba"):
                task = "classification"
            elif isinstance(model, RegressorMixin):
                task = "regression"
            else:
                raise ValueError(
                    "Cannot infer task type; specify task='classification' or 'regression'."
                )

        # --------------------------------------------------------------
        # 3) Get components from registry
        # --------------------------------------------------------------
        MethodCls = get_component(f"method.{method_name}")
        MetricCls = get_component(f"metric.{metric_name}")
        RendererCls = get_component(f"renderer.{renderer_name}")
        ExplainerCls = get_component(f"explainer.{task}")

        # --------------------------------------------------------------
        # 4) Build DataInterface only if the domain will be created here
        # --------------------------------------------------------------
        if domain is None:
            if data_interface is None:
                data_interface = DataInterface(
                    preprocessor=feature_transformer,
                    debug=debug,
                )
        else:
            # Pre-loaded domain is expected to already have its own data_interface.
            pass

        # --------------------------------------------------------------
        # 5) Create domain (if needed) and optionally build index
        # --------------------------------------------------------------
        if domain is None:
            LOGGER.info("[FACTORY] Creating new domain from scratch.")

            # TabularDomain has a specific signature:
            #   (model, data_interface, categorical_features, use_cane, debug, output_dir)
            if domain_name == "tabular":
                domain = DomainCls(
                    model=model,
                    data_interface=data_interface,
                    categorical_features=categorical_features or [],
                    use_cane=use_cane,
                    debug=debug,
                    output_dir=output_dir,
                )
            else:
                # Generic attempt for other domains while passing debug if possible
                try:
                    domain = DomainCls(
                        model=model,
                        data_interface=data_interface,
                        categorical_features=categorical_features,
                        output_dir=output_dir,
                        debug=debug,
                    )
                except TypeError:
                    # Fallback: remove debug/categorical_features if signature does not accept them
                    domain = DomainCls(
                        model=model,
                        data_interface=data_interface,
                        output_dir=output_dir,
                    )

            if bypass_index:
                LOGGER.info("[FACTORY] Index creation bypassed (bypass_index=True).")
            else:
                if X_train is None:
                    raise ValueError(
                        "X_train must be provided when creating a new domain "
                        "unless bypass_index=True."
                    )

                LOGGER.info("[FACTORY] Building hybrid-space HNSW index...")
                domain.build_index(X_train)
        else:
            LOGGER.info("[FACTORY] Domain already pre-loaded — skipping index creation.")

        # --------------------------------------------------------------
        # 5.5) Lightweight domain validation
        # --------------------------------------------------------------
        neighbor_finder = getattr(domain, "neighbor_finder", None)
        if neighbor_finder is None and not bypass_index:
            LOGGER.warning(
                "[FACTORY WARNING] Domain has no neighbor_finder. "
                "Methods that rely on neighborhood (e.g., DSA) will fail."
            )

        # --------------------------------------------------------------
        # 6) Instantiate method, metric, and renderer
        # --------------------------------------------------------------
        # Method and Metric follow BaseMethod/BaseMetric (debug is optional)
        try:
            method = MethodCls(debug=debug)
        except TypeError:
            method = MethodCls()

        method.domain = domain
        method.neighbor_finder = neighbor_finder

        try:
            metric = MetricCls(debug=debug)
        except TypeError:
            metric = MetricCls()

        # Renderer may or may not accept output_dir / debug
        renderer = None
        try:
            renderer = RendererCls(output_dir=output_dir, debug=debug)
        except TypeError:
            try:
                renderer = RendererCls(output_dir=output_dir)
            except TypeError:
                renderer = RendererCls(debug=debug)

        # --------------------------------------------------------------
        # 7) Assemble final explainer
        # --------------------------------------------------------------
        # Explainer is typically based on LunaExplainer (normalizer + debug)
        # or BaseExplainer (which might not accept these arguments yet).
        try:
            explainer = ExplainerCls(
                model=model,
                domain=domain,
                method=method,
                metric=metric,
                renderer=renderer,
                output_dir=output_dir,
                normalizer=normalizer,
                debug=debug,
            )
        except TypeError:
            # Fallback for older explainers without normalizer/debug in the signature
            explainer = ExplainerCls(
                model=model,
                domain=domain,
                method=method,
                metric=metric,
                renderer=renderer,
                output_dir=output_dir,
            )

        LOGGER.info("[FACTORY] Explainer successfully created (task=%s).", task)
        return explainer

# ---------------------------------------------------------------------
# Função helper de alto nível para criar explicador tabular
# ---------------------------------------------------------------------
def create_tabular_explainer(
    model: Any,
    X_train: Optional[Any] = None,
    *,
    task: Optional[str] = None,                 # "classification" or "regression"
    categorical_features: Optional[list] = None,
    output_dir: str = "results",
    method_name: str = "dsa",
    metric_name: str = "ch",
    renderer_name: str = "plot",
    normalizer: Optional[Any] = arctan_normalize,
    data_interface: Optional[DataInterface] = None,
    domain: Optional[Any] = None,              # domain already loaded (TabularDomainManager.load)
    bypass_index: bool = False,
    debug: bool = False,
    use_cane: bool = True,
) -> Any:
    """
    Convenient shortcut to create a TabularExplainer using the registry.

    If `domain` is passed, the index is not recreated.
    If `domain` is None, the factory creates the TabularDomain and the HNSW index (unless
    bypass_index=True).
    """
    LOGGER.info(
        "[FACTORY] create_tabular_explainer(task=%s, method=%s, metric=%s, renderer=%s)",
        task,
        method_name,
        metric_name,
        renderer_name,
    )

    return ExplainerFactory.create(
        model=model,
        X_train=X_train,
        categorical_features=categorical_features,
        output_dir=output_dir,
        task=task,
        domain_name="tabular",
        method_name=method_name,
        metric_name=metric_name,
        renderer_name=renderer_name,
        normalizer=normalizer,
        data_interface=data_interface,
        domain=domain,
        bypass_index=bypass_index,
        debug=debug,
        use_cane=use_cane,
    )