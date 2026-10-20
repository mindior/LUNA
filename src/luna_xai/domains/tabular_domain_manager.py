"""
TabularDomainManager
====================

High-level utility for creating and reusing tabular domains.

Stage 1 (offline / one-time):
-----------------------------
- Build HybridPreprocessor (CANE + scaler)
- Build HNSW index
- Save:
    - training snapshot         → X_train.parquet
    - training snapshot hash    → domain.json (train_hash)
    - index + metadata          → index.bin / index.meta.json
    - hybrid preprocessor       → preprocessor.pkl
    - domain metadata           → domain.json

Stage 2 (online / repeated):
----------------------------
- Reload HybridPreprocessor
- Reload HNSW index
- Reload X_train from Parquet snapshot
- Verify snapshot integrity via stored hash
- Recompute cached datasets from X_train
- Rebuild TabularDomain with a new model + pipelines
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd

from luna_xai.data import DataInterface
from luna_xai.domains.tabular_domain import TabularDomain
from luna_xai.neighbors.hnsw_builder import HNSWBuilder
from luna_xai.neighbors.hybrid_neighbor_manager import HybridNeighborManager
from luna_xai.utils import ensure_directory
from luna_xai.utils.preprocessing import HybridPreprocessor
from luna_xai.utils.logging_utils import get_luna_logger

LOGGER = get_luna_logger("domain.tabular.manager")


class TabularDomainManager:
    """
    High-level helper for building and persisting tabular domains.

    It **does** now save:
    - X_train snapshot in Parquet (`X_train.parquet`)
    - SHA-256 hash of that snapshot (in `domain.json`)

    And still saves:
    - HNSW index (+ metadata)
    - HybridPreprocessor (CANE + scaler)
    - Lightweight domain configuration
    """

    # ------------------------------------------------------------------
    @staticmethod
    def create_and_save(
        model: Any,
        pipeline_data: Any,
        pipeline_label: Any,
        X_train: pd.DataFrame,
        categorical_features: list[str],
        save_dir: str | Path,
        *,
        use_cane: bool = True,
        space: str = "l2",
    ) -> None:
        """
        Stage 1: build preprocessing + HNSW index and save artifacts.

        This function:
        - creates a Parquet snapshot of X_train;
        - computes a SHA-256 hash of the Parquet bytes;
        - builds HybridPreprocessor + HNSW index using X_train;
        - saves all artifacts + metadata.
        """
        save_dir = Path(save_dir)
        ensure_directory(save_dir)

        if not isinstance(X_train, pd.DataFrame):
            raise TypeError(
                "TabularDomainManager.create_and_save espera X_train como pandas.DataFrame."
            )

        LOGGER.info(
            "Starting domain creation and persistence in '%s' "
            "(space=%s, use_cane=%s, n_rows=%d, n_features=%d).",
            str(save_dir),
            space,
            use_cane,
            X_train.shape[0],
            X_train.shape[1],
        )

        # ------------------------------------------------------------------
        # 0) Snapshot of X_train → Parquet + hash (single conversion)
        # ------------------------------------------------------------------
        train_parquet_path = save_dir / "X_train.parquet"

        LOGGER.info(
            "Creating Parquet snapshot of training data at '%s' for reproducible domain reload.",
            str(train_parquet_path),
        )

        buffer = io.BytesIO()
        # Convert X_train to Parquet in memory (once only)
        X_train.to_parquet(buffer, index=False)
        parquet_bytes = buffer.getvalue()

        # Writes the same bytes to the Parquet file
        with train_parquet_path.open("wb") as f:
            f.write(parquet_bytes)

        # Calculate SHA-256 hash of snapshot
        train_hash = hashlib.sha256(parquet_bytes).hexdigest()

        LOGGER.info(
            "Training snapshot saved (Parquet) with SHA-256 hash=%s.",
            train_hash,
        )

        # ------------------------------------------------------------------
        # 1) Build the HybridPreprocessor (CANE + scaler) with the original X_train
        # ------------------------------------------------------------------
        preproc = HybridPreprocessor(
            categorical_features=categorical_features,
            use_cane=use_cane,
            verbose=True,
        )
        X_hnsw, _, _ = preproc.fit_transform(X_train)

        # ------------------------------------------------------------------
        # 2) Build HNSW index (CANONICAL API: index = build_index(X))
        # ------------------------------------------------------------------
        builder = HNSWBuilder(space=space)
        index = builder.build_index(X_hnsw)  # builder.dim and builder.num_elements filled

        # ------------------------------------------------------------------
        # 3) Save HNSW index (index.bin + index.meta.json)
        # ------------------------------------------------------------------
        index_base = save_dir / "index"
        builder.save_index(str(index_base))

        # ------------------------------------------------------------------
        # 4) Save HybridPreprocessor
        # ------------------------------------------------------------------
        preproc_path = save_dir / "preprocessor.pkl"
        preproc.save(str(preproc_path))

        # ------------------------------------------------------------------
        # 5) Save lightweight domain metadata (incluindo snapshot + hash)
        # ------------------------------------------------------------------
        domain_meta = {
            "space": space,
            "dim": builder.dim,
            "categorical_features": categorical_features,
            "use_cane": use_cane,
            "index_path": str(index_base),
            "preprocessor_path": str(preproc_path),
            "train_parquet_path": str(train_parquet_path),
            "train_hash": train_hash,
        }

        meta_path = save_dir / "domain.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(domain_meta, f, indent=4)

        LOGGER.info(
            "Domain artifacts successfully saved in '%s' "
            "(index_base='%s', preprocessor='%s', meta='%s', train_parquet='%s').",
            str(save_dir),
            str(index_base),
            str(preproc_path),
            str(meta_path),
            str(train_parquet_path),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def load(
        save_dir: str | Path,
        model: Any,
        pipeline_data: Any,
        pipeline_label: Any,
        categorical_features: list[str],
        output_dir: str | Path = "results",
        *,
        debug: bool = False,
        strict_hash_check: bool = True,
    ) -> TabularDomain:
        """
        Stage 2: reload a persisted domain for a new session.

        It is now **NOT** necessary to pass X_train.
        It is automatically reloaded from the Parquet snapshot saved during the creation step.

        Parameters
        ----------
        save_dir: string or Path
            Directory where domain artifacts were saved.
        model : Any
            Model trained to be used in the domain.
        pipeline_data : Any
            Data preprocessing pipeline (model).
        pipeline_label : Any
            Label preprocessing pipeline (if any).
        categorical_features : list[str]
            List of categorical features (fallback if meta does not contain).
        output_dir : str ou Path
            Directory for saving output artifacts (plots, logs, etc.).
        debug : bool, default False
            Enable debug logs in DataInterface and TabularDomain.
        strict_hash_check : bool, default True
            - If True: hash mismatch raises ValueError.
            - If False: only logs a WARNING and continues.
        """
        save_dir = Path(save_dir)
        meta_path = save_dir / "domain.json"

        if not meta_path.exists():
            raise FileNotFoundError(
                f"[TabularDomainManager] domain.json not found in {save_dir}"
            )

        LOGGER.info(
            "Loading persisted tabular domain from '%s' (meta='%s').",
            str(save_dir),
            str(meta_path),
        )

        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        space = meta.get("space", "l2")
        use_cane = meta.get("use_cane", True)
        meta_categoricals = meta.get("categorical_features", categorical_features)

        # Caminhos resolvidos
        index_base = Path(meta["index_path"])
        if not index_base.is_absolute():
            index_base = save_dir / index_base

        preproc_path = Path(meta["preprocessor_path"])
        if not preproc_path.is_absolute():
            preproc_path = save_dir / preproc_path

        train_parquet_path = Path(meta.get("train_parquet_path", ""))
        if not train_parquet_path.is_absolute():
            train_parquet_path = save_dir / train_parquet_path

        expected_hash = meta.get("train_hash")

        LOGGER.debug(
            "Resolved artifact paths: index_base='%s', preprocessor='%s', train_parquet='%s'.",
            str(index_base),
            str(preproc_path),
            str(train_parquet_path),
        )

        if not train_parquet_path.exists():
            raise FileNotFoundError(
                f"[TabularDomainManager] Training snapshot (Parquet) not found: {train_parquet_path}"
            )

        # ------------------------------------------------------------------
        # 1) Reads the snapshot once and verifies integrity via hash
        # ------------------------------------------------------------------
        with train_parquet_path.open("rb") as f:
            parquet_bytes = f.read()

        if expected_hash is not None:
            current_hash = hashlib.sha256(parquet_bytes).hexdigest()

            if current_hash != expected_hash:
                msg = (
                    "[TabularDomainManager] Training snapshot hash mismatch. "
                    "The saved Parquet may have been modified or corrupted."
                    f"Expected hash: {expected_hash}\n"
                    f"Current hash : {current_hash}\n"
                    f"Path         : {train_parquet_path}"
                )
                if strict_hash_check:
                    raise ValueError(msg)
                else:
                    LOGGER.warning(msg)

        # ------------------------------------------------------------------
        #2) Reloads X_train from bytes already read (without new disk access)
        # ------------------------------------------------------------------
        X_train = pd.read_parquet(io.BytesIO(parquet_bytes))

        LOGGER.info(
            "Loaded training snapshot from '%s' with shape=%s.",
            str(train_parquet_path),
            X_train.shape,
        )

        # ------------------------------------------------------------------
        # 3) Rebuild DataInterface from user pipelines
        # ------------------------------------------------------------------
        data_interface = DataInterface(
            preprocessor=pipeline_data,
            label_preprocessor=pipeline_label,
            debug=debug,
        )

        # ------------------------------------------------------------------
        # 4) Create domain skeleton
        # ------------------------------------------------------------------
        domain = TabularDomain(
            model=model,
            data_interface=data_interface,
            categorical_features=meta_categoricals,
            use_cane=use_cane,
            output_dir=str(output_dir),
            debug=debug,
        )

        # ------------------------------------------------------------------
        # 5) Load HybridPreprocessor
        # ------------------------------------------------------------------
        hybrid_preproc = HybridPreprocessor.load(str(preproc_path))

        # ------------------------------------------------------------------
        # 6) Load HNSW index (API: load_index(path) → index)
        # ------------------------------------------------------------------
        builder = HNSWBuilder(space=space)
        index = builder.load_index(str(index_base))

        # ------------------------------------------------------------------
        # 7) Recompute cached datasets from X_train
        # ------------------------------------------------------------------
        data_hnsw = hybrid_preproc.transform(X_train).reset_index(drop=True)
        data_original = X_train.reset_index(drop=True)

        # ------------------------------------------------------------------
        # 8) Wire everything back into the domain
        # ------------------------------------------------------------------
        domain.hybrid_preprocessor = hybrid_preproc
        domain.hnsw_builder = builder
        domain.index = index
        domain.data_hnsw = data_hnsw
        domain.data_original = data_original

        neighbor_manager = HybridNeighborManager(
            preprocessor_hnsw=hybrid_preproc,
            preprocessor_model=data_interface.preprocessor,
            index=index,
            data_base_hnsw=data_hnsw,
            data_base_original=data_original,
        )

        domain.neighbor_manager = neighbor_manager
        domain.neighbor_finder = neighbor_manager

        LOGGER.info(
            "Domain successfully loaded from '%s' with %d rows and %d features.",
            str(save_dir),
            data_original.shape[0],
            data_original.shape[1],
        )
        return domain
