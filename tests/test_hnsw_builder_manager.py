# tests/test_hnsw_builder_manager.py
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration

hnswlib = pytest.importorskip("hnswlib")

from luna_xai.neighbors.hnsw_builder import HNSWBuilder
from luna_xai.neighbors.hnsw_manager import HNSWManager


def test_hnsw_builder_and_manager_basic(tmp_path):
    # Dataset pequeno numérico
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 4)).astype(np.float32)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])

    builder = HNSWBuilder(space="l2", ef_construction=50, M=8, random_seed=42)
    index = builder.build_index(X)

    # save and reload to ensure persistence
    path = tmp_path / "index"
    builder.save_index(str(path))
    builder2 = HNSWBuilder()
    index2 = builder2.load_index(str(path))

    assert builder2.num_elements == builder.num_elements
    assert builder2.dim == builder.dim

    manager = HNSWManager(index=index2, data_ref=df, preprocessor=None, ef_search=50)

    # central instance
    instance = df.iloc[0]
    neighbors_df, dists = manager.find_neighbors(instance, n_neighbors=5)

    assert len(neighbors_df) == len(dists)
    assert len(neighbors_df) <= 5
    # strictly positive distances
    assert (dists > 0).all()
    # ordered
    assert np.all(np.diff(dists) >= 0)
