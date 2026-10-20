# LUNA-XAI 
***Local Understanding through Neighborhood Analysis for Explainable AI***

LUNA is a Python library for **model-agnostic explainability** on **tabular** machine learning models.  
It focuses on **local explanations** built from **neighborhood analysis** (HNSW), **sensitivity analysis** (DSA-style), and a **Calinski–Harabasz-based importance metric** with robust normalization.

The current version accepts data managed by pandas and sklearn models and pipelines.

LUNA is designed to be:

- **Model-agnostic** — works with any sklearn-like estimator (regression or classification).
- **Tabular-first** — optimized for tabular problems with mixed numerical & categorical features.
- **Neighborhood-based** — uses HNSW neighbors + synthetic sampling (LHS, CTGAN) for local exploration.
- **Metric-driven** — feature importances derived from CH-style separability scores, with optional arctan normalization.
- **Extensible** — pluggable methods, metrics, renderers, and domains.


---

## Main Features

- **Hybrid preprocessing for neighbors**
  - `HybridPreprocessor` with:
    - Optional **CANE IDF** encoding for categorical features.
    - Global scaling via `StandardScaler` (configurable).
    - Fully numeric representation to build HNSW indices.

- **Efficient neighborhood search**
  - HNSW index via `hnswlib`:
    - `HNSWBuilder` + `HNSWIndexManager`.
    - Save/load index and metadata separately from training data.

- **Local sensitivity analysis (DSA-style)**
  - `DSAMethod`:
    - Finds neighbors in hybrid space.
    - Generates synthetic samples via:
      - **Empirical LHS** (`LatinHypercube` from `scipy.stats.qmc`).
      - **CTGAN** (optional, if installed).
    - Perturbs features over discretized levels for sensitivity analysis.

- **CH-based metric with sign & normalization**
  - `CHMetric` + utilities:
    - `calculate_ch_index()` — per-feature CH-like separability across levels.
    - `adjust_signal()` — flips sign based on monotonic trend direction.
    - `arctan_normalize()` — squashes importances to `[-1, 1]` for stability.

- **Explainers**
  - `TabularRegressionExplainer`
    - Works on predictions `y_hat`.
    - 2-stage design:
      1. `generate_explanation_data()` → sensitivity data.
      2. `explain_from_data()` → importances + rendering.
  - `TabularClassificationExplainer`
    - Uses `predict_proba` for the **predicted class** (argmax).
    - Passes `target_class` down to the method for class-conditional analysis.

- **Plotting & rendering**
  - `PlotRenderer` + `MatplotlibPngExporter`:
    - Local feature importances (horizontal bars).
    - ICE-like line plots (feature trends).
    - Per-feature boxplots for sensitivity distributions.
  - Low-level plotting helpers in `luna_xai.utils.plotting`.

- **Logging utilities**
  - `logging_utils`:
    - `get_luna_logger()` — scoped loggers under `luna_xai`.
    - `configure_luna_logging()` — console/file configuration.
    - `log_execution_segment()` — context manager for timing segments.

---

## Documentation

Complete documentation, tutorials, and examples are available at:

[LUNA XAI Documentation](https://mindior.github.io/LUNA/)

## 📦 Installation

> **Prerequisites:** Python `>=3.11, <3.13` and a C++ toolchain (for `hnswlib`).

### Windows Installation Notes

On Windows, the `hnswlib` dependency may require a C++ compiler to be installed, as precompiled binaries may not be available for all Python versions.

If installation fails with an error such as:

```text
error: Microsoft Visual C++ 14.0 or greater is required.
```

Install the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select the **Desktop development with C++** workload, including the MSVC compiler and Windows SDK.

After installation, restart your terminal and retry:

```bash
pip install luna-xai
```

This requirement comes from the `hnswlib` dependency and does not require any changes to the LUNA source code.

## Disclaimer and Limitations

This package is provided under the MIT License and is intended to support the interpretation and analysis of machine learning models.

As with other explainable AI methods, the generated explanations are subject to methodological assumptions and may vary depending on the underlying model, data, and configuration. They should be interpreted within the context of the specific application and are not guaranteed to accurately or completely represent the model's behavior in every scenario.

Users are encouraged to assess the suitability of the explanations for their intended use and exercise appropriate judgment when incorporating them into decision-making processes.

The software is provided "AS IS", without warranties of any kind. To the maximum extent permitted by applicable law, the authors and contributors shall not be liable for any claims, damages, losses, or other liabilities arising from the use of, inability to use, or reliance upon the software or its outputs, including inaccurate, incomplete, or misleading explanations.

For further details, please refer to the [MIT License](LICENSE).