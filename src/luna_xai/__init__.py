"""
LUNA XAI — two-stage explanation workflow
=========================================

LUNA explanations are produced in two stages:

Stage 1 — Domain creation (run once per model + training data)
--------------------------------------------------------------
In this stage, LUNA builds and persists the tabular domain artifacts:
- HybridPreprocessor (CANE + scaler)
- HNSW index for neighborhood search
- Minimal domain metadata (JSON)

These artifacts are saved to disk and can be reused across sessions,
as long as the model and training data remain the same.

Stage 2 — Explanations using the persisted domain
-------------------------------------------------
In this stage, the saved domain is loaded and used to:
- Reconstruct the neighborhood structure
- Run the chosen sensitivity method and metric
- Render plots (importances, line plots, boxplots) for one or more instances

Stage 1 only needs to be executed once per (model, training data) pair;
Stage 2 can then be called repeatedly to explain any test instance.
"""
