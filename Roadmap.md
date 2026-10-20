# LUNA XAI – Roadmap

LUNA XAI is an evolving research-backed library for local explanation through neighborhood analysis.  
This roadmap is intentionally high-level: it communicates the overall direction of the project without exposing all research details, which may appear later in the form of academic publications and new releases.

The plan is organic and may change as new results, use cases, and collaborators emerge.

---

## Guiding principles

- **Research-driven**: new features should be grounded in experiments and, whenever possible, associated with scientific publications.
- **Practical & usable**: APIs and interfaces should remain simple enough for practitioners in industry.
- **Model-agnostic by design**: LUNA aims to work across different model families and ecosystems.
- **Transparent explanations**: prioritize explanations that are interpretable, auditable, and robust.

---

## Short-term (next releases)

These are the main priorities for the next development cycles.  
Each of them is a candidate topic for dedicated studies and publications.

### 1. Broader model ecosystem support

Extend LUNA beyond the current scikit-learn–centric workflow, making it easier to plug in:

- Models trained in other popular machine learning / deep learning frameworks.
- Custom prediction interfaces, as long as they expose a `predict` (and optionally `predict_proba`)–like API.

**Goal:** make LUNA usable in more real-world stacks without forcing users to rewrite their modeling code.

---

### 2. Interaction-aware explanations

Move from purely marginal/local feature effects to **interaction-aware** analysis, including:

- Local inspection of how pairs (or small sets) of features jointly influence the prediction.
- Neighborhood-based strategies to highlight where interactions are especially relevant.
- Visual summaries that help users understand “feature A depends on feature B” type behavior.

**Goal:** give users more insight into *how* features work together, not only their isolated contribution.

---

### 3. Explanation reliability / confidence scoring

Introduce a notion of **reliability** or **trust score** for each explanation, for example based on:

- Neighborhood quality and density.
- Local stability of the explanation under small perturbations.
- Internal consistency checks between different views of the same instance.

The exact formulation is under active research and may be refined over time.

**Goal:** help users distinguish between “strong, well-supported” explanations and “weak, unstable” ones.

---

## Medium-term directions

These ideas are planned for subsequent phases, after the short-term items above mature.

### 4. Beyond local: global and semi-global views

- Aggregate local analyses to derive **global patterns** (e.g., typical behavior segments or global ranking of influential features).
- Support **semi-global** views, such as explanations restricted to a slice of the data (e.g., a cluster or segment).
- Provide high-level summaries that remain faithful to the underlying local neighborhoods.

### 5. Enhanced visualization and text-based explanations

- Richer visual dashboards (e.g., more interactive or multi-panel views).
- Optional **textual summaries** of explanations, aimed at non-technical stakeholders.
- Improved export options for reports and presentations.

---

## Longer-term vision

These items are intentionally broad and exploratory:

- Apply neighborhood-based explanation ideas to other data modalities (e.g., time series or mixed tabular-temporal settings).
- Build tools that make it easier to audit, monitor, and compare explanations across models and versions over time.

---

## Notes on research and publications

Some of the directions listed above are being developed as **independent research projects and articles**.  
As results mature and are published, the corresponding techniques will be integrated into LUNA XAI in a stable, documented form.

The public roadmap will remain high-level by design, to balance:

- openness about the evolution of the library, and  
- the natural need to protect ongoing academic work until it is ready to be shared.

---

## Feedback & contributions

Suggestions, issues, and discussions are welcome via:

- GitHub Issues – for bugs, feature requests, and questions  
- Pull Requests – for improvements to documentation, demos, or core components

LUNA XAI is intended to grow as a community-driven, research-grounded toolkit for trustworthy neighborhood-based explanations.
