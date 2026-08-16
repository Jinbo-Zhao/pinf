# Plausible Inference

This repository contains code for **plausible inference** methods (Zhao et al. 2025) for analyzing the outputs of simulation experiments. The methods involve optimization sub-routines built on
[Pyomo](https://www.pyomo.org/). Given a set of simulated solutions — each with a
confidence region on its objective value(s) — and assumptions about the structure of
the underlying objective function (e.g. convexity, Lipschitz continuity), this package
performs inference tasks including:

- **Screening:** Could a candidate solution `x0` be *acceptable* (e.g. optimal) for some
  objective function that is consistent with the data and assumptions?
- **Plausible intervals:** For each objective, what are the largest / smallest objective values at
  `x0` over all objective values that are consistent with the data and assumptions?
- **Pixelization:** which regions of the decision (input) space or objective (output) space
  correspond to plausibly acceptable solutions?

---

## Installation

The package uses a `src/` layout (package name `plausible-inference`).

```bash
# core
pip install -e .

# with the Streamlit GUI extras
pip install -e ".[gui]"

# dev (pytest, ruff)
pip install -e ".[dev]"
```

Requires Python ≥ 3.10, NumPy, SciPy, Pyomo, and a MILP/NLP **solver** (see
[Solvers](#solvers)).

---

## Main Elements

| Concept | Meaning |
|---|---|
| **Experimental set** | `exp_set`, shape `(k, s)`: `k` simulated solutions in `s`-dim decision space. |
| **`m[i, d]`** | Objective `d` at point `i`. Index `0` is the candidate `x0`; `1..k` are the simulated points. |
| **Functional properties** | Structural assumptions on the objective: `convexity`, `concavity`, `linearity`, `Lipschitz_continuity`, `directional_Lipschitz_continuity` (one set per objective). |
| **Acceptability** | What makes `x0` "acceptable" (see below). All minimization. Defined in terms of `m[0, :]`. |
| **Confidence region** | Constrains each simulated `m[i, d]` to a region implied by its statistics (or explicit bounds). |
| **Discrepancy type** | How the confidence region is defined from statistics. |

### Inference Types

`screening`, `upper_plausible_interval`, `lower_plausible_interval`,
`input_pixelization`, `output_pixelization`.

Plausible intervals are computed **one objective at a time**: `model_construction` takes
`interval_objective_index` (1-based, default 1) selecting which objective `m[0, d]` is
bounded. The GUI's `plausible_intervals` procedure sweeps every objective and reports a
`[lower, upper]` pair per objective (`plausible_{upper,lower}_bound_obj{d}`).

### Acceptabilities

| Name | Constraint | Parameters |
|---|---|---|
| `single-objective-optimality` | `m[0,1] ≤ m[i,1]` ∀i (gradient `g[0]=0` under convex/concave) | — |
| `delta-optimality` | `m[0,1] − m[i,1] ≤ δ` ∀i (within δ of the best) | `{'delta': ≥0}` |
| `feasibility` | `m[0,1] ≤ c` | `{'threshold': c}` |
| `within-threshold` | `c − δ ≤ m[0,1] ≤ c + δ` | `{'threshold': c, 'delta': ≥0}` |

Multi-objective: `Pareto-optimality` (`x0` not dominated).

Parameters are passed via `acceptability_parameters`.

### Discrepancy Types

`norm_1`, `norm_2`, `norm_infinite`, `CRN` (from per-point statistics, with a
Bonferroni-corrected cutoff across objectives), or `confidence_region`
(use explicit precomputed `upper/lower_confidence_bounds`).

---

## Quick Start

```python
import numpy as np
from plausible_inference.model_construction import model_construction
from plausible_inference.utils.solver_config import get_plausibility_solver
from plausible_inference.execution_engine import ExecutionConfig, run_execution_engine

# Two simulated solutions in 2-D, single objective, with explicit confidence bounds.
exp_set = np.array([[0.0, 0.0], [1.0, 1.0]])
lower   = np.array([[1.0], [2.0]])
upper   = np.array([[2.0], [3.0]])

# Build a screening model: is x0 a near-optimal (within 0.5) point for some convex
# objective consistent with the data?
model = model_construction(
    exp_set,
    num_objectives=1,
    inference_type="screening",
    functional_properties_list=[["convexity"]],
    acceptability="delta-optimality",
    acceptability_parameters={"delta": 0.5},
    discrepancy_type="confidence_region",
    upper_confidence_bounds=upper,
    lower_confidence_bounds=lower,
    feasibility_check_solver=None,   # skip the structural feasibility pre-check
)

solver = get_plausibility_solver("gurobi")
candidates = np.array([[5.0, 5.0]])           # one candidate x0, shape (n, s)
results = run_execution_engine(ExecutionConfig(inference_type="screening"), model, solver, candidates)
print(results)   # DataFrame: idx, solve_status='completed', result='returned'/'screened_out'
```

Using statistics instead of explicit bounds:

```python
model = model_construction(
    exp_set,
    num_objectives=1,
    inference_type="screening",
    functional_properties_list=[["convexity"]],
    acceptability="single-objective-optimality",
    discrepancy_type="norm_infinite",
    sample_mean=np.array([[1.0], [2.0]]),
    sample_var=np.array([[0.5], [0.5]]),       # must be finite and > 0
    sample_size=100.0,
    confidence_level=0.95,                     # must be in (0, 1)
    feasibility_check_solver=None,
)
```

Once you have a model, use `run_execution_engine` to solve the inference over your
candidate points. It returns one `pandas.DataFrame` with a row per point:

```python
from plausible_inference.execution_engine import ExecutionConfig, run_execution_engine

results = run_execution_engine(
    ExecutionConfig(inference_type="screening"),
    model,
    solver,
    candidates,
)
```

---

## Solvers

Plausible inference must distinguish **infeasible** from **unbounded**, which solver
presolve can otherwise hide (`INF_OR_UNBD`). Always create the solver via
`get_plausibility_solver(name)`, which sets the right options:

- **Gurobi** — `DualReductions = 0`
- **SCIP** — disables strong/weak dual reductions

GLPK is **not** supported for plausibility inference.

The structural feasibility
pre-check (`feasibility_check_solver`) supports `gurobi` / `gurobi_direct` / `scip`.

---

## Streamlit GUI

```bash
streamlit run GUI/GUI_beta.py
```

The GUI guides the user through problem setup, acceptability + threshold inputs, simulation
upload (statistics or bounds templates), procedure-specific inputs (candidate points or
pixel grids), and runs the inference with downloadable results.

---

## Project Layout

```
src/plausible_inference/
├── model_construction.py        # build the Pyomo model (entry point)
├── execution_engine.py         # batch-solve the optimization problems (run_execution_engine)
├── rules/
│   ├── functional_structure.py  # convexity / concavity / linearity / Lipschitz
│   ├── acceptability.py         # acceptability constraints
│   ├── confidence_regions.py    # discrepancy measures + cutoffs
│   └── auxiliary_constraints.py # abs/max linearization helpers
└── utils/
    ├── cutoff_calculation.py    # Monte-Carlo / closed-form discrepancy cutoffs
    ├── model_construction_inputs.py        # input validation
    ├── functional_properties_validation.py # property-parameter validation
    └── solver_config.py         # solver factory + time limits
GUI/                             # Streamlit app + helpers + example data
```

---

## References

- Zhao, J., G. Keslin, D. J. Eckman, and B. L. Nelson. "[Methods of Plausible Inference: The Definitive Cookbook](https://www.informs-sim.org/wsc25papers/inv171.pdf)". Proceedings of the 2025 Winter Simulation Conference. 2025. 88–102.
