# Plausible Inference

[![Documentation Status](https://readthedocs.org/projects/pinf/badge/?version=latest)](https://pinf.readthedocs.io/en/latest/)

📖 **Documentation:** <https://pinf.readthedocs.io>

<!-- start: intro -->
This repository contains code for **plausible inference** methods (Zhao et al. 2025) for analyzing the outputs of simulation experiments. The methods involve optimization subroutines built on [Pyomo](https://www.pyomo.org/).

Given a set of simulated solutions, estimates of their objective values—provided as summary statistics (e.g., sample means and sample variances) or confidence regions—and assumptions about the structure of the underlying objective function (e.g., convexity, Lipschitz continuity), this package performs inference tasks including:

- **Screening:** Perform statistical inference or hypothesis testing on the acceptability (e.g., optimality) of a candidate solution `x_0`.
- **Plausible intervals:** Construct confidence intervals or regions for the objective value(s) at a given solution `x_0`, or confidence bands for the objective function over the feasible region.
- **Pixelization:** Construct confidence regions for the set of acceptable solutions in decision (input) space, or for its image in objective (output) space.
<!-- end: intro -->

---

## Installation
<!-- start: installation -->

Requires Python ≥ 3.10.

### Using the package

Install the package from PyPI:

```bash
pip install pinf
```

This also installs the Python interfaces of the Gurobi and SCIP solvers (see
the Solvers section). The package is imported as `pinf`.

To also install the Streamlit GUI (launched with `pinf-gui`, see the Streamlit
GUI section):

```bash
pip install "pinf[gui]"
```

### Working from a clone

The tests, the demo notebooks and the documentation sources live in this
repository but are not part of the published package. To use them, clone the
repository and install it in editable mode from the repository root, inside the
Python environment you will use (for example after `conda activate <env>`):

```bash
git clone https://github.com/Jinbo-Zhao/pinf.git
cd pinf
pip install -e ".[gui]"     # Streamlit GUI (pinf-gui) and the demo notebooks
pip install -e ".[dev]"     # tests and linting (pytest, ruff); includes gui
pip install -e ".[docs]"    # build the documentation locally
```

### Available extras

| Extra | Adds | Needed for |
|---|---|---|
| *(none)* `pip install pinf` | NumPy, pandas, SciPy, Pyomo, gurobipy, PySCIPOpt | Building and solving models |
| `gui` | Streamlit, Matplotlib, openpyxl | The Streamlit GUI (`pinf-gui`) and the demo notebooks |
| `dev` | pytest, ruff, and the `gui` extra | Tests and linting (from a clone) |
| `docs` | Sphinx, MyST, Read the Docs theme, sphinx-autoapi | Building the documentation (from a clone) |

Every extra already includes the core package. Combine extras in one command,
for example `pip install -e ".[gui,dev]"`.
<!-- end: installation -->

---

## Main Elements
<!-- start: concepts -->

| Concept | Meaning |
|---|---|
| **Experimental set** | `exp_set`, shape `(k, s)`: `k` simulated solutions in `s`-dim decision space. |
| **Plausible objective values** | `m[i, d]`: an intermediate modeling quantity representing objective `d` at solution `i`. Index `0` corresponds to the candidate solution `x_0`; indices `1..k` correspond to the simulated solutions. |
| **Functional properties** | Structural assumptions on the objective: `convexity`, `concavity`, `Lipschitz_continuity`, `directional_Lipschitz_continuity` (one set per objective). |
| **Acceptability** | What makes `x_0` "acceptable" (see below). Smaller values are assumed to be preferred for every objective. Defined in terms of `m[0, :]`. |
| **Confidence region** | A region that constrains the objective values at the simulated solutions, specified directly through bounds on each objective or constraints. |
| **Discrepancy type** | Specifies how to construct a confidence region from summary statistics. Only one of a discrepancy type or an explicitly specified confidence region is required. |

### Inference Types

All inference types use statistics computed from the simulation outputs at the simulated solutions, or a confidence region for their objective values.

- **`screening`**: Test whether a given solution `x_0` is one of the acceptable solutions.
- **`upper_plausible_interval`**: Construct an upper confidence bound on a selected objective value at a given solution `x_0`.
- **`lower_plausible_interval`**: Construct a lower confidence bound on a selected objective value at a given solution `x_0`.
- **`input_pixelization`**: Test whether a given region in solution (input) space intersects an implicitly defined confidence region for acceptable solutions. 
- **`output_pixelization`**: Test whether a given region in objective (output) space intersects an implicitly defined confidence region for the image of acceptable solutions under the objective function. 

Inference procedures based on the same simulation statistics or confidence region provide a **uniform confidence guarantee**: with probability at least `1 - alpha`, none of the resulting conclusions makes a Type I error. Here, `1 - alpha` is the confidence level of the provided confidence region, or the confidence level specified when selecting a discrepancy type.

Plausible intervals are computed one objective at a time: pass `interval_objective_index` (1-based, default `1`) to `model_construction` to choose which objective is bounded.

### Acceptabilities

Smaller values are assumed to be preferred for every objective. Objectives for which larger values are preferred can be multiplied by `-1` to follow this convention.

| Name | Definition | Parameters |
|---|---|---|
| `single-objective-optimality` | The solution achieves the smallest objective value over the feasible region. | — |
| `delta-optimality` | The solution's objective value is within `delta` of the smallest objective value over the feasible region. | `{'delta': ≥0}` |
| `feasibility` | The solution's objective value is at or below a specified `threshold`. | `{'threshold': c}` (scalar, or one value per objective) |
| `closeness-to-target` | The solution's objective value differs from a specified `threshold` by at most `delta`. | `{'threshold': c, 'delta': ≥0}` (each a scalar, or one value per objective) |
| `Pareto-optimality` | No other feasible solution is at least as good in every objective and strictly better in at least one objective. | — |

`single-objective-optimality` and `delta-optimality` apply to a single objective, and `Pareto-optimality` applies to multiple objectives. `feasibility` and `closeness-to-target` apply to any number of objectives, imposing the condition on each objective separately. Parameters are passed via `acceptability_parameters`.

### Discrepancy Types

A discrepancy is a distance measure that quantifies the disagreement between objective values and simulation observations.

For each objective `d`, let

$$
z_{i,d} = \frac{m[i,d] - \bar{Y}_{i,d}}{S_{i,d}/\sqrt{n_{i,d}}},
$$

where $\bar{Y}_{i,d}$, $S_{i,d}^2$, and $n_{i,d}$ are the sample mean, sample variance, and sample size at simulated solution `i`.

- **`norm_1`**: Sum the absolute standardized differences:

  $$
  D_d = \sum_{i=1}^{k} |z_{i,d}|.
  $$

- **`norm_2`**: Sum the squared standardized differences:

  $$
  D_d = \sum_{i=1}^{k} z_{i,d}^{2}.
  $$

- **`norm_infinite`**: Take the largest absolute standardized difference:

  $$
  D_d = \max_{i=1,\ldots,k} |z_{i,d}|.
  $$

- **`CRN`**: Use a covariance-adjusted squared distance to account for correlations between simulation outputs at different solutions, such as those induced by common random numbers:

  $$
  D_d = n_d(\mathbf{m}_d-\bar{\mathbf{Y}}_d)^\top
  \widehat{\Sigma}_d^{-1}
  (\mathbf{m}_d-\bar{\mathbf{Y}}_d).
  $$

  Here, $\mathbf{m}_d=(m[1,d],\ldots,m[k,d])^\top$, $\bar{\mathbf{Y}}_d$ is the vector of sample means, $\widehat{\Sigma}_d$ is the sample covariance matrix of simulation outputs, and $n_d$ is the common sample size across simulated solutions.

- **`confidence_region`**: Directly use a precomputed confidence region specified through `lower_confidence_bounds` and `upper_confidence_bounds`.

The three norm-based types assume independent sampling across simulated solutions and use `sample_mean`, `sample_var`, and `sample_size`. For `CRN`, `sample_var` provides a sample covariance matrix across simulated solutions for each objective.

For statistics-based types, the package constructs a confidence region by bounding each discrepancy using a cutoff determined by `confidence_level`, with a Bonferroni correction across objectives. For each objective `d`, the resulting region for $(m[1,d], \ldots, m[k,d])$ is a polyhedron for `norm_infinite` (a hyperrectangle) and `norm_1` (a cross-polytope), an axis-aligned ellipsoid for `norm_2`, and an ellipsoid whose axes follow $\widehat{\Sigma}_d$ for `CRN`; with multiple objectives, the overall region is the Cartesian product of these per-objective regions. For `confidence_region`, the region is the hyperrectangle given by the bounds, which should jointly cover the objective values across all simulated solutions and objectives at the intended confidence level.
<!-- end: concepts -->

---

## Quick Start
<!-- start: quickstart -->

**Before you run it:** install the package:

```bash
pip install pinf
```

```python
import numpy as np
from pinf.model_construction import model_construction
from pinf.utils.solver_config import get_plausibility_solver
from pinf.execution_engine import ExecutionConfig, run_execution_engine

# Two simulated solutions in 2-D, single objective, with explicit confidence bounds.
exp_set = np.array([[0.0, 0.0], [1.0, 1.0]])
lower   = np.array([[1.0], [2.0]])
upper   = np.array([[2.0], [3.0]])

# Build a screening model: is x_0 a near-optimal (within 0.5) point for some convex
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
candidates = np.array([[5.0, 5.0]])           # one candidate x_0, shape (n, s)
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
from pinf.execution_engine import ExecutionConfig, run_execution_engine

results = run_execution_engine(
    ExecutionConfig(inference_type="screening"),
    model,
    solver,
    candidates,
)
```
<!-- end: quickstart -->

---

## Solvers
<!-- start: solvers -->

Plausible inference must distinguish **infeasible** from **unbounded**, which solver
presolve can otherwise hide (`INF_OR_UNBD`). Always create the solver via
`get_plausibility_solver(name)`, which sets the right options:

- **Gurobi** — `DualReductions = 0`
- **SCIP** — `misc/allowstrongdualreds = False`, `misc/allowweakdualreds = False`

Models built with `norm_2` or `CRN` contain quadratic constraints; both solvers
handle them.

### Gurobi

Solver names: `gurobi`, `gurobi_direct`.

```bash
pip install pinf   # includes gurobipy
```

The pip-installed `gurobipy` includes a size-limited license that is enough for
small examples. Larger models need a full Gurobi license (free for academic use);
see the [Gurobi license documentation](https://support.gurobi.com/hc/en-us/articles/12684663118993-How-do-I-obtain-a-Gurobi-license).

### SCIP

SCIP is open source but can be reached through two Pyomo interfaces, which are
installed differently:

| Solver name | What it uses | How to install |
|---|---|---|
| `scip_direct` | PySCIPOpt (Python bindings that bundle the SCIP library) | included in `pip install pinf` |
| `scip` | the `scip` command-line executable | `conda install -c conda-forge scip`, or download an installer from [scipopt.org](https://www.scipopt.org/) |

`pip install pyscipopt` does **not** provide the `scip` executable, so with a
pip-only setup use `scip_direct`. To use `scip`, check that the executable is on
`PATH` with `scip --version`. `scip_direct` requires Pyomo 6.10.1 or
later, which `pip install pinf` installs.

### Checking a solver

```python
from pinf.utils.solver_config import get_plausibility_solver

solver = get_plausibility_solver("scip_direct")   # or "gurobi", "scip", ...
print(solver.available(exception_flag=False))      # True when Pyomo can use it
```

### Structural feasibility check

`model_construction` first checks that the functional-property and
confidence-region constraints are feasible. Choose the solver with
`feasibility_check_solver` (`gurobi`, `gurobi_direct`, `scip` or `scip_direct`;
default `scip_direct`), or pass `None` to skip the check.
<!-- end: solvers -->

---

## Streamlit GUI
<!-- start: gui -->

The GUI ships with the package. Install it with the GUI extra:

```bash
pip install "pinf[gui]"
```

Then start it from the directory where you want results to be saved:

```bash
pinf-gui
```

Streamlit prints a local URL (usually <http://localhost:8501>) and opens it in
your browser. Stop the app with Ctrl+C in the terminal. Results are written to
`inference_results/` under the launch directory. Options after `pinf-gui` are
passed to `streamlit run`, for example `pinf-gui --server.port 8600`.

The GUI guides the user through problem setup, acceptability + threshold inputs, simulation
upload (statistics or bounds templates), procedure-specific inputs (candidate points or
pixel grids), and runs the inference with downloadable results.
<!-- end: gui -->

---

## Project Layout

```
src/pinf/
├── model_construction.py        # build the Pyomo model (entry point)
├── execution_engine.py         # batch-solve the optimization problems (run_execution_engine)
├── rules/
│   ├── functional_structure.py  # convexity / concavity / linearity / Lipschitz
│   ├── acceptability.py         # acceptability constraints
│   ├── confidence_regions.py    # discrepancy measures + cutoffs
│   └── auxiliary_constraints.py # abs/max linearization helpers
├── utils/
│   ├── cutoff_calculation.py    # Monte-Carlo / closed-form discrepancy cutoffs
│   ├── model_construction_inputs.py        # input validation
│   ├── functional_properties_validation.py # property-parameter validation
│   └── solver_config.py         # solver factory + time limits
└── gui/
    ├── app.py                   # Streamlit app
    ├── helpers.py               # parsing, templates, argument builders
    ├── launcher.py              # `pinf-gui` entry point
    └── seeded_data/             # example CSV/XLSX inputs
```

---

## License

This project is licensed under the MIT License. See [`LICENSE`](https://github.com/Jinbo-Zhao/pinf/blob/main/LICENSE) for details.

Copyright (c) 2026 Jinbo Zhao and David J. Eckman.
## Testing
<!-- start: testing -->

```bash
pip install -e ".[dev]"
pytest
```

Solve tests need Gurobi and SCIP available through Pyomo. The package
installs `gurobipy` and PySCIPOpt (for `scip_direct`). Tests of the `scip`
interface also need the SCIP executable on `PATH`
(e.g. `conda install -c conda-forge scip`).
<!-- end: testing -->

---

## References
<!-- start: references -->

- Zhao, J., G. Keslin, D. J. Eckman, and B. L. Nelson. "[Methods of Plausible Inference: The Definitive Cookbook](https://www.informs-sim.org/wsc25papers/inv171.pdf)". Proceedings of the 2025 Winter Simulation Conference. 2025. 88–102.
<!-- end: references -->
