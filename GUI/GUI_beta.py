# =============================================================================
# Plausible Inference — Streamlit GUI
# =============================================================================
# Orchestrates user input, validation, calls to ``plausible_inference`` (Pyomo),
# and result export. Does not implement inference logic.
#
# Install (must run from REPOSITORY ROOT — where pyproject.toml lives, not GUI/):
#   cd /path/to/PI_Package
#   pip install -e ".[gui]"
#
# Run Streamlit (also from repository root so imports resolve):
#   streamlit run GUI/GUI_beta.py
#
# If your shell is already inside GUI/:
#   cd .. && pip install -e ".[gui]" && streamlit run GUI/GUI_beta.py
#   # or: streamlit run GUI_beta.py   (still recommended: cwd = repo root)
# =============================================================================
from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from gui_helpers import (
    PROP_LABEL_TO_INTERNAL,
    build_functional_properties_list,
    build_functional_properties_parameters,
    build_input_pixel_grid,
    build_output_pixel_grid,
    default_results_dir,
    ensure_package_on_path,
    parse_simulation_dataframe,
    parse_subset_points,
    parse_x0_feasible_region_excel,
    project_root,
    read_uploaded_table,
    simulation_template_csv_bounds,
    simulation_template_csv_stats,
    subset_template_csv,
    template_filename_bounds,
    template_filename_stats,
    template_filename_subset,
    template_filename_x0_feasible_xlsx,
    x0_feasible_unit_box_template_xlsx_bytes,
)

# Must run before importing plausible_inference
ensure_package_on_path()

from plausible_inference.execution_engine import (  # noqa: E402
    ExecutionConfig,
    run_execution_engine,
)
from plausible_inference.model_construction import model_construction  # noqa: E402
from plausible_inference.utils.solver_config import get_plausibility_solver  # noqa: E402

# ---------------------------------------------------------------------------
# UI constants
# ---------------------------------------------------------------------------

INFERENCE_FORM_OPTIONS = {
    "Screening": "screening",
    "Plausible intervals": "plausible_intervals",
    "Input pixelization": "input_pixelization",
    "Output pixelization": "output_pixelization",
}

# Package discrepancy names (proper terms; keep as shown).
DISCREPANCY_TYPES_STATS = ("norm_infinite", "norm_1", "norm_2")

FUNCTIONAL_PROP_LABELS = list(PROP_LABEL_TO_INTERNAL.keys())

EXECUTION_PARALLEL = "Parallel"
EXECUTION_SEQUENTIAL_COLD = "Sequential cold start"
EXECUTION_SEQUENTIAL_WARM = "Sequential warm start"
EXECUTION_MODES = (
    EXECUTION_PARALLEL,
    EXECUTION_SEQUENTIAL_COLD,
    EXECUTION_SEQUENTIAL_WARM,
)


def _engine_execution(execution_mode: str) -> str:
    """Map a GUI execution-mode label to the engine's execution kind."""
    return "parallel" if execution_mode == EXECUTION_PARALLEL else "sequential"


# Acceptabilities selectable in the UI (package name -> label). feasibility and
# closeness-to-target are per-objective, so they appear for both single and multi-objective.
ACCEPTABILITY_OPTION_LABELS = {
    "single-objective-optimality": "Single-objective optimality (x0 is a minimizer)",
    "delta-optimality": "Delta-optimality (within δ of the best)",
    "feasibility": "Feasibility (per objective: m(x0)[d] ≤ threshold_d)",
    "closeness-to-target": "Closeness to target (per objective: m(x0)[d] in [c_d−δ_d, c_d+δ_d])",
    "Pareto-optimality": "Pareto-optimality (x0 not dominated)",
}
SINGLE_OBJ_ACCEPTABILITY_ORDER = (
    "single-objective-optimality",
    "delta-optimality",
    "feasibility",
    "closeness-to-target",
)
MULTI_OBJ_ACCEPTABILITY_ORDER = (
    "Pareto-optimality",
    "feasibility",
    "closeness-to-target",
)


def _per_objective_number_input(label: str, num_objectives: int, default: float,
                                min_value: Optional[float] = None) -> Any:
    """Collect a per-objective parameter: a scalar for single-objective, else one
    value per objective (returned as a list). Mirrors the package's scalar-or-vector input."""
    extra = {} if min_value is None else {"min_value": min_value}
    if num_objectives == 1:
        return float(st.number_input(label, value=default, step=0.1, **extra))
    return [
        float(st.number_input(f"{label} — objective {d}", value=default, step=0.1,
                              key=f"acc_{label}_{d}", **extra))
        for d in range(1, num_objectives + 1)
    ]


def _acceptability_ui(
    inference_type: str, num_objectives: int
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Streamlit selector for acceptability + its numeric parameters.

    Returns ``(acceptability, acceptability_parameters)``. For plausible_intervals
    acceptability is None (the package forbids it for interval inference). feasibility and
    closeness-to-target collect per-objective threshold / δ (scalar for single-objective,
    one value per objective otherwise).
    """
    if inference_type == "plausible_intervals":
        return None, None

    order = MULTI_OBJ_ACCEPTABILITY_ORDER if num_objectives > 1 else SINGLE_OBJ_ACCEPTABILITY_ORDER
    labels = [ACCEPTABILITY_OPTION_LABELS[k] for k in order]
    chosen_label = st.selectbox("Acceptability", labels)
    acceptability = order[labels.index(chosen_label)]

    params: Dict[str, Any] = {}
    if acceptability == "delta-optimality":
        params["delta"] = float(
            st.number_input(
                "δ (delta) — max gap of m(x0) above the best simulated objective",
                min_value=0.0,
                value=0.5,
                step=0.1,
            )
        )
    elif acceptability == "feasibility":
        params["threshold"] = _per_objective_number_input(
            "Threshold — require objective m(x0) ≤ threshold", num_objectives, default=0.0
        )
    elif acceptability == "closeness-to-target":
        params["threshold"] = _per_objective_number_input(
            "Target c — band center", num_objectives, default=0.0
        )
        params["delta"] = _per_objective_number_input(
            "δ (delta) — half-width of the band [c−δ, c+δ]", num_objectives, default=0.5,
            min_value=0.0,
        )
    return acceptability, (params or None)


def _validate_at_least_one_property_per_objective(fpl: List[List[str]]) -> None:
    for i, fp in enumerate(fpl, start=1):
        if not fp:
            raise ValueError(f"Select at least one functional property for objective {i}.")


def _run_batch_screening_like(
    inference_type: str,
    points: pd.DataFrame,
    s_dim: int,
    M_model,
    solver,
    execution_mode: str,
    pool_processes: int,
) -> List[Dict[str, Any]]:
    """Screening or single-endpoint interval over rows of points (n, s)."""
    X = parse_subset_points(points, s_dim)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if inference_type not in (
        "screening",
        "upper_plausible_interval",
        "lower_plausible_interval",
    ):
        raise ValueError(
            f"Not a screening/single-endpoint interval type: {inference_type} "
            "(use plausible_intervals at the UI layer for both bounds)."
        )

    config = ExecutionConfig(
        inference_type=inference_type,
        execution=_engine_execution(execution_mode),
        warm_start=execution_mode == EXECUTION_SEQUENTIAL_WARM,
        pool_processes=pool_processes,
    )
    df = run_execution_engine(config, M_model, solver, X)
    return df.to_dict("records")


def _merge_plausible_interval_results(
    per_obj: Dict[int, Dict[str, List[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    """One row per candidate idx with upper/lower plausible bounds for every objective.

    ``per_obj`` maps objective index d (1-based) -> {"upper": results, "lower": results},
    each a sweep output for ``m[0, d]``. Bound/note/traceback columns are suffixed ``_obj{d}``.
    """
    indexed: Dict[int, Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]] = {}
    all_idx: set = set()
    for d, sides in per_obj.items():
        by_u = {int(r["idx"]): r for r in sides.get("upper", [])}
        by_l = {int(r["idx"]): r for r in sides.get("lower", [])}
        indexed[d] = (by_u, by_l)
        all_idx |= set(by_u) | set(by_l)

    out: List[Dict[str, Any]] = []
    for i in sorted(all_idx):
        row: Dict[str, Any] = {"idx": i}
        for d in sorted(per_obj):
            by_u, by_l = indexed[d]
            u, lo = by_u.get(i, {}), by_l.get(i, {})
            row[f"plausible_upper_bound_obj{d}"] = u.get("bound")
            row[f"plausible_lower_bound_obj{d}"] = lo.get("bound")
            upper_note = None if u.get("solve_status") == "completed" else u.get("solve_status")
            if upper_note:
                row[f"upper_solve_note_obj{d}"] = upper_note
            lower_note = None if lo.get("solve_status") == "completed" else lo.get("solve_status")
            if lower_note:
                row[f"lower_solve_note_obj{d}"] = lower_note
            if u.get("traceback"):
                row[f"upper_traceback_obj{d}"] = u["traceback"]
            if lo.get("traceback"):
                row[f"lower_traceback_obj{d}"] = lo["traceback"]
        out.append(row)
    return out


def _run_plausible_intervals_merged(
    subset_df: pd.DataFrame,
    s_dim: int,
    exp_set: Any,
    num_objectives: int,
    fpl: List[List[str]],
    kw_mc: Dict[str, Any],
    solver: Any,
    execution_mode: str,
    pool_processes: int,
    status: Any,
) -> List[Dict[str, Any]]:
    """Build upper- and lower-bound models, run both sweeps, merge rows (package types unchanged)."""
    # Run the interval procedure once per objective and per direction. The constraint set
    # (functional properties + confidence region) is identical across all interval models,
    # so the structural feasibility check is run only on the first build to avoid repeating
    # it 2 * num_objectives times.
    per_obj: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    feas_checked = False
    for d in range(1, num_objectives + 1):
        sides: Dict[str, List[Dict[str, Any]]] = {}
        for label, it in (
            ("upper", "upper_plausible_interval"),
            ("lower", "lower_plausible_interval"),
        ):
            kw = dict(kw_mc)
            if feas_checked:
                kw["feasibility_check_solver"] = None
            status.write(f"Building model (objective {d}, {label} bound)…")
            mdl = model_construction(
                exp_set, num_objectives, it, fpl, interval_objective_index=d, **kw
            )
            feas_checked = True
            status.write(f"Solving (objective {d}, {label} bound)…")
            sides[label] = _run_batch_screening_like(
                it,
                subset_df,
                s_dim,
                mdl,
                solver,
                execution_mode,
                pool_processes,
            )
        per_obj[d] = sides
    return _merge_plausible_interval_results(per_obj)


def _run_batch_input_pixelization(
    pixel_lb: Any,
    pixel_ub: Any,
    M_model,
    solver,
    execution_mode: str,
    pool_processes: int,
) -> List[Dict[str, Any]]:
    config = ExecutionConfig(
        inference_type="input_pixelization",
        execution=_engine_execution(execution_mode),
        warm_start=execution_mode == EXECUTION_SEQUENTIAL_WARM,
        pool_processes=pool_processes,
    )
    df = run_execution_engine(config, M_model, solver, (pixel_lb, pixel_ub))
    return df.to_dict("records")


def _run_batch_output_pixelization(
    m0_lb: "Any",
    m0_ub: "Any",
    M_model,
    solver,
    execution_mode: str,
    pool_processes: int,
    time_limit_sec: Optional[float],
) -> List[Dict[str, Any]]:
    config = ExecutionConfig(
        inference_type="output_pixelization",
        execution=_engine_execution(execution_mode),
        warm_start=execution_mode == EXECUTION_SEQUENTIAL_WARM,
        time_limit_sec=time_limit_sec,
        pool_processes=pool_processes,
    )
    df = run_execution_engine(config, M_model, solver, (m0_lb, m0_ub))
    return df.to_dict("records")


def normalize_results(
    inference_type: str,
    results: List[Dict[str, Any]],
    num_objectives: int,
    num_decision_dims: int,
    points: Optional[Any] = None,
    pixel_lb: Optional[Any] = None,
    pixel_ub: Optional[Any] = None,
    m0_lb: Optional[Any] = None,
    m0_ub: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Map every procedure's raw worker results to one uniform, self-describing schema.

    Stable column order: ``idx``, the query geometry as per-dimension columns, the verdict,
    then ``solve_status`` and (when present) ``traceback``. The verdict is the boolean
    ``returned`` for screening / input- / output-pixelization, and ``lower_bound_obj{d}`` /
    ``upper_bound_obj{d}`` for plausible intervals. Geometry is attached by joining the worker
    ``idx`` against the inputs that were submitted (candidate points or pixel bounds).
    """
    s, d = num_decision_dims, num_objectives
    rows: List[Dict[str, Any]] = []

    if inference_type in ("screening", "input_pixelization", "output_pixelization"):
        for r in results:
            i = int(r["idx"])
            row: Dict[str, Any] = {"idx": i}
            if inference_type == "screening" and points is not None and i < len(points):
                for j in range(s):
                    row[f"x0_{j + 1}"] = float(points[i][j])
            elif inference_type == "input_pixelization" and pixel_lb is not None and i < len(pixel_lb):
                for j in range(s):
                    row[f"x_lb_{j + 1}"] = float(pixel_lb[i][j])
                    row[f"x_ub_{j + 1}"] = float(pixel_ub[i][j])
            elif inference_type == "output_pixelization" and m0_lb is not None and i < len(m0_lb):
                for j in range(d):
                    row[f"m_lb_{j + 1}"] = float(m0_lb[i][j])
                    row[f"m_ub_{j + 1}"] = float(m0_ub[i][j])
            row["returned"] = r.get("result") == "returned"
            row["solve_status"] = r.get("solve_status", "completed")
            if r.get("traceback"):
                row["traceback"] = r["traceback"]
            rows.append(row)
        return rows

    if inference_type == "plausible_intervals":
        for r in results:
            i = int(r["idx"])
            row = {"idx": i}
            if points is not None and i < len(points):
                for j in range(s):
                    row[f"x0_{j + 1}"] = float(points[i][j])
            elif "x0" in r:
                for j, v in enumerate(r["x0"]):
                    row[f"x0_{j + 1}"] = float(v)
            notes: List[str] = []
            tbs: List[str] = []
            any_error = any_timeout = False
            for dd in range(1, d + 1):
                lo = r.get(f"plausible_lower_bound_obj{dd}")
                up = r.get(f"plausible_upper_bound_obj{dd}")
                row[f"lower_bound_obj{dd}"] = lo
                row[f"upper_bound_obj{dd}"] = up
                ln = r.get(f"lower_solve_note_obj{dd}")
                un = r.get(f"upper_solve_note_obj{dd}")
                if ln:
                    notes.append(f"obj{dd} lower: {ln}")
                if un:
                    notes.append(f"obj{dd} upper: {un}")
                if r.get(f"lower_traceback_obj{dd}"):
                    tbs.append(str(r[f"lower_traceback_obj{dd}"]))
                if r.get(f"upper_traceback_obj{dd}"):
                    tbs.append(str(r[f"upper_traceback_obj{dd}"]))
                for note in (ln, un):
                    if note == "error":
                        any_error = True
                    elif note == "timeout":
                        any_timeout = True
            row["solve_status"] = "error" if any_error else ("timeout" if any_timeout else "completed")
            if notes:
                row["solve_detail"] = " | ".join(notes)
            if tbs:
                row["traceback"] = "\n---\n".join(tbs)
            rows.append(row)
        return rows

    return list(results)


def main() -> None:
    st.set_page_config(page_title="Plausible Inference", layout="wide")
    st.title("Plausible Inference")
    st.caption("Screening, plausible intervals, and pixelization from your simulation data.")

    root = project_root()
    st.sidebar.markdown(f"**Project root:** `{root}`")
    results_dir = default_results_dir()
    st.sidebar.markdown(f"**Results folder:** `{results_dir}`")

    cpu_max = max(1, os.cpu_count() or 1)

    # ----- Problem type & objectives -----
    st.header("1. Problem type")
    problem_type = st.radio(
        "Objectives",
        ("One objective", "Multiple objectives"),
        horizontal=True,
    )
    if problem_type == "One objective":
        num_objectives = 1
    else:
        num_objectives = st.number_input("Number of objectives", min_value=2, max_value=12, value=2, step=1)
        num_objectives = int(num_objectives)

    num_decision_dims = st.number_input(
        "Number of decision variables (must match simulation columns x_1, x_2, …)",
        min_value=1,
        max_value=32,
        value=2,
        step=1,
    )
    num_decision_dims = int(num_decision_dims)

    # ----- Inference form -----
    st.header("2. Inference type")
    form_label = st.selectbox("Procedure", list(INFERENCE_FORM_OPTIONS.keys()))
    inference_type = INFERENCE_FORM_OPTIONS[form_label]

    acceptability, acceptability_parameters = _acceptability_ui(inference_type, num_objectives)

    # ----- Functional properties (per objective) -----
    st.header("3. Functional properties (per objective)")
    selections: List[List[str]] = []
    lip_constants: List[float] = []
    dir_lip_texts: List[str] = []
    for obj in range(num_objectives):
        with st.expander(f"Objective {obj + 1}", expanded=(obj == 0)):
            sel = st.multiselect(
                f"Properties for objective {obj + 1}",
                FUNCTIONAL_PROP_LABELS,
                default=[],
                key=f"props_{obj}",
            )
            selections.append(sel)
            if "Lipschitz continuity" in sel:
                lip_constants.append(
                    float(
                        st.number_input(
                            f"Lipschitz constant — objective {obj + 1}",
                            min_value=0.0,
                            value=1.0,
                            step=0.1,
                            key=f"lip_{obj}",
                        )
                    )
                )
            else:
                lip_constants.append(0.0)
            if "Directional Lipschitz continuity" in sel:
                dir_lip_texts.append(
                    st.text_input(
                        f"Directional coefficients (comma-separated, {num_decision_dims} values) "
                        f"— objective {obj + 1}",
                        value=",".join(["1.0"] * num_decision_dims),
                        key=f"dlip_{obj}",
                    )
                )
            else:
                dir_lip_texts.append("")

    # ----- Simulation outputs -----
    st.header("4. Simulation outputs")
    st.caption(
        "Templates match the objective count and decision-variable count you set in step 1. "
        "Change those values and download again if you need a different layout."
    )

    data_mode = st.radio(
        "Input mode",
        (
            "Detect from file: use bounds columns if present, otherwise statistics",
            "Statistics only: mean, variance, sample size",
            "Bounds only: lower and upper per objective",
        ),
    )
    force_bounds: Optional[bool] = None
    if "Statistics only" in data_mode:
        force_bounds = False
    elif "Bounds only" in data_mode:
        force_bounds = True

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download simulation template (statistics)",
            simulation_template_csv_stats(num_decision_dims, num_objectives).encode("utf-8"),
            file_name=template_filename_stats(num_objectives, num_decision_dims),
            mime="text/csv",
            key=f"dl_sim_stats_m{num_objectives}_s{num_decision_dims}",
        )
    with c2:
        st.download_button(
            "Download simulation template (bounds)",
            simulation_template_csv_bounds(num_decision_dims, num_objectives).encode("utf-8"),
            file_name=template_filename_bounds(num_objectives, num_decision_dims),
            mime="text/csv",
            key=f"dl_sim_bnd_m{num_objectives}_s{num_decision_dims}",
        )

    sim_file = st.file_uploader("Upload simulation CSV or Excel", type=["csv", "xlsx", "xls"])

    uses_bounds = False
    parsed = None
    confidence_level: Optional[float] = None
    discrepancy_key = "norm_infinite"

    if sim_file is not None:
        try:
            df_sim = read_uploaded_table(sim_file)
            parsed = parse_simulation_dataframe(
                df_sim, num_decision_dims, num_objectives, force_bounds_mode=force_bounds
            )
            uses_bounds = parsed.uses_bounds
            st.success(f"Loaded **{len(df_sim)}** simulated points.")
            if uses_bounds:
                pass
            else:
                confidence_level = st.slider("Confidence level", 0.5, 0.999, 0.95, 0.001)
                discrepancy_key = st.selectbox(
                    "Discrepancy type",
                    DISCREPANCY_TYPES_STATS,
                )
        except Exception as e:
            st.error(f"Failed to parse simulation file: {e}")
            parsed = None

    # ----- Extra inputs per inference form -----
    st.header("5. Procedure-specific inputs")
    subset_df: Optional[pd.DataFrame] = None
    pixel_lb = pixel_ub = None
    m0_lb = m0_ub = None
    x0_feas_A: Optional[Any] = None
    x0_feas_b: Optional[Any] = None
    use_x0_feas = False

    if inference_type in ("screening", "plausible_intervals"):
        st.markdown(
            "Upload a table with one row per candidate point and one column per decision variable "
            "(x_1, x_2, …, same naming as in the subset template)."
        )
        st.download_button(
            "Download subset template",
            subset_template_csv(num_decision_dims).encode("utf-8"),
            file_name=template_filename_subset(num_decision_dims),
            mime="text/csv",
            key=f"dl_subset_s{num_decision_dims}",
        )
        sub_file = st.file_uploader("Subset / candidate points file", type=["csv", "xlsx", "xls"])
        if sub_file is not None:
            try:
                subset_df = read_uploaded_table(sub_file)
                st.success(f"Subset file loaded: **{len(subset_df)}** rows.")
            except Exception as e:
                st.error(str(e))

    elif inference_type == "input_pixelization":
        st.markdown("Partition the decision space with an interval grid (one range per decision variable).")
        bounds = []
        parts = []
        for j in range(num_decision_dims):
            c1, c2, c3 = st.columns(3)
            with c1:
                lo = float(st.number_input(f"Dim {j+1} min", value=0.0, key=f"iplo_{j}"))
            with c2:
                hi = float(st.number_input(f"Dim {j+1} max", value=1.0, key=f"iphi_{j}"))
            with c3:
                npt = int(st.number_input(f"Dim {j+1} partitions", min_value=1, value=5, key=f"ipn_{j}"))
            bounds.append((lo, hi))
            parts.append(npt)
        try:
            pixel_lb, pixel_ub = build_input_pixel_grid(bounds, parts)
            st.caption(f"Generated **{pixel_lb.shape[0]}** input pixels (boxes).")
        except Exception as e:
            st.error(str(e))

    elif inference_type == "output_pixelization":
        st.markdown(
            "Partition predicted objective values at the reference solution: one interval range per objective."
        )
        bounds_m = []
        parts_m = []
        for d in range(num_objectives):
            c1, c2, c3 = st.columns(3)
            with c1:
                lo = float(
                    st.number_input(
                        f"Objective {d + 1} range minimum",
                        value=0.0,
                        key=f"oplo_{d}",
                    )
                )
            with c2:
                hi = float(
                    st.number_input(
                        f"Objective {d + 1} range maximum",
                        value=1.0,
                        key=f"ophi_{d}",
                    )
                )
            with c3:
                npt = int(st.number_input(f"Objective {d+1} partitions", min_value=1, value=5, key=f"opn_{d}"))
            bounds_m.append((lo, hi))
            parts_m.append(npt)
        try:
            m0_lb, m0_ub = build_output_pixel_grid(bounds_m, parts_m)
            st.caption(f"Generated **{m0_lb.shape[0]}** output pixels.")
        except Exception as e:
            st.error(str(e))

    if inference_type in ("input_pixelization", "output_pixelization"):
        st.subheader("Reference solution feasible region (optional)")
        st.markdown(
            "Optional linear constraints on the reference decision vector: each row of matrix **A** times "
            "the vector is less than or equal to the matching entry in **b**. Use the same number of "
            "decision variables as in your simulation file."
        )
        use_x0_feas = st.checkbox("Upload Excel with sheets A and b", value=False)
        st.caption(
            f"Example unit box from 0 to 1 in each dimension: download the template for your current "
            f"number of decision variables ({num_decision_dims})."
        )
        try:
            _x0_xlsx = x0_feasible_unit_box_template_xlsx_bytes(num_decision_dims)
            st.download_button(
                "Download feasible region template (Excel)",
                data=_x0_xlsx,
                file_name=template_filename_x0_feasible_xlsx(num_decision_dims),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dl_x0_xlsx_s{num_decision_dims}",
            )
        except ImportError as ie:
            st.warning(f"{ie} Install with: pip install openpyxl")
        feas_file = st.file_uploader(
            "Excel: sheet A (one row per constraint), sheet b (one value per constraint)",
            type=["xlsx", "xls"],
            disabled=not use_x0_feas,
            key="x0_feas_excel",
        )
        if use_x0_feas and feas_file is not None:
            try:
                raw = feas_file.read()
                x0_feas_A, x0_feas_b = parse_x0_feasible_region_excel(raw, num_decision_dims)
                st.success("Feasible region loaded.")
            except Exception as e:
                st.error(f"Feasible region Excel: {e}")
                x0_feas_A, x0_feas_b = None, None

    # ----- Execution & solver -----
    st.header("6. Execution settings")
    execution_mode = st.selectbox("Execution mode", EXECUTION_MODES, index=0)
    pool_processes = st.number_input(
        f"Parallel workers (used in parallel mode only; maximum {cpu_max})",
        min_value=1,
        max_value=cpu_max,
        value=max(1, cpu_max // 2),
        step=1,
        help=f"At most {cpu_max} processes on this machine.",
    )
    pool_processes = int(min(max(1, pool_processes), cpu_max))
    solver_name = st.selectbox("Solver", ("gurobi", "scip"), index=0)
    skip_feas_check = st.checkbox(
        "Skip structural feasibility check",
        value=False,
    )
    output_time_limit = None
    if inference_type == "output_pixelization":
        output_time_limit = st.number_input(
            "Per-pixel time limit in seconds (0 means none)",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )
        if output_time_limit <= 0:
            output_time_limit = None

    # ----- Run -----
    st.header("7. Run inference")
    run_clicked = st.button("Run inference", type="primary")

    if not run_clicked:
        st.stop()

    errors: List[str] = []
    if parsed is None:
        errors.append("Upload and successfully parse a simulation file.")
    if inference_type in ("screening", "plausible_intervals"):
        if subset_df is None:
            errors.append("Upload a subset / candidate-point file for this procedure.")
    elif inference_type == "input_pixelization":
        if pixel_lb is None:
            errors.append("Define input pixelization bounds.")
    elif inference_type == "output_pixelization":
        if m0_lb is None:
            errors.append("Define output pixelization bounds.")

    if inference_type in ("input_pixelization", "output_pixelization") and use_x0_feas:
        if x0_feas_A is None or x0_feas_b is None:
            errors.append(
                "Upload a valid feasible-region Excel file (sheets A and b), or turn off that option."
            )

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    assert parsed is not None

    try:
        fpl = build_functional_properties_list(selections)
        _validate_at_least_one_property_per_objective(fpl)
        fpp = build_functional_properties_parameters(
            num_objectives, fpl, lip_constants, dir_lip_texts, num_decision_dims
        )
    except Exception as e:
        st.error(f"Functional property configuration invalid: {e}")
        st.stop()

    if uses_bounds:
        disc_type = "confidence_region"
        conf_level = None
        sm = sv = ss = None
        ucb = parsed.upper_confidence_bounds
        lcb = parsed.lower_confidence_bounds
    else:
        disc_type = discrepancy_key
        conf_level = float(confidence_level) if confidence_level is not None else 0.95
        sm, sv, ss = parsed.sample_mean, parsed.sample_var, parsed.sample_size
        ucb = lcb = None

    feas_solver = None if skip_feas_check else solver_name

    meta = {
        "inference_type": inference_type,
        "interval_package_inference_types": (
            ["upper_plausible_interval", "lower_plausible_interval"]
            if inference_type == "plausible_intervals"
            else None
        ),
        "num_objectives": num_objectives,
        "num_decision_dims": num_decision_dims,
        "acceptability": acceptability,
        "acceptability_parameters": acceptability_parameters,
        "functional_properties_list": fpl,
        "discrepancy_type": disc_type,
        "confidence_level": conf_level,
        "uses_bounds_for_confidence": uses_bounds,
        "execution_mode": execution_mode,
        "solver": solver_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "x0_feasible_region_excel": bool(
            inference_type in ("input_pixelization", "output_pixelization")
            and use_x0_feas
            and x0_feas_A is not None
        ),
    }

    with st.status("Running inference…", expanded=True) as status:
        try:
            kw_mc: Dict[str, Any] = dict(
                acceptability=acceptability,
                acceptability_parameters=acceptability_parameters,
                discrepancy_type=disc_type,
                confidence_level=conf_level,
                sample_mean=sm,
                sample_var=sv,
                sample_size=ss,
                functional_properties_parameters=fpp,
                upper_confidence_bounds=ucb,
                lower_confidence_bounds=lcb,
                feasibility_check_solver=feas_solver,
            )
            if (
                inference_type in ("input_pixelization", "output_pixelization")
                and x0_feas_A is not None
                and x0_feas_b is not None
            ):
                kw_mc["x0_feasible_region_A_ineq"] = x0_feas_A
                kw_mc["x0_feasible_region_b_ineq"] = x0_feas_b

            status.write("Creating solver…")
            solver = get_plausibility_solver(solver_name)

            if inference_type == "plausible_intervals":
                assert subset_df is not None
                results = _run_plausible_intervals_merged(
                    subset_df,
                    num_decision_dims,
                    parsed.exp_set,
                    num_objectives,
                    fpl,
                    kw_mc,
                    solver,
                    execution_mode,
                    int(pool_processes),
                    status,
                )
            else:
                status.write("Building optimization model…")
                M_model = model_construction(
                    parsed.exp_set,
                    num_objectives,
                    inference_type,
                    fpl,
                    **kw_mc,
                )
                status.write("Solving batch…")
                if inference_type == "screening":
                    assert subset_df is not None
                    results = _run_batch_screening_like(
                        inference_type,
                        subset_df,
                        num_decision_dims,
                        M_model,
                        solver,
                        execution_mode,
                        int(pool_processes),
                    )
                elif inference_type == "input_pixelization":
                    results = _run_batch_input_pixelization(
                        pixel_lb, pixel_ub, M_model, solver, execution_mode, int(pool_processes)
                    )
                else:
                    results = _run_batch_output_pixelization(
                        m0_lb,
                        m0_ub,
                        M_model,
                        solver,
                        execution_mode,
                        int(pool_processes),
                        output_time_limit,
                    )

            status.update(label="Done.", state="complete")
        except Exception as e:
            status.update(label="Failed.", state="error")
            st.error(f"{type(e).__name__}: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())
            st.stop()

    if inference_type == "plausible_intervals":
        st.success(f"Completed **{len(results)}** candidate rows (upper and lower bounds).")
    else:
        st.success(f"Completed **{len(results)}** solves.")

    if inference_type in ("screening", "plausible_intervals"):
        assert subset_df is not None
        _pts = parse_subset_points(subset_df, num_decision_dims)
        if _pts.ndim == 1:
            _pts = _pts.reshape(1, -1)
    else:
        _pts = None
    norm_results = normalize_results(
        inference_type,
        results,
        num_objectives=num_objectives,
        num_decision_dims=num_decision_dims,
        points=_pts,
        pixel_lb=pixel_lb,
        pixel_ub=pixel_ub,
        m0_lb=m0_lb,
        m0_ub=m0_ub,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = f"pi_inference_{inference_type}_{ts}"
    json_path = results_dir / f"{base}.json"
    csv_path = results_dir / f"{base}_results.csv"

    out_payload = {"metadata": meta, "results": norm_results}
    json_path.write_text(json.dumps(out_payload, indent=2, default=str), encoding="utf-8")

    res_df = pd.json_normalize(norm_results)
    res_df.to_csv(csv_path, index=False)

    st.subheader("Output files")
    st.text(str(json_path))
    st.text(str(csv_path))

    with open(json_path, "rb") as f:
        st.download_button(
            "Download JSON",
            f.read(),
            file_name=json_path.name,
            mime="application/json",
        )
    with open(csv_path, "rb") as f:
        st.download_button(
            "Download CSV (tabular results)",
            f.read(),
            file_name=csv_path.name,
            mime="text/csv",
        )

    if inference_type == "plausible_intervals":
        st.caption(
            "Columns **lower_bound_obj{d}** / **upper_bound_obj{d}** are the interval endpoints for each "
            "objective d at the reference solution; **x0_*** give its coordinates. An endpoint is a finite "
            "number when the solve is optimal, or +/-inf when it is unbounded in that direction "
            "(a normal result). **solve_status** is 'completed' / 'timeout' / 'error'; **solve_detail** "
            "explains any side that timed out or errored."
        )
    else:
        st.caption(
            "Column **returned** flags plausibly-acceptable rows; the geometry columns "
            "(**x0_***, or **x_lb_*/x_ub_*** for input pixels, **m_lb_*/m_ub_*** for output pixels) "
            "identify each row. **solve_status** is 'completed' / 'timeout' / 'error'."
        )
    st.dataframe(res_df.head(50), use_container_width=True)
    if len(res_df) > 50:
        st.caption("Showing first 50 rows.")


if __name__ == "__main__":
    main()
