"""
Thin helpers for the Streamlit GUI: path setup, CSV/DataFrame parsing, templates,
and building arguments for plausible_inference (no inference logic here).
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Repository / import path (package lives under src/plausible_inference)
# ---------------------------------------------------------------------------


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def seeded_data_dir() -> Path:
    """Bundled example CSV/XLSX for the GUI (under ``GUI/seeded_data``)."""
    return Path(__file__).resolve().parent / "seeded_data"


def ensure_package_on_path() -> Path:
    """Insert ``src`` on sys.path so ``import plausible_inference`` works when not installed."""
    root = project_root()
    src = root / "src"
    s = str(src)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


# ---------------------------------------------------------------------------
# Templates (CSV)
# ---------------------------------------------------------------------------

def simulation_template_csv_stats(
    num_decision_dims: int,
    num_objectives: int,
    num_example_rows: int = 3,
) -> str:
    """Statistics-path CSV template (means, variances, sample sizes per objective)."""
    xcols = [f"x_{j}" for j in range(1, num_decision_dims + 1)]
    means = [f"mean_{d}" for d in range(1, num_objectives + 1)]
    vars_ = [f"var_{d}" for d in range(1, num_objectives + 1)]
    header = ["solution_id"] + xcols + ["sample_size"] + means + vars_
    lines = [
        f"# Simulation output (statistics): {num_objectives} objective(s), "
        f"{num_decision_dims} decision variable(s). Remove lines starting with # before upload.",
        ",".join(header),
    ]
    n = max(1, int(num_example_rows))
    for r in range(n):
        sid = f"s{r + 1}"
        xs = [f"{0.1 * ((r + j) % 7 + 1):.2f}" for j in range(num_decision_dims)]
        n_rep = str(80 + r * 5)
        mus = [f"{0.15 * ((r + d) % 5):.2f}" for d in range(num_objectives)]
        vs = [f"{0.1 + 0.05 * ((r + d) % 4):.2f}" for d in range(num_objectives)]
        lines.append(",".join([sid] + xs + [n_rep] + mus + vs))
    return "\n".join(lines) + "\n"


def simulation_template_csv_bounds(
    num_decision_dims: int,
    num_objectives: int,
    num_example_rows: int = 3,
) -> str:
    """Bounds-path CSV template (lower/upper per objective)."""
    xcols = [f"x_{j}" for j in range(1, num_decision_dims + 1)]
    lbs = [f"lower_{d}" for d in range(1, num_objectives + 1)]
    ubs = [f"upper_{d}" for d in range(1, num_objectives + 1)]
    header = ["solution_id"] + xcols + lbs + ubs
    lines = [
        f"# Bounds path: {num_objectives} objective(s), {num_decision_dims} decision variable(s). "
        "No sample size / mean / variance columns. Remove # lines before upload.",
        ",".join(header),
    ]
    n = max(1, int(num_example_rows))
    for r in range(n):
        sid = f"s{r + 1}"
        xs = [f"{0.1 * ((r + j) % 7 + 1):.2f}" for j in range(num_decision_dims)]
        lo = [f"{-0.05 * (1 + (r + d) % 4):.2f}" for d in range(num_objectives)]
        hi = [f"{0.45 + 0.08 * ((r + d) % 5):.2f}" for d in range(num_objectives)]
        lines.append(",".join([sid] + xs + lo + hi))
    return "\n".join(lines) + "\n"


def subset_template_csv(num_decision_dims: int, num_example_rows: int = 3) -> str:
    """Candidate-point CSV template (screening / plausible intervals)."""
    xcols = [f"x_{j}" for j in range(1, num_decision_dims + 1)]
    header = ["row_id"] + xcols
    lines = [
        "# One row per candidate point. Column count matches the number of decision variables.",
        ",".join(header),
    ]
    n = max(1, int(num_example_rows))
    for i in range(n):
        # simple distinct example coordinates
        xs = [f"{(i + j * 0.17) % 1.0:.3f}" for j in range(num_decision_dims)]
        lines.append(",".join([str(i)] + xs))
    return "\n".join(lines) + "\n"


def unit_box_A_b(num_decision_dims: int) -> Tuple[np.ndarray, np.ndarray]:
    """Half-space description of [0, 1]^s: -x_j <= 0 and x_j <= 1 for each j."""
    s = int(num_decision_dims)
    if s < 1:
        raise ValueError("num_decision_dims must be >= 1")
    rows_a = []
    rows_b = []
    for j in range(s):
        row = [0.0] * s
        row[j] = -1.0
        rows_a.append(row)
        rows_b.append(0.0)
    for j in range(s):
        row = [0.0] * s
        row[j] = 1.0
        rows_a.append(row)
        rows_b.append(1.0)
    return np.asarray(rows_a, dtype=float), np.asarray(rows_b, dtype=float)


def x0_feasible_unit_box_template_xlsx_bytes(num_decision_dims: int) -> bytes:
    """
    Excel workbook (sheets ``A``, ``b``) for the unit box [0,1]^s, matching current **s**.
    Requires **openpyxl** (declared in ``[gui]`` extras).
    """
    try:
        from openpyxl import Workbook
    except ImportError as e:
        raise ImportError(
            "Generating the x0 feasible-region .xlsx template requires openpyxl. "
            "Install with: pip install openpyxl"
        ) from e

    s = int(num_decision_dims)
    A, bvec = unit_box_A_b(s)
    wb = Workbook()
    ws_a = wb.active
    ws_a.title = "A"
    ws_a.append([f"x_{j}" for j in range(1, s + 1)])
    for row in A:
        ws_a.append([float(x) for x in row])
    ws_b = wb.create_sheet("b")
    ws_b.append(["b"])
    for v in bvec:
        ws_b.append([float(v)])
    bio = io.BytesIO()
    wb.save(bio)
    return bio.getvalue()


def template_filename_stats(num_objectives: int, num_decision_dims: int) -> str:
    return f"pi_simulation_template_stats_m{num_objectives}_s{num_decision_dims}.csv"


def template_filename_bounds(num_objectives: int, num_decision_dims: int) -> str:
    return f"pi_simulation_template_bounds_m{num_objectives}_s{num_decision_dims}.csv"


def template_filename_subset(num_decision_dims: int) -> str:
    return f"pi_subset_template_s{num_decision_dims}.csv"


def template_filename_x0_feasible_xlsx(num_decision_dims: int) -> str:
    return f"pi_x0_feasible_unit_box_s{num_decision_dims}.xlsx"


# ---------------------------------------------------------------------------
# Parsing uploaded tables
# ---------------------------------------------------------------------------


def _strip_comment_lines(text: str) -> str:
    out_lines = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    """Read CSV or Excel from Streamlit UploadedFile."""
    name = (uploaded_file.name or "").lower()
    raw = uploaded_file.read()
    bio = io.BytesIO(raw)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(bio)
    text = raw.decode("utf-8-sig", errors="replace")
    text = _strip_comment_lines(text)
    return pd.read_csv(io.StringIO(text))


@dataclass
class ParsedSimulation:
    exp_set: np.ndarray  # (k, s)
    sample_mean: Optional[np.ndarray]
    sample_var: Optional[np.ndarray]
    sample_size: Optional[np.ndarray]
    upper_confidence_bounds: Optional[np.ndarray]
    lower_confidence_bounds: Optional[np.ndarray]
    uses_bounds: bool


def _find_x_columns(df: pd.DataFrame, s: int) -> List[str]:
    candidates = []
    for j in range(1, s + 1):
        for name in (f"x_{j}", f"x{j}", f"X_{j}", f"X{j}"):
            if name in df.columns:
                candidates.append(name)
                break
        else:
            raise ValueError(f"Missing decision column for dimension {j}: expected x_{j} (or x{j})")
    return candidates


def _has_bounds_columns(df: pd.DataFrame, d: int) -> bool:
    for obj in range(1, d + 1):
        if not (
            (f"lower_{obj}" in df.columns or f"lb_{obj}" in df.columns)
            and (f"upper_{obj}" in df.columns or f"ub_{obj}" in df.columns)
        ):
            return False
    return True


def parse_simulation_dataframe(
    df: pd.DataFrame,
    num_decision_dims: int,
    num_objectives: int,
    force_bounds_mode: Optional[bool] = None,
) -> ParsedSimulation:
    """
    Build exp_set and either (mean, var, n) or (lower, upper) arrays.
    If force_bounds_mode is True/False, enforce that path; if None, infer from columns.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    k = len(df)
    if k == 0:
        raise ValueError("Simulation table is empty")

    xcols = _find_x_columns(df, num_decision_dims)
    exp_set = df[xcols].to_numpy(dtype=float)

    has_bounds = _has_bounds_columns(df, num_objectives)
    if force_bounds_mode is True and not has_bounds:
        raise ValueError("Bounds mode selected but lower_*/upper_* (or lb_*/ub_*) columns are missing")
    if force_bounds_mode is False and has_bounds:
        # user explicitly chose statistics — still allow if they also have stats columns
        pass

    uses_bounds = has_bounds if force_bounds_mode is None else bool(force_bounds_mode)

    if uses_bounds:
        lb = np.zeros((k, num_objectives))
        ub = np.zeros((k, num_objectives))
        for obj in range(1, num_objectives + 1):
            lname = f"lower_{obj}" if f"lower_{obj}" in df.columns else f"lb_{obj}"
            uname = f"upper_{obj}" if f"upper_{obj}" in df.columns else f"ub_{obj}"
            lb[:, obj - 1] = df[lname].to_numpy(dtype=float)
            ub[:, obj - 1] = df[uname].to_numpy(dtype=float)
        return ParsedSimulation(
            exp_set=exp_set,
            sample_mean=None,
            sample_var=None,
            sample_size=None,
            upper_confidence_bounds=ub,
            lower_confidence_bounds=lb,
            uses_bounds=True,
        )

    # Statistics path
    if "sample_size" not in df.columns and "n" not in df.columns:
        raise ValueError("Expected column 'sample_size' (or 'n') when not using bounds columns")
    n_col = "sample_size" if "sample_size" in df.columns else "n"
    sample_size_1d = df[n_col].to_numpy(dtype=float)

    mean_cols = []
    var_cols = []
    for obj in range(1, num_objectives + 1):
        m_candidates = (f"mean_{obj}", f"sample_mean_{obj}", f"m_{obj}")
        v_candidates = (f"var_{obj}", f"sample_var_{obj}", f"v_{obj}")
        mc = next((c for c in m_candidates if c in df.columns), None)
        vc = next((c for c in v_candidates if c in df.columns), None)
        if mc is None or vc is None:
            raise ValueError(f"Missing mean/var columns for objective {obj} (e.g. mean_{obj}, var_{obj})")
        mean_cols.append(mc)
        var_cols.append(vc)

    sample_mean = df[mean_cols].to_numpy(dtype=float)
    sample_var = df[var_cols].to_numpy(dtype=float)
    if sample_size_1d.shape == (k,):
        sample_size = np.broadcast_to(sample_size_1d.reshape(k, 1), (k, num_objectives)).copy()
    else:
        sample_size = np.asarray(sample_size_1d, dtype=float)
        if sample_size.shape not in ((k, num_objectives), (k, 1), (k,)):
            raise ValueError(f"sample_size column has incompatible shape {sample_size.shape}")

    return ParsedSimulation(
        exp_set=exp_set,
        sample_mean=sample_mean,
        sample_var=sample_var,
        sample_size=sample_size,
        upper_confidence_bounds=None,
        lower_confidence_bounds=None,
        uses_bounds=False,
    )


def parse_subset_points(df: pd.DataFrame, num_decision_dims: int) -> np.ndarray:
    """Return array shape (n, s) for screening / interval grid."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    xcols = _find_x_columns(df, num_decision_dims)
    arr = df[xcols].to_numpy(dtype=float)
    if arr.ndim != 2 or arr.shape[1] != num_decision_dims:
        raise ValueError(f"Subset table must have shape (n, {num_decision_dims})")
    return arr


# ---------------------------------------------------------------------------
# Functional properties → package types
# ---------------------------------------------------------------------------

PROP_LABEL_TO_INTERNAL = {
    "Convexity": "convexity",
    "Concavity": "concavity",
    # "Linearity": "linearity",  # dormant: hidden from the GUI; uncomment to re-enable
    "Lipschitz continuity": "Lipschitz_continuity",
    "Directional Lipschitz continuity": "directional_Lipschitz_continuity",
}


def build_functional_properties_list(
    selections_per_objective: Sequence[Sequence[str]],
) -> List[List[str]]:
    out: List[List[str]] = []
    for sel in selections_per_objective:
        internal = [PROP_LABEL_TO_INTERNAL[x] for x in sel]
        out.append(internal)
    return out


def build_functional_properties_parameters(
    num_objectives: int,
    functional_properties_list: List[List[str]],
    lip_constants: Sequence[float],
    directional_vectors_text: Sequence[str],
    decision_dim: int,
) -> Optional[List[Dict[str, Any]]]:
    """Returns None if no parameterised property is used; else list of dicts (one per objective)."""
    needs = any(
        "Lipschitz_continuity" in fp or "directional_Lipschitz_continuity" in fp
        for fp in functional_properties_list
    )
    if not needs:
        return None
    params: List[Dict[str, Any]] = []
    for d in range(num_objectives):
        fp = functional_properties_list[d]
        entry: Dict[str, Any] = {}
        if "Lipschitz_continuity" in fp:
            entry["lip_CST"] = float(lip_constants[d])
        if "directional_Lipschitz_continuity" in fp:
            vec = _parse_float_vector(directional_vectors_text[d], decision_dim)
            entry["lip_CST_vector"] = vec
        params.append(entry)
    return params


def _parse_float_vector(text: str, expected_len: int) -> List[float]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Directional Lipschitz requires comma-separated coefficients")
    parts = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(parts) == 1 and expected_len > 1:
        parts = parts * expected_len
    if len(parts) != expected_len:
        raise ValueError(f"Directional Lipschitz vector length {len(parts)} != decision dimension {expected_len}")
    return parts


# ---------------------------------------------------------------------------
# x0 feasible region: A @ x0 <= b (input / output pixelization only in the package)
# ---------------------------------------------------------------------------


def _pick_excel_sheet(xl: pd.ExcelFile, candidates: Sequence[str]) -> str:
    lower_map = {n.lower(): n for n in xl.sheet_names}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise ValueError(
        f"Expected one of sheets {list(candidates)}; workbook has {list(xl.sheet_names)}"
    )


def parse_x0_feasible_region_excel(raw: bytes, num_decision_dims: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read matrix A (n_ineq × s) and vector b (n_ineq,) from an Excel workbook.

    Sheets (case-insensitive): ``A`` / ``A_matrix`` / ``A_ineq`` for A;
    ``b`` / ``b_ineq`` / ``rhs`` for b.

    If columns ``x_1`` … ``x_s`` exist on sheet A, they are used; otherwise all numeric
    columns are taken and must count exactly ``s``.
    """
    bio = io.BytesIO(raw)
    xl = pd.ExcelFile(bio)
    sheet_a = _pick_excel_sheet(xl, ("A", "a", "A_matrix", "A_ineq"))
    sheet_b = _pick_excel_sheet(xl, ("b", "B", "b_ineq", "rhs"))

    df_a = pd.read_excel(xl, sheet_name=sheet_a)
    df_a.columns = [str(c).strip() for c in df_a.columns]
    xnames = [f"x_{j}" for j in range(1, num_decision_dims + 1)]
    if all(c in df_a.columns for c in xnames):
        A = df_a[xnames].to_numpy(dtype=float)
    else:
        num = df_a.select_dtypes(include=[np.number])
        if num.shape[1] != num_decision_dims:
            raise ValueError(
                f"Sheet A must have {num_decision_dims} numeric columns (or columns x_1…x_{num_decision_dims}); "
                f"got {num.shape[1]} numeric columns"
            )
        A = num.to_numpy(dtype=float)

    df_b = pd.read_excel(xl, sheet_name=sheet_b)
    df_b.columns = [str(c).strip() for c in df_b.columns]
    col_b = None
    for c in df_b.columns:
        if c.lower() == "b":
            col_b = c
            break
    if col_b is not None:
        b = df_b[col_b].to_numpy(dtype=float).ravel()
    else:
        b = df_b.iloc[:, 0].to_numpy(dtype=float).ravel()

    n = A.shape[0]
    if b.shape[0] != n:
        raise ValueError(f"Length of b ({b.shape[0]}) must equal number of rows of A ({n})")
    return A, b


# ---------------------------------------------------------------------------
# Pixel grids
# ---------------------------------------------------------------------------


def build_input_pixel_grid(
    bounds_per_dim: Sequence[Tuple[float, float]],
    partitions_per_dim: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cartesian product of intervals [edge[i], edge[i+1]] per decision dimension.
    bounds_per_dim: list of (min, max) length s
    partitions_per_dim: list of int, same length
    Returns pixel_lb, pixel_ub each (n_pixels, s)
    """
    s = len(bounds_per_dim)
    if len(partitions_per_dim) != s:
        raise ValueError("partitions_per_dim must match number of decision dimensions")

    edges_per_dim = []
    for (lo, hi), n in zip(bounds_per_dim, partitions_per_dim):
        if n < 1:
            raise ValueError("Number of partitions per dimension must be >= 1")
        edges_per_dim.append(np.linspace(float(lo), float(hi), int(n) + 1))

    pixels_lb = []
    pixels_ub = []
    for multi_idx in product(*[range(len(e) - 1) for e in edges_per_dim]):
        lb = [edges_per_dim[dim][multi_idx[dim]] for dim in range(s)]
        ub = [edges_per_dim[dim][multi_idx[dim] + 1] for dim in range(s)]
        pixels_lb.append(lb)
        pixels_ub.append(ub)
    return np.asarray(pixels_lb, dtype=float), np.asarray(pixels_ub, dtype=float)


def build_output_pixel_grid(
    bounds_per_objective: Sequence[Tuple[float, float]],
    partitions_per_objective: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Same as input grid but for m[0, :] space; d = num_objectives."""
    return build_input_pixel_grid(bounds_per_objective, partitions_per_objective)


def default_results_dir() -> Path:
    d = project_root() / "GUI" / "inference_results"
    d.mkdir(parents=True, exist_ok=True)
    return d
