"""
Execution engine: single-point Pyomo solves, batch helpers, and the unified entry point.

Public API (the only names intended for external use):

- ``ExecutionConfig``: dataclass carrying all sweep settings (inference_type,
  execution, warm_start, indices, time_limit_sec, pool_processes).
- ``run_execution_engine(config, model, solver, data) -> pandas.DataFrame``:
  dispatches to the private workers below and returns a uniform result table
  with columns ``idx``, ``solve_status`` (``completed`` / ``timeout`` /
  ``error``), ``result`` (``returned`` / ``screened_out``, feasibility types),
  ``bound`` (interval types), and ``traceback`` (error rows only).

Private implementation (prefixed with ``_``):

- ``_*_transfer``: fixed-length tuples for ``multiprocessing.Pool.map``; screening /
  pixelization use cold ``load_solutions=False``. Plausible-interval workers still
  call ``solve(..., load_solutions=True)`` so ``OBJ`` is defined when optimal.
- ``_*_sequential``: single-process sweeps on one model; optional ``solve_mode`` enables
  chained ``load_solutions=True`` (no ``warmstart=`` kw).
- ``_*_parallel_tasks`` / ``_run_*_parallel`` / ``_run_*_sequential_*``: build row tasks and
  validate shape and indices before solving.
- ``_run_*_sequential_warm``: uses ``warm_next`` only if the **previous** row was successful
  (screening/input/output: ``feasible``; interval: ``bound`` in result); otherwise the next
  solve is **cold** (no chained primal).
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import os
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from .utils.solver_config import set_solver_time_limit


@contextmanager
def _quiet_pyomo_solver_load_warnings() -> Iterator[None]:
    """Hide Pyomo WARNING when loading SolverResults with non-ok status (e.g. infeasible).

    Expected when ``load_solutions=True``; screening still reads termination from ``results``.
    """
    names = ("pyomo", "pyomo.core", "pyomo.opt", "pyomo.solvers", "pyutilib")
    loggers = [logging.getLogger(n) for n in names]
    previous = [(lg, lg.level) for lg in loggers]
    try:
        for lg in loggers:
            lg.setLevel(logging.ERROR)
        yield
    finally:
        for lg, lvl in previous:
            lg.setLevel(lvl)


def _solve_with_optional_quiet(solver, model, **solve_kw):
    """Call ``solver.solve``; suppress routine load-solution warnings when ``load_solutions`` is true."""
    load_solutions = bool(solve_kw.get("load_solutions", False))
    ctx = (
        _quiet_pyomo_solver_load_warnings()
        if load_solutions
        else contextlib.nullcontext()
    )
    with ctx:
        return solver.solve(model, **solve_kw)

# Tuple contracts (wrong length -> TypeError at unpack or explicit check)
# _screening_x0_transfer: (idx, x0_row, model, solver)
# _screening_x0_sequential: same, or append solve_mode
# _input_pixelization_pixel_transfer: (idx, pixel_lb, pixel_ub, model, solver)
# _output_pixelization_pixel_transfer: 5-tuple no limit, 6-tuple with time_limit_sec
# _output_pixelization_pixel_sequential: 5/6 cold; 6-tuple with last str/bool = solve_mode;
#   7-tuple = (..., time_limit_sec, solve_mode)
# _interval_x0_transfer / _interval_x0_sequential: same 4 / 5 pattern as screening


def _normalize_screening_solve_mode(solve_mode):
    """Map legacy True/False (notebook warm chain) to string modes."""
    if solve_mode is True:
        return "warm_next"
    if solve_mode is False:
        return "warm_first"
    return solve_mode


def _screening_load_solutions(solve_mode: str) -> bool:
    """Only ``\"cold\"`` skips loading solutions; otherwise ``load_solutions=True``."""
    return solve_mode != "cold"


def _as_2d(a, name: str) -> np.ndarray:
    """Require a 2-D *array layout* for batch helpers (not "problem dimension is 2").

    - Axis 0: number of sweep rows (grid points / pixels), length ``n``.
    - Axis 1: length of the vector *for one row* (e.g. screening ``s``, output pixel ``d``).
    - Row ``arr[i]`` is passed to the single-point solvers as one ``x0`` or bound vector.
    - A single coordinate per row still uses shape ``(n, 1)``, not ``(n,)``, so indexing
      and tuple building stay consistent with ``s > 1``.
    """
    arr = np.asarray(a)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D with shape (n_points, n_components_per_point), got {arr.shape}"
        )
    return arr


def _row_indices(n_rows: int, indices: Optional[Iterable[int]]) -> Tuple[int, ...]:
    """Validate indices into axis-0 of the tables validated by ``_as_2d``."""
    if indices is None:
        return tuple(range(n_rows))
    idx = tuple(int(i) for i in indices)
    for i in idx:
        if i < 0 or i >= n_rows:
            raise IndexError(f"row index {i} out of range [0, {n_rows})")
    return idx


def _screening_run(idx, current_x0_array, M_model, solver, load_solutions: bool):
    s_val = len(current_x0_array)
    x0_dict = {i + 1: float(current_x0_array[i]) for i in range(s_val)}
    M_model.x0_param.store_values(x0_dict)
    results = _solve_with_optional_quiet(
        solver, M_model, tee=False, load_solutions=load_solutions
    )
    status = results.solver.status
    tc = results.solver.termination_condition
    feasible = status == pyo.SolverStatus.ok and tc in (
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible,
    )
    return {"idx": idx, "feasible": feasible}


def _screening_x0_transfer(args):
    """Parallel worker: 4-tuple, cold start every call."""
    idx, current_x0_array, M_model, solver = args
    try:
        return _screening_run(idx, current_x0_array, M_model, solver, load_solutions=False)
    except Exception as e:
        return {
            "idx": idx,
            "feasible": False,
            "status": f"Error: {str(e)}",
            "traceback": traceback.format_exc(),
        }


def _screening_x0_sequential(args):
    """4-tuple same as transfer; 5-tuple adds solve_mode (cold / warm_first / warm_next)."""
    n = len(args)
    if n == 4:
        return _screening_x0_transfer(args)
    if n != 5:
        raise TypeError(f"_screening_x0_sequential expected 4 or 5 arguments, got {n}")
    idx, current_x0_array, M_model, solver, solve_mode = args
    solve_mode = _normalize_screening_solve_mode(solve_mode)
    try:
        return _screening_run(
            idx, current_x0_array, M_model, solver, _screening_load_solutions(solve_mode)
        )
    except Exception as e:
        return {
            "idx": idx,
            "feasible": False,
            "status": f"Error: {str(e)}",
            "traceback": traceback.format_exc(),
        }


def _ensure_screening_transfer_ok(result, where="screening"):
    """If result has ``status``, print traceback and raise (for notebooks)."""
    if "status" not in result:
        return
    print(f"\n[solve failed] {where} idx={result.get('idx')}:\n{result['status']}")
    tb = result.get("traceback")
    if tb:
        print(tb)
    raise RuntimeError(f"{where} idx={result.get('idx')}: {result['status']}")


def _input_pixel_apply_bounds(M_model, pixel_lb, pixel_ub):
    s_val = len(pixel_lb)
    lb_dict = {i + 1: float(pixel_lb[i]) for i in range(s_val)}
    ub_dict = {i + 1: float(pixel_ub[i]) for i in range(s_val)}
    M_model.x0_pixel_lb.store_values(lb_dict)
    M_model.x0_pixel_ub.store_values(ub_dict)


def _input_pixel_run(idx, pixel_lb, pixel_ub, M_model, solver, load_solutions: bool):
    _input_pixel_apply_bounds(M_model, pixel_lb, pixel_ub)
    results = _solve_with_optional_quiet(
        solver, M_model, tee=False, load_solutions=load_solutions
    )
    status = results.solver.status
    tc = results.solver.termination_condition
    feasible = status == pyo.SolverStatus.ok and tc in (
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible,
    )
    return {"idx": idx, "feasible": feasible}


def _input_pixelization_pixel_transfer(args):
    """Parallel worker: (idx, pixel_lb, pixel_ub, model, solver), cold."""
    idx, pixel_lb, pixel_ub, M_model, solver = args
    try:
        return _input_pixel_run(idx, pixel_lb, pixel_ub, M_model, solver, load_solutions=False)
    except Exception as e:
        return {
            "idx": idx,
            "feasible": False,
            "status": f"Error: {str(e)}",
            "traceback": traceback.format_exc(),
        }


def _input_pixelization_pixel_sequential(args):
    """5-tuple same as transfer; 6-tuple adds solve_mode."""
    n = len(args)
    if n == 5:
        return _input_pixelization_pixel_transfer(args)
    if n != 6:
        raise TypeError(f"_input_pixelization_pixel_sequential expected 5 or 6 arguments, got {n}")
    idx, pixel_lb, pixel_ub, M_model, solver, solve_mode = args
    solve_mode = _normalize_screening_solve_mode(solve_mode)
    try:
        return _input_pixel_run(
            idx, pixel_lb, pixel_ub, M_model, solver, _screening_load_solutions(solve_mode)
        )
    except Exception as e:
        return {
            "idx": idx,
            "feasible": False,
            "status": f"Error: {str(e)}",
            "traceback": traceback.format_exc(),
        }


def _output_pixel_apply_bounds(M_model, m0_pixel_lb, m0_pixel_ub):
    d_val = len(m0_pixel_lb)
    lb_dict = {i + 1: float(m0_pixel_lb[i]) for i in range(d_val)}
    ub_dict = {i + 1: float(m0_pixel_ub[i]) for i in range(d_val)}
    M_model.m0_pixel_lb.store_values(lb_dict)
    M_model.m0_pixel_ub.store_values(ub_dict)


def _output_pixel_run(idx, m0_pixel_lb, m0_pixel_ub, M_model, solver, time_limit_sec, load_solutions: bool):
    if time_limit_sec is not None:
        set_solver_time_limit(solver, time_limit_sec)
    _output_pixel_apply_bounds(M_model, m0_pixel_lb, m0_pixel_ub)
    results = _solve_with_optional_quiet(
        solver, M_model, tee=False, load_solutions=load_solutions
    )
    status = results.solver.status
    tc = results.solver.termination_condition
    max_tl = getattr(pyo.TerminationCondition, "maxTimeLimit", None)
    if max_tl is not None and tc == max_tl:
        return {"idx": idx, "feasible": True, "timeout": True}
    feasible = status == pyo.SolverStatus.ok and tc in (
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible,
    )
    return {"idx": idx, "feasible": feasible}


def _output_pixelization_pixel_transfer(args):
    """5-tuple without time limit; 6-tuple ends with time_limit_sec (seconds)."""
    n = len(args)
    if n == 5:
        idx, m0_pixel_lb, m0_pixel_ub, M_model, solver = args
        time_limit_sec = None
    elif n == 6:
        idx, m0_pixel_lb, m0_pixel_ub, M_model, solver, time_limit_sec = args
    else:
        raise TypeError(f"_output_pixelization_pixel_transfer expected 5 or 6 arguments, got {n}")
    try:
        return _output_pixel_run(
            idx, m0_pixel_lb, m0_pixel_ub, M_model, solver, time_limit_sec, load_solutions=False
        )
    except Exception as e:
        return {
            "idx": idx,
            "feasible": False,
            "status": f"Error: {str(e)}",
            "traceback": traceback.format_exc(),
        }


def _output_pixelization_pixel_sequential(args):
    """5/6 same as transfer; 6-tuple with last str/bool is solve_mode; 7-tuple is (..., time_limit_sec, solve_mode)."""
    n = len(args)
    if n not in (5, 6, 7):
        raise TypeError(f"_output_pixelization_pixel_sequential expected 5, 6, or 7 arguments, got {n}")
    if n == 5:
        return _output_pixelization_pixel_transfer(args)
    if n == 6:
        if isinstance(args[5], (str, bool)):
            idx, m0_lb, m0_ub, m, s, sm = args
            return _output_seq_warm(idx, m0_lb, m0_ub, m, s, None, sm)
        return _output_pixelization_pixel_transfer(args)
    idx, m0_lb, m0_ub, m, s, tl, sm = args
    return _output_seq_warm(idx, m0_lb, m0_ub, m, s, tl, sm)


def _output_seq_warm(idx, m0_pixel_lb, m0_pixel_ub, M_model, solver, time_limit_sec, solve_mode):
    solve_mode = _normalize_screening_solve_mode(solve_mode)
    try:
        return _output_pixel_run(
            idx,
            m0_pixel_lb,
            m0_pixel_ub,
            M_model,
            solver,
            time_limit_sec,
            _screening_load_solutions(solve_mode),
        )
    except Exception as e:
        return {
            "idx": idx,
            "feasible": False,
            "status": f"Error: {str(e)}",
            "traceback": traceback.format_exc(),
        }


def _interval_run(idx, current_x0_array, M_model, solver, load_solutions: bool):
    _ = load_solutions  # callers pass for API symmetry; solve always loads (see below).
    s_val = len(current_x0_array)
    x0_dict = {i + 1: float(current_x0_array[i]) for i in range(s_val)}
    M_model.x0_param.store_values(x0_dict)

    results = _solve_with_optional_quiet(
        solver, M_model, tee=False, load_solutions=True
    )
    tc = results.solver.termination_condition
    if tc == pyo.TerminationCondition.optimal:
        bound = pyo.value(M_model.OBJ)
        return {"idx": idx, "bound": bound, "x0": current_x0_array.tolist()}

    unbounded_tcs = (
        pyo.TerminationCondition.unbounded,
        pyo.TerminationCondition.infeasibleOrUnbounded,
    )
    if tc in unbounded_tcs:
        sign = 1.0 if M_model.OBJ.sense == pyo.maximize else -1.0
        return {"idx": idx, "bound": sign * float("inf"), "unbounded": True,
                "x0": current_x0_array.tolist()}
    if tc == pyo.TerminationCondition.maxTimeLimit:
        return {"idx": idx, "timeout": True, "x0": current_x0_array.tolist()}
    return {
        "idx": idx,
        "status": f"Termination: {tc}",
    }


def _interval_x0_transfer(args):
    """Parallel worker: plausible interval, 4-tuple, cold."""
    idx, current_x0_array, M_model, solver = args
    try:
        return _interval_run(idx, current_x0_array, M_model, solver, load_solutions=False)
    except Exception as e:
        return {"idx": idx, "status": f"Error: {str(e)}", "traceback": traceback.format_exc()}


def _interval_x0_sequential(args):
    """4-tuple same as transfer; 5-tuple adds solve_mode."""
    n = len(args)
    if n == 4:
        return _interval_x0_transfer(args)
    if n != 5:
        raise TypeError(f"_interval_x0_sequential expected 4 or 5 arguments, got {n}")
    idx, current_x0_array, M_model, solver, solve_mode = args
    solve_mode = _normalize_screening_solve_mode(solve_mode)
    try:
        return _interval_run(
            idx, current_x0_array, M_model, solver, _screening_load_solutions(solve_mode)
        )
    except Exception as e:
        return {"idx": idx, "status": f"Error: {str(e)}", "traceback": traceback.format_exc()}


# =============================================================================
# Batch: task lists + Pool.map / sequential loops
# =============================================================================


def _screening_parallel_tasks(
    feas_region,
    M_model,
    solver,
    indices: Optional[Iterable[int]] = None,
) -> List[Tuple[Any, Any, Any, Any]]:
    """``feas_region`` shape (n, s); tasks for ``pool.map(_screening_x0_transfer, tasks)``."""
    X = _as_2d(feas_region, "feas_region")
    idxs = _row_indices(X.shape[0], indices)
    return [(i, X[i], M_model, solver) for i in idxs]


def _run_screening_parallel(pool, tasks: Sequence[Tuple[Any, Any, Any, Any]]) -> List[dict]:
    """``pool`` must provide ``.map(fn, iterable)`` (e.g. ``multiprocessing.Pool``)."""
    return pool.map(_screening_x0_transfer, list(tasks))


def _run_screening_sequential_cold(
    feas_region,
    M_model,
    solver,
    indices: Optional[Iterable[int]] = None,
) -> List[dict]:
    """Sequential screening on one model, cold each point."""
    X = _as_2d(feas_region, "feas_region")
    idxs = _row_indices(X.shape[0], indices)
    return [_screening_x0_sequential((i, X[i], M_model, solver)) for i in idxs]


def _run_screening_sequential_warm(
    feas_region,
    M_model,
    solver,
    indices: Optional[Iterable[int]] = None,
) -> List[dict]:
    """Sequential screening: ``warm_next`` only after a **feasible** previous row; else cold.

    First row uses ``warm_first`` (no prior primal to reuse). After an infeasible row, the
    next call is a 4-tuple cold solve (``load_solutions=False``).
    """
    X = _as_2d(feas_region, "feas_region")
    idxs = _row_indices(X.shape[0], indices)
    out: List[dict] = []
    prev_feasible = False
    for k, i in enumerate(idxs):
        if k == 0:
            out.append(_screening_x0_sequential((i, X[i], M_model, solver, "warm_first")))
        elif prev_feasible:
            out.append(_screening_x0_sequential((i, X[i], M_model, solver, "warm_next")))
        else:
            out.append(_screening_x0_sequential((i, X[i], M_model, solver)))
        prev_feasible = bool(out[-1].get("feasible", False))
    return out


def _input_pixelization_parallel_tasks(
    pixel_lb,
    pixel_ub,
    M_model,
    solver,
    indices: Optional[Iterable[int]] = None,
) -> List[Tuple[Any, Any, Any, Any, Any]]:
    """``pixel_lb`` and ``pixel_ub`` same shape (n, s)."""
    LB = _as_2d(pixel_lb, "pixel_lb")
    UB = _as_2d(pixel_ub, "pixel_ub")
    if LB.shape != UB.shape:
        raise ValueError(f"pixel_lb shape {LB.shape} != pixel_ub shape {UB.shape}")
    idxs = _row_indices(LB.shape[0], indices)
    return [(i, LB[i], UB[i], M_model, solver) for i in idxs]


def _run_input_pixelization_parallel(pool, tasks: Sequence[Tuple[Any, ...]]) -> List[dict]:
    return pool.map(_input_pixelization_pixel_transfer, list(tasks))


def _run_input_pixelization_sequential_cold(
    pixel_lb,
    pixel_ub,
    M_model,
    solver,
    indices: Optional[Iterable[int]] = None,
) -> List[dict]:
    tasks = _input_pixelization_parallel_tasks(pixel_lb, pixel_ub, M_model, solver, indices)
    return [_input_pixelization_pixel_sequential(t) for t in tasks]


def _run_input_pixelization_sequential_warm(
    pixel_lb,
    pixel_ub,
    M_model,
    solver,
    indices: Optional[Iterable[int]] = None,
) -> List[dict]:
    """``warm_next`` only if the previous pixel solve was ``feasible``; otherwise cold."""
    tasks = _input_pixelization_parallel_tasks(pixel_lb, pixel_ub, M_model, solver, indices)
    out: List[dict] = []
    prev_feasible = False
    for k, t in enumerate(tasks):
        i, lb, ub, m, s = t
        if k == 0:
            out.append(_input_pixelization_pixel_sequential((i, lb, ub, m, s, "warm_first")))
        elif prev_feasible:
            out.append(_input_pixelization_pixel_sequential((i, lb, ub, m, s, "warm_next")))
        else:
            out.append(_input_pixelization_pixel_sequential((i, lb, ub, m, s)))
        prev_feasible = bool(out[-1].get("feasible", False))
    return out


def _output_pixelization_parallel_tasks(
    m0_pixel_lb,
    m0_pixel_ub,
    M_model,
    solver,
    indices: Optional[Iterable[int]] = None,
    time_limit_sec: Optional[float] = None,
) -> List[Tuple[Any, ...]]:
    """``m0_pixel_lb`` / ``m0_pixel_ub`` shape (n, d); if ``time_limit_sec`` set, each task is 6-tuple."""
    LB = _as_2d(m0_pixel_lb, "m0_pixel_lb")
    UB = _as_2d(m0_pixel_ub, "m0_pixel_ub")
    if LB.shape != UB.shape:
        raise ValueError(f"m0_pixel_lb shape {LB.shape} != m0_pixel_ub shape {UB.shape}")
    idxs = _row_indices(LB.shape[0], indices)
    if time_limit_sec is None:
        return [(i, LB[i], UB[i], M_model, solver) for i in idxs]
    tl = float(time_limit_sec)
    return [(i, LB[i], UB[i], M_model, solver, tl) for i in idxs]


def _run_output_pixelization_parallel(pool, tasks: Sequence[Tuple[Any, ...]]) -> List[dict]:
    return pool.map(_output_pixelization_pixel_transfer, list(tasks))


def _run_output_pixelization_sequential_cold(
    m0_pixel_lb,
    m0_pixel_ub,
    M_model,
    solver,
    indices: Optional[Iterable[int]] = None,
    time_limit_sec: Optional[float] = None,
) -> List[dict]:
    tasks = _output_pixelization_parallel_tasks(
        m0_pixel_lb, m0_pixel_ub, M_model, solver, indices, time_limit_sec
    )
    return [_output_pixelization_pixel_transfer(t) for t in tasks]


def _run_output_pixelization_sequential_warm(
    m0_pixel_lb,
    m0_pixel_ub,
    M_model,
    solver,
    indices: Optional[Iterable[int]] = None,
    time_limit_sec: Optional[float] = None,
) -> List[dict]:
    """``warm_next`` only if the previous row was ``feasible``; else cold. Optional 7-tuple when ``time_limit_sec`` is set."""
    LB = _as_2d(m0_pixel_lb, "m0_pixel_lb")
    UB = _as_2d(m0_pixel_ub, "m0_pixel_ub")
    if LB.shape != UB.shape:
        raise ValueError(f"m0_pixel_lb shape {LB.shape} != m0_pixel_ub shape {UB.shape}")
    idxs = _row_indices(LB.shape[0], indices)
    out: List[dict] = []
    prev_feasible = False
    tl = float(time_limit_sec) if time_limit_sec is not None else None
    for k, i in enumerate(idxs):
        if k == 0:
            mode = "warm_first"
        elif prev_feasible:
            mode = "warm_next"
        else:
            mode = None
        if mode is None:
            if tl is None:
                args = (i, LB[i], UB[i], M_model, solver)
            else:
                args = (i, LB[i], UB[i], M_model, solver, tl)
            out.append(_output_pixelization_pixel_transfer(args))
        else:
            if tl is None:
                args = (i, LB[i], UB[i], M_model, solver, mode)
            else:
                args = (i, LB[i], UB[i], M_model, solver, tl, mode)
            out.append(_output_pixelization_pixel_sequential(args))
        prev_feasible = bool(out[-1].get("feasible", False))
    return out


def _interval_parallel_tasks(
    x0_grid,
    M_model,
    solver,
    indices: Optional[Iterable[int]] = None,
) -> List[Tuple[Any, Any, Any, Any]]:
    """``x0_grid`` shape (n, s), one x0 per row."""
    X = _as_2d(x0_grid, "x0_grid")
    idxs = _row_indices(X.shape[0], indices)
    return [(i, X[i], M_model, solver) for i in idxs]


def _run_interval_parallel(pool, tasks: Sequence[Tuple[Any, Any, Any, Any]]) -> List[dict]:
    return pool.map(_interval_x0_transfer, list(tasks))


def _run_interval_sequential_cold(
    x0_grid,
    M_model,
    solver,
    indices: Optional[Iterable[int]] = None,
) -> List[dict]:
    tasks = _interval_parallel_tasks(x0_grid, M_model, solver, indices)
    return [_interval_x0_sequential(t) for t in tasks]


def _run_interval_sequential_warm(
    x0_grid,
    M_model,
    solver,
    indices: Optional[Iterable[int]] = None,
) -> List[dict]:
    """``warm_next`` only if the previous row returned a ``bound`` (optimal); else cold."""
    X = _as_2d(x0_grid, "x0_grid")
    idxs = _row_indices(X.shape[0], indices)
    out: List[dict] = []
    prev_ok = False
    for k, i in enumerate(idxs):
        if k == 0:
            out.append(_interval_x0_sequential((i, X[i], M_model, solver, "warm_first")))
        elif prev_ok:
            out.append(_interval_x0_sequential((i, X[i], M_model, solver, "warm_next")))
        else:
            out.append(_interval_x0_sequential((i, X[i], M_model, solver)))
        prev_ok = "bound" in out[-1]
    return out


# =============================================================================
# Unified public API
# =============================================================================

_FEASIBILITY_TYPES = ("screening", "input_pixelization", "output_pixelization")
_INTERVAL_TYPES = ("upper_plausible_interval", "lower_plausible_interval")
_ALL_INFERENCE_TYPES = _FEASIBILITY_TYPES + _INTERVAL_TYPES


@dataclass
class ExecutionConfig:
    """Settings for one single-model execution-engine sweep.

    ``inference_type`` selects the procedure. ``execution`` is ``"sequential"``
    or ``"parallel"``. ``warm_start`` chains primal solutions across rows and is
    honored only for ``"sequential"`` (parallel is always cold). ``indices``
    optionally restricts which rows are solved. ``time_limit_sec`` applies only
    to ``"output_pixelization"``. ``pool_processes`` sizes the pool when
    ``execution == "parallel"`` (defaults to all CPUs).
    """

    inference_type: str
    execution: str = "sequential"
    warm_start: bool = False
    indices: Optional[Iterable[int]] = None
    time_limit_sec: Optional[float] = None
    pool_processes: Optional[int] = None


def _validate_config(config: "ExecutionConfig") -> None:
    if config.inference_type not in _ALL_INFERENCE_TYPES:
        raise ValueError(
            f"inference_type must be one of {_ALL_INFERENCE_TYPES}, got {config.inference_type!r}"
        )
    if config.execution not in ("sequential", "parallel"):
        raise ValueError(
            f"execution must be 'sequential' or 'parallel', got {config.execution!r}"
        )


def _spawn_pool(pool_processes: Optional[int]):
    """Spawn-context multiprocessing Pool context manager (all CPUs by default)."""
    ctx = multiprocessing.get_context("spawn")
    n = pool_processes if pool_processes is not None else (os.cpu_count() or 1)
    return ctx.Pool(processes=max(1, int(n)))


def _rows_to_dataframe(inference_type: str, rows: List[dict]) -> "pd.DataFrame":
    """Map raw worker result dicts to the uniform engine schema.

    Feasibility types produce ``result`` in {``returned``, ``screened_out``};
    interval types produce ``bound`` (a finite value when optimal, or +/-inf when
    the endpoint is unbounded in the optimization direction). ``solve_status`` is
    one of ``completed`` / ``timeout`` / ``error``; ``screened_out`` and an
    unbounded interval endpoint are both ``completed`` outcomes, not failures.
    """
    records: List[dict] = []
    for r in rows:
        rec: dict = {"idx": int(r["idx"])}
        if "status" in r:
            rec["solve_status"] = "error"
            rec["traceback"] = r.get("traceback")
        elif r.get("timeout"):
            rec["solve_status"] = "timeout"
        else:
            rec["solve_status"] = "completed"
        if inference_type in _FEASIBILITY_TYPES:
            if rec["solve_status"] == "error":
                rec["result"] = None
            else:
                rec["result"] = "returned" if r.get("feasible", False) else "screened_out"
        else:
            rec["bound"] = r.get("bound")
        records.append(rec)
    return pd.DataFrame.from_records(records)


def _resolve_data(inference_type: str, data):
    """Split ``data`` into positional arrays per inference_type."""
    if inference_type in ("screening",) + _INTERVAL_TYPES:
        return (data,)
    lb, ub = data
    return (lb, ub)


def run_execution_engine(config: "ExecutionConfig", model, solver, data) -> "pd.DataFrame":
    """Run one execution-engine sweep and return a uniform result DataFrame.

    See ``ExecutionConfig`` for settings. ``data`` is an ``x0_grid`` (screening /
    interval) or a ``(lower, upper)`` bound pair (input / output pixelization).
    The returned DataFrame always has ``idx`` and ``solve_status`` columns; it
    adds ``result`` for feasibility types and ``bound`` for interval types, plus
    ``traceback`` when a row errored.
    """
    _validate_config(config)
    it = config.inference_type
    seq = config.execution == "sequential"
    warm = bool(config.warm_start)

    if it == "screening":
        (X,) = _resolve_data(it, data)
        if not seq:
            tasks = _screening_parallel_tasks(X, model, solver, config.indices)
            with _spawn_pool(config.pool_processes) as pool:
                rows = _run_screening_parallel(pool, tasks)
        elif warm:
            rows = _run_screening_sequential_warm(X, model, solver, config.indices)
        else:
            rows = _run_screening_sequential_cold(X, model, solver, config.indices)

    elif it == "input_pixelization":
        lb, ub = _resolve_data(it, data)
        if not seq:
            tasks = _input_pixelization_parallel_tasks(lb, ub, model, solver, config.indices)
            with _spawn_pool(config.pool_processes) as pool:
                rows = _run_input_pixelization_parallel(pool, tasks)
        elif warm:
            rows = _run_input_pixelization_sequential_warm(lb, ub, model, solver, config.indices)
        else:
            rows = _run_input_pixelization_sequential_cold(lb, ub, model, solver, config.indices)

    elif it == "output_pixelization":
        lb, ub = _resolve_data(it, data)
        if not seq:
            tasks = _output_pixelization_parallel_tasks(
                lb, ub, model, solver, config.indices, config.time_limit_sec
            )
            with _spawn_pool(config.pool_processes) as pool:
                rows = _run_output_pixelization_parallel(pool, tasks)
        elif warm:
            rows = _run_output_pixelization_sequential_warm(
                lb, ub, model, solver, config.indices, config.time_limit_sec
            )
        else:
            rows = _run_output_pixelization_sequential_cold(
                lb, ub, model, solver, config.indices, config.time_limit_sec
            )

    else:  # upper/lower plausible interval
        (X,) = _resolve_data(it, data)
        if not seq:
            tasks = _interval_parallel_tasks(X, model, solver, config.indices)
            with _spawn_pool(config.pool_processes) as pool:
                rows = _run_interval_parallel(pool, tasks)
        elif warm:
            rows = _run_interval_sequential_warm(X, model, solver, config.indices)
        else:
            rows = _run_interval_sequential_cold(X, model, solver, config.indices)

    return _rows_to_dataframe(it, rows)
