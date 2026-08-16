"""
Solver configuration for plausible inference.

When solving LP models for screening or interval inference, solvers may report
"infeasible or unbounded" (e.g. INF_OR_UNBD) because presolve (e.g. dual reduction)
can hide the true status. For correct inference we need to distinguish
infeasible from unbounded.

This module provides a solver factory that sets options accordingly:
- Gurobi: DualReductions=0. See https://support.gurobi.com/hc/en-us/articles/4402704428177
- SCIP: misc/allowstrongdualreds and misc/allowweakdualreds = False.
GLPK is not supported for plausibility inference (no special handling).
"""

import pyomo.environ as pyo


# Known solver names that we can configure (lowercase for comparison).
GUROBI_NAMES = ('gurobi', 'gurobi_direct')
SCIP_NAMES = ('scip',)

# Solvers supported for the structural feasibility check (reliably report infeasible vs feasible).
# GLPK is not supported.
STRUCTURAL_FEASIBILITY_CHECK_SOLVERS = ('gurobi', 'gurobi_direct', 'scip')


def get_plausibility_solver(solver_name, **solver_options):
    """
    Return a Pyomo solver configured to distinguish infeasible vs unbounded.

    Use this instead of SolverFactory(...) when running screening or
    interval inference, so that termination condition is either
    INFEASIBLE or UNBOUNDED rather than INF_OR_UNBD.

    Parameters
    ----------
    solver_name : str
        Name of the solver (e.g. 'gurobi', 'scip'). GLPK is not supported.
    **solver_options
        Additional options to pass to the solver. These are applied after
        the infeasibility/unboundedness options.

    Returns
    -------
    solver : pyomo.opt.solver.solver.Solver
        Configured solver instance.

    Notes
    -----
    - Gurobi: sets DualReductions=0 so presolve does not hide infeasibility.
    - SCIP: sets misc/allowstrongdualreds and misc/allowweakdualreds to False.
    - Sequential screening warm chains use ``load_solutions=True`` only in the
      sequential worker path (no ``warmstart=`` kw to ``solve``, for Pyomo ConfigDict
      compatibility). Parallel execution is always cold.
    - GLPK is not supported (no special options; do not use for plausibility check).
    - Other solvers: returned without special options.
    """
    solver = pyo.SolverFactory(solver_name)
    name_lower = (solver_name or '').strip().lower()

    if name_lower in GUROBI_NAMES:
        solver.options['DualReductions'] = 0
    elif name_lower in SCIP_NAMES:
        # Disable dual reductions so SCIP can distinguish infeasible vs unbounded
        # (avoids INFORUNBD). Parameter names follow SCIP convention.
        solver.options['misc/allowstrongdualreds'] = False
        solver.options['misc/allowweakdualreds'] = False
    # else: no special options

    for key, value in solver_options.items():
        solver.options[key] = value

    return solver


def set_solver_time_limit(solver, seconds):
    """
    Set time limit (seconds) on a Pyomo solver so it stops after that time.
    Uses solver-native options: Gurobi TimeLimit, SCIP limits/time.
    Other solvers are left unchanged.

    Parameters
    ----------
    solver : pyomo.opt.solver.solver.Solver
        Solver instance from SolverFactory.
    seconds : float or int
        Time limit in seconds.
    """
    name = (getattr(solver, 'name', None) or '').strip().lower()
    if not name:
        for key in ('solver_name', '_solver_name'):
            n = getattr(solver, key, None)
            if n:
                name = str(n).strip().lower()
                break
    if not name:
        name = str(solver).lower()
    if name and 'gurobi' in name:
        solver.options['TimeLimit'] = float(seconds)
    elif name and 'scip' in name:
        solver.options['limits/time'] = float(seconds)
