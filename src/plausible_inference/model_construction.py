import pyomo.environ as pyo
import numpy as np
from .rules.functional_structure import add_functional_properties_constraints
from .rules.acceptability import add_acceptability_constraints
from .rules.confidence_regions import add_confidence_region_constraints
from .utils.model_construction_inputs import validate_model_construction_core_inputs
from .utils.solver_config import get_plausibility_solver, STRUCTURAL_FEASIBILITY_CHECK_SOLVERS
from .execution_engine import ExecutionConfig, run_execution_engine


def _rule_x0_pixel_lb(m, s):
    """Constraint rule: x0[s] >= x0_pixel_lb[s]. Module-level so the model is picklable for multiprocessing."""
    return m.x0[s] >= m.x0_pixel_lb[s]


def _rule_x0_pixel_ub(m, s):
    """Constraint rule: x0[s] <= x0_pixel_ub[s]. Module-level so the model is picklable for multiprocessing."""
    return m.x0[s] <= m.x0_pixel_ub[s]


def _rule_x0_feasible_ineq(m, i):
    """Linear inequality i: sum_s A_ineq[i,s]*x0[s] <= b_ineq[i]. Module-level for pickle."""
    return sum(m.A_ineq[i, s] * m.x0[s] for s in m.S_set) <= m.b_ineq[i]


def _rule_m0_pixel_lb(m, d):
    """m[0,d] >= m0_pixel_lb[d]. Module-level for pickle."""
    return m.m[0, d] >= m.m0_pixel_lb[d]


def _rule_m0_pixel_ub(m, d):
    """m[0,d] <= m0_pixel_ub[d]. Module-level for pickle."""
    return m.m[0, d] <= m.m0_pixel_ub[d]


def parameter_definition(exp_set, num_objectives, inference_type):
    """
    Initialize the model with the experimental set.

    For inference_type in {screening, upper_plausible_interval, lower_plausible_interval},
    x0 is fixed (mutable Param x0_param; values injected later via store_values).
    Otherwise x0 is a decision variable; when inference_type == 'input_pixelization',
    mutable Params x0_pixel_lb and x0_pixel_ub are added; when inference_type ==
    'output_pixelization', mutable Params m0_pixel_lb and m0_pixel_ub (on D_set) are
    added so m[0,:] is constrained to [lb, ub]. Values are set later via store_values.
    """
    model = pyo.ConcreteModel()
    k = exp_set.shape[0]  # number of experimental solutions
    s = exp_set.shape[1]  # dimension of each experimental solution
    d=num_objectives
    S_set_values = list(range(1, s + 1))
    model.S_set = pyo.Set(initialize=S_set_values)
    K_set_values = list(range(1, k + 1))
    model.K_set = pyo.Set(initialize=K_set_values)
    K_ext_set_values = list(range(k + 1))
    model.K_ext_set = pyo.Set(initialize=K_ext_set_values)
    D_set_values = list(range(1, d + 1))
    model.D_set = pyo.Set(initialize=D_set_values)

    model.m = pyo.Var(model.K_ext_set,model.D_set, domain=pyo.Reals)
    # NOTE: g is not necessarily needed for the model
    # I think it cost litte to do so; otherwise, we only define it when needed (such as when convexity is involved).
    model.g = pyo.Var(model.K_ext_set, model.D_set, model.S_set, domain=pyo.Reals)

    INFERENCE_TYPES_LIST_X0_TYPE = {'screening', 'upper_plausible_interval', 'lower_plausible_interval'}
    if inference_type in INFERENCE_TYPES_LIST_X0_TYPE:
        x0_type = 'fixed'
    else:
        x0_type = 'decision_variable'

    if x0_type == 'fixed':
        # For plausible screening, plausible interval, or Lipschitz constant inference, the x0 is a fixed vector.
        # We define it as a mutable parameter. As those form oftent required to be conducted at different x0.
        model.x0_param = pyo.Param(model.S_set, mutable=True, initialize={i: 0.0 for i in S_set_values})
        x0_initial = np.array([[model.x0_param[i] for i in S_set_values]])  # shape (1, s)
        model._exp_set_data = np.vstack([x0_initial, exp_set])  # shape (k+1, s), x0 at index 0
    else:
        # x0 is a decision variable (e.g. input_pixelization: x0 constrained to a pixel [lb, ub]).
        model.x0 = pyo.Var(model.S_set, domain=pyo.Reals)
        x0_initial = np.array([[model.x0[i] for i in S_set_values]])
        model._exp_set_data = np.vstack([x0_initial, exp_set])
        # When inference_type is input_pixelization, pixel bounds are added here as mutable Params
        # (same idea as x0_param: not model inputs; initialized to 0, then set via store_values per solve).
        if inference_type == 'input_pixelization':
            model.x0_pixel_lb = pyo.Param(model.S_set, mutable=True, initialize={i: 0.0 for i in S_set_values})
            model.x0_pixel_ub = pyo.Param(model.S_set, mutable=True, initialize={i: 0.0 for i in S_set_values})
            model.C_x0_pixel_lb = pyo.Constraint(model.S_set, rule=_rule_x0_pixel_lb)
            model.C_x0_pixel_ub = pyo.Constraint(model.S_set, rule=_rule_x0_pixel_ub)
        elif inference_type == 'output_pixelization':
            # Pixelize m[0,:] (output at x0). Params set later via store_values in the output-pixelization worker (run_execution_engine).
            model.m0_pixel_lb = pyo.Param(model.D_set, mutable=True, initialize={i: 0.0 for i in D_set_values})
            model.m0_pixel_ub = pyo.Param(model.D_set, mutable=True, initialize={i: 1e10 for i in D_set_values})
            model.C_m0_pixel_lb = pyo.Constraint(model.D_set, rule=_rule_m0_pixel_lb)
            model.C_m0_pixel_ub = pyo.Constraint(model.D_set, rule=_rule_m0_pixel_ub)
    return model




def model_construction(exp_set, num_objectives, inference_type, functional_properties_list, acceptability=None, discrepancy_type=None, confidence_level=None, sample_mean=None, sample_var=None, sample_size=None, functional_properties_parameters=None, upper_confidence_bounds=None, lower_confidence_bounds=None, feasibility_check_solver='scip', x0_feasible_region_A_ineq=None, x0_feasible_region_b_ineq=None, acceptability_parameters=None, interval_objective_index=1):
    """
    Construct the model (functional + confidence region + optional acceptability + objective).

    After adding functional and confidence-region constraints (and before
    adding acceptability), a feasibility check is run once: no constraints are
    removed; a temporary objective is added, the model is solved, then the
    temporary objective is deleted. If infeasible, inference is aborted.
    Because this check is independent of x0, it runs only once per
    model_construction call (e.g. once in the main process before a parallel
    loop). Pass feasibility_check_solver=None to skip the check.
    Only solvers that reliably report infeasible (e.g. 'gurobi') are
    supported for this check (e.g. Gurobi, SCIP). See
    STRUCTURAL_FEASIBILITY_CHECK_SOLVERS in solver_config.
    When feasibility_check_solver is 'gurobi', the check uses gurobi_direct
    (in-memory API, no LP file) so it works for both linear and nonlinear models.
    For inference_type='input_pixelization' or 'output_pixelization', optional x0_feasible_region_A_ineq (2D, n_ineq x s)
    and x0_feasible_region_b_ineq (1D, n_ineq) add linear constraints A_ineq @ x0 <= b_ineq.
    acceptability_parameters (dict) supplies numeric constants for the threshold-based
    acceptabilities (minimization):
    'delta-optimality' (single-objective) -> {'delta': >=0};
    'feasibility' (per-objective) -> {'threshold': scalar or length-num_objectives sequence};
    'closeness-to-target' (per-objective) -> {'threshold': scalar/sequence, 'delta': >=0 scalar/sequence}.
    Ignored for 'single-objective-optimality' and 'Pareto-optimality'.
    interval_objective_index (1-based, default 1) selects which objective is bounded for
    inference_type in {'upper_plausible_interval', 'lower_plausible_interval'}; callers
    sweep it over 1..num_objectives to get a plausible interval per objective.
    """
    validate_model_construction_core_inputs(
        num_objectives, functional_properties_list, inference_type, acceptability,
        acceptability_parameters=acceptability_parameters,
        discrepancy_type=discrepancy_type,
    )

    #============================================
    # Step 1: Model initialization (pixel constraints for input_pixelization are added inside parameter_definition).
    # For input_pixelization, x0_pixel_lb and x0_pixel_ub are mutable Params (like x0_param), set later via store_values.
    model = parameter_definition(exp_set, num_objectives, inference_type)
    if inference_type in ('input_pixelization', 'output_pixelization') and x0_feasible_region_A_ineq is not None and x0_feasible_region_b_ineq is not None:
        A = np.asarray(x0_feasible_region_A_ineq)
        b = np.asarray(x0_feasible_region_b_ineq).flatten()
        n_ineq, s = A.shape[0], exp_set.shape[1]
        if A.shape[1] != s or b.shape[0] != n_ineq:
            raise ValueError(f"x0_feasible_region_A_ineq must be (n_ineq x {s}), x0_feasible_region_b_ineq length n_ineq.")
        model.FeasibleIneq_set = pyo.Set(initialize=range(1, n_ineq + 1))
        model.A_ineq = pyo.Param(
            model.FeasibleIneq_set, model.S_set,
            initialize={(i, j): float(A[i - 1, j - 1]) for i in range(1, n_ineq + 1) for j in range(1, s + 1)}
        )
        model.b_ineq = pyo.Param(model.FeasibleIneq_set, initialize={i: float(b[i - 1]) for i in range(1, n_ineq + 1)})
        model.C_x0_feasible_ineq = pyo.Constraint(model.FeasibleIneq_set, rule=_rule_x0_feasible_ineq)

    #============================================
    # Step 2: Add constraints describing the functional properties

    model = add_functional_properties_constraints(model, functional_properties_list, functional_properties_parameters)

    #============================================
    # Step 3: Add confidence region (discrepancy) constraints
    if discrepancy_type is None:
        raise ValueError("discrepancy_type must be provided")

    discrepancy_set = {'norm_infinite', 'norm_1', 'norm_2', 'CRN'}
    if discrepancy_type in discrepancy_set:
        if confidence_level is None:
            raise ValueError(f"confidence_level must be provided when discrepancy_type is {discrepancy_type}")
        model = add_confidence_region_constraints(
            model, discrepancy_type,
            sample_mean=sample_mean, sample_var=sample_var, sample_size=sample_size,
            confidence_level=confidence_level, num_objectives=num_objectives
        )
    elif discrepancy_type == 'confidence_region':
        if upper_confidence_bounds is None or lower_confidence_bounds is None:
            raise ValueError("upper_confidence_bounds and lower_confidence_bounds must be provided when discrepancy_type is 'confidence_region'")

        # Convert to numpy arrays if not already
        upper_confidence_bounds = np.asarray(upper_confidence_bounds)
        lower_confidence_bounds = np.asarray(lower_confidence_bounds)

        expected_shape = (exp_set.shape[0], num_objectives)
        upper_shape = upper_confidence_bounds.shape
        lower_shape = lower_confidence_bounds.shape

        # Handle the case where num_objectives = 1 and bounds are 1D arrays (k,) instead of (k, 1)
        if num_objectives == 1:
            if upper_shape == (exp_set.shape[0],):
                upper_confidence_bounds = upper_confidence_bounds.reshape(-1, 1)
            if lower_shape == (exp_set.shape[0],):
                lower_confidence_bounds = lower_confidence_bounds.reshape(-1, 1)

        # Check shapes again after potential reshaping
        upper_shape = upper_confidence_bounds.shape
        lower_shape = lower_confidence_bounds.shape
        if upper_shape != expected_shape or lower_shape != expected_shape:
            raise ValueError(
                f"upper_confidence_bounds and lower_confidence_bounds must have the shape "
                f"(num of simulated solutions, num of objectives) = {expected_shape}. "
                f"Got upper_confidence_bounds shape: {upper_shape}, lower_confidence_bounds shape: {lower_shape}. "
                f"exp_set.shape[0] = {exp_set.shape[0]}, num_objectives = {num_objectives}"
            )
        model = add_confidence_region_constraints(model, discrepancy_type, upper_confidence_bounds=upper_confidence_bounds, lower_confidence_bounds=lower_confidence_bounds)

    #============================================
    # Step 4: Structural feasibility check (no constraint removal: only add temp objective, solve, remove)
    # Always use gurobi_direct when user asks for gurobi (in-memory API, works for linear and nonlinear; no LP file).
    if feasibility_check_solver is not None:
        name_lower = (feasibility_check_solver or "").strip().lower()
        if name_lower not in STRUCTURAL_FEASIBILITY_CHECK_SOLVERS:
            raise ValueError(
                f"feasibility_check_solver={feasibility_check_solver!r} is not supported for the structural feasibility check. "
                f"Use one of {STRUCTURAL_FEASIBILITY_CHECK_SOLVERS} (e.g. 'gurobi'), or feasibility_check_solver=None to skip the check. "
                "Use one of the supported solvers (e.g. Gurobi, SCIP) or set feasibility_check_solver=None to skip."
            )
        check_solver = 'gurobi_direct' if name_lower == 'gurobi' else feasibility_check_solver.strip()
        model._feas_check_obj = pyo.Objective(expr=0, sense=pyo.minimize)
        try:
            feas_solver = get_plausibility_solver(check_solver)
            try:
                results = feas_solver.solve(model, tee=False, load_solutions=False)
            except Exception as solve_err:
                raise RuntimeError(
                    f"Structural feasibility check failed: solver ({check_solver}) did not exit normally. "
                    f"Original error: {solve_err!s}. "
                    "Try another solver (e.g. Gurobi) or set feasibility_check_solver=None to skip the check."
                ) from solve_err
            tc = results.solver.termination_condition
            # Unbounded means feasible region is non-empty (objective unbounded); for min 0 that is feasible.
            feasible_tc = (
                pyo.TerminationCondition.optimal,
                pyo.TerminationCondition.feasible,
                pyo.TerminationCondition.unbounded,
            )
            # Only explicit infeasible; infeasibleOrUnbounded is ambiguous (solver could not tell) -> treat as "other".
            infeasible_tc = (pyo.TerminationCondition.infeasible,)
            if tc in feasible_tc:
                pass  # structural part is feasible, continue
            elif tc in infeasible_tc:
                raise ValueError(
                    "Model is infeasible under functional-property and confidence-region constraints only. "
                    "Inference rejects the hypothesis on the function information (e.g. confidence region incompatible with functional properties)."
                )
            else:
                raise RuntimeError(
                    f"Solver could not determine structural feasibility. Termination condition: {tc}. "
                    "Use a supported solver (e.g. Gurobi, SCIP via get_plausibility_solver) or set feasibility_check_solver=None to skip the check."
                )
        finally:
            model.del_component(model._feas_check_obj)

    #============================================
    # Step 5: Add acceptability constraints
    if acceptability is not None:
        model = add_acceptability_constraints(model, acceptability, functional_properties_list, acceptability_parameters=acceptability_parameters)

    #============================================
    # Step 6: Set the objective function
    if inference_type in ('screening', 'input_pixelization', 'output_pixelization'):
        model.OBJ = pyo.Objective(expr=0, sense=pyo.minimize)
    elif inference_type in ('upper_plausible_interval', 'lower_plausible_interval'):
        # Plausible interval is computed one objective at a time; interval_objective_index
        # (1-based) selects which objective's value at x0 is bounded.
        if not (1 <= interval_objective_index <= num_objectives):
            raise ValueError(
                f"interval_objective_index must be in [1, {num_objectives}], got {interval_objective_index}"
            )
        sense = pyo.maximize if inference_type == 'upper_plausible_interval' else pyo.minimize
        model.OBJ = pyo.Objective(expr=model.m[0, interval_objective_index], sense=sense)

    return model