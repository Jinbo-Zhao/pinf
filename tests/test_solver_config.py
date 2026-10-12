import numpy as np
import pyomo.environ as pyo

from plausible_inference.utils.solver_config import (
    get_plausibility_solver,
    set_solver_time_limit,
)


def _single_variable_model(kind):
    """Minimize one free variable. Infeasible adds the contradictory bounds 1 <= x <= 0."""
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Reals)
    model.obj = pyo.Objective(expr=model.x, sense=pyo.minimize)
    if kind == "infeasible":
        model.lower = pyo.Constraint(expr=model.x >= 1)
        model.upper = pyo.Constraint(expr=model.x <= 0)
    elif kind == "unbounded":
        pass
    return model


def _assert_distinguishes_infeasible_and_unbounded(solver_name, expected_options):
    solver = get_plausibility_solver(solver_name)
    for key, expected in expected_options.items():
        # Index rather than .get(): scip_direct's options dict returns a
        # placeholder from .get() even for keys that were set.
        assert solver.options[key] == expected
    infeasible = solver.solve(
        _single_variable_model("infeasible"), tee=False, load_solutions=False
    )
    unbounded = solver.solve(
        _single_variable_model("unbounded"), tee=False, load_solutions=False
    )
    assert infeasible.solver.termination_condition == pyo.TerminationCondition.infeasible
    assert unbounded.solver.termination_condition == pyo.TerminationCondition.unbounded


def test_get_plausibility_solver_gurobi_distinguishes_infeasible_and_unbounded(gurobi_solver):
    _assert_distinguishes_infeasible_and_unbounded("gurobi", {"DualReductions": 0})
    assert gurobi_solver.options.get("DualReductions") == 0


def test_get_plausibility_solver_scip_distinguishes_infeasible_and_unbounded(scip_solver):
    _assert_distinguishes_infeasible_and_unbounded(
        "scip",
        {
            "misc/allowstrongdualreds": False,
            "misc/allowweakdualreds": False,
        },
    )
    assert scip_solver.options.get("misc/allowstrongdualreds") is False
    assert scip_solver.options.get("misc/allowweakdualreds") is False


def _assert_time_limit(solver, option_key):
    set_solver_time_limit(solver, 12.5)
    try:
        assert float(solver.options[option_key]) == 12.5
    finally:
        del solver.options[option_key]


def test_gurobi_time_limit(gurobi_solver):
    _assert_time_limit(gurobi_solver, "TimeLimit")


def test_scip_time_limit(scip_solver):
    _assert_time_limit(scip_solver, "limits/time")


def test_get_plausibility_solver_scip_direct_distinguishes_infeasible_and_unbounded(
    scip_direct_solver,
):
    _assert_distinguishes_infeasible_and_unbounded(
        "scip_direct",
        {
            "misc/allowstrongdualreds": False,
            "misc/allowweakdualreds": False,
        },
    )


def test_scip_direct_time_limit(scip_direct_solver):
    _assert_time_limit(scip_direct_solver, "limits/time")


def test_scip_direct_is_supported_for_structural_feasibility_check(scip_direct_solver):
    from plausible_inference.model_construction import model_construction

    model = model_construction(
        np.array([[0.0], [1.0]]),
        num_objectives=1,
        inference_type="screening",
        functional_properties_list=[["convexity"]],
        acceptability="single-objective-optimality",
        discrepancy_type="confidence_region",
        upper_confidence_bounds=np.array([[2.0], [3.0]]),
        lower_confidence_bounds=np.array([[1.0], [2.0]]),
        feasibility_check_solver="scip_direct",
    )
    assert model is not None
