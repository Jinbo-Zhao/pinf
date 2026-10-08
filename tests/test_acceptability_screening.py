import numpy as np

from plausible_inference.execution_engine import ExecutionConfig, run_execution_engine
from plausible_inference.model_construction import model_construction

def _linear_function(exp_set):
    return np.asarray(exp_set, dtype=float)

def _quadratic_function(exp_set, center=0.0):
    return np.square(np.asarray(exp_set, dtype=float) - center)

def _screen(gurobi_solver, exp_set, values, candidates, acceptability, functional_properties, acceptability_parameters=None):
    values = np.asarray(values, dtype=float)
    model = model_construction(
        np.asarray(exp_set, dtype=float),
        num_objectives=values.shape[1],
        inference_type="screening",
        functional_properties_list=functional_properties,
        acceptability=acceptability,
        acceptability_parameters=acceptability_parameters,
        discrepancy_type="confidence_region",
        upper_confidence_bounds=values,
        lower_confidence_bounds=values,
        feasibility_check_solver=None,
    )
    df = run_execution_engine(
        ExecutionConfig(inference_type="screening", execution="sequential"),
        model,
        gurobi_solver,
        np.asarray(candidates, dtype=float),
    )
    assert list(df["solve_status"]) == ["completed"] * len(candidates)
    return list(df["result"])

def test_feasibility_keeps_points_at_or_below_threshold(gurobi_solver):
    exp_set = np.array([[0.0], [1.0], [2.0]])
    candidates = np.array([[0.0], [0.5], [2.0]])
    assert _screen(
        gurobi_solver,
        exp_set,
        _linear_function(exp_set),
        candidates,
        "feasibility",
        [["convexity", "concavity"]],
        {"threshold": 1.5},
    ) == ["returned", "returned", "screened_out"]

def test_closeness_to_target_keeps_only_the_target(gurobi_solver):
    exp_set = np.array([[0.0], [1.0], [2.0]])
    candidates = np.array([[0.0], [1.0], [2.0]])
    assert _screen(
        gurobi_solver,
        exp_set,
        _linear_function(exp_set),
        candidates,
        "closeness-to-target",
        [["convexity", "concavity"]],
        {"threshold": 1.0, "delta": 0.2},
    ) == ["screened_out", "returned", "screened_out"]

def test_delta_optimality_keeps_points_within_delta_of_the_best(gurobi_solver):
    exp_set = np.array([[0.0], [1.0], [2.0]])
    candidates = np.array([[0.0], [0.5], [2.0]])
    assert _screen(
        gurobi_solver,
        exp_set,
        _linear_function(exp_set),
        candidates,
        "delta-optimality",
        [["convexity", "concavity"]],
        {"delta": 0.6},
    ) == ["returned", "returned", "screened_out"]

def test_single_objective_optimality_of_x_squared(gurobi_solver):
    exp_set = np.array([[-1.5], [-1.0], [1.0], [1.5]])
    candidates = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    assert _screen(
        gurobi_solver,
        exp_set,
        _quadratic_function(exp_set),
        candidates,
        "single-objective-optimality",
        [["convexity"]],
    ) == ["screened_out", "returned", "returned", "returned", "screened_out"]

def test_pareto_screens_dominated_points_and_keeps_non_dominated_ones(gurobi_solver):
    exp_set = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    candidates = np.array([[-3.0], [-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]])
    values = np.hstack([
        _quadratic_function(exp_set, center=-1.0),
        _quadratic_function(exp_set, center=1.0),
    ])
    assert _screen(
        gurobi_solver,
        exp_set,
        values,
        candidates,
        "Pareto-optimality",
        [["convexity"], ["convexity"]],
    ) == [
        "screened_out",
        "screened_out",
        "returned",
        "returned",
        "returned",
        "screened_out",
        "screened_out",
    ]
