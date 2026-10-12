import numpy as np

from plausible_inference.execution_engine import ExecutionConfig, run_execution_engine
from plausible_inference.model_construction import model_construction

def _linear_function(exp_set):
    return np.asarray(exp_set, dtype=float)

def _quadratic_function(exp_set, center=0.0):
    return np.square(np.asarray(exp_set, dtype=float) - center)

def _pixelize(gurobi_solver, inference_type, exp_set, values, pixel_lb, pixel_ub, acceptability, functional_properties, acceptability_parameters=None):
    values = np.asarray(values, dtype=float)
    model = model_construction(
        np.asarray(exp_set, dtype=float),
        num_objectives=values.shape[1],
        inference_type=inference_type,
        functional_properties_list=functional_properties,
        acceptability=acceptability,
        acceptability_parameters=acceptability_parameters,
        discrepancy_type="confidence_region",
        upper_confidence_bounds=values,
        lower_confidence_bounds=values,
        feasibility_check_solver=None,
    )
    df = run_execution_engine(
        ExecutionConfig(inference_type=inference_type, execution="sequential"),
        model,
        gurobi_solver,
        (np.asarray(pixel_lb, dtype=float), np.asarray(pixel_ub, dtype=float)),
    )
    df = df.sort_values("idx")
    assert list(df["solve_status"]) == ["completed"] * len(df)
    return list(df["result"])

def test_input_pixels_of_a_line_keep_intervals_reaching_the_threshold(gurobi_solver):
    exp_set = np.array([[0.0], [1.0], [2.0]])
    pixel_lb = np.array([[-1.0], [1.0], [2.0]])
    pixel_ub = np.array([[0.0], [2.0], [3.0]])
    assert _pixelize(
        gurobi_solver,
        "input_pixelization",
        exp_set,
        _linear_function(exp_set),
        pixel_lb,
        pixel_ub,
        "feasibility",
        [["convexity", "concavity"]],
        {"threshold": 1.5},
    ) == ["returned", "returned", "screened_out"]

def test_output_pixels_of_a_line_keep_values_at_or_below_the_threshold(gurobi_solver):
    exp_set = np.array([[0.0], [1.0], [2.0]])
    pixel_lb = np.array([[-1.0], [1.0], [1.6]])
    pixel_ub = np.array([[0.0], [1.5], [3.0]])
    assert _pixelize(
        gurobi_solver,
        "output_pixelization",
        exp_set,
        _linear_function(exp_set),
        pixel_lb,
        pixel_ub,
        "feasibility",
        [["convexity", "concavity"]],
        {"threshold": 1.5},
    ) == ["returned", "returned", "screened_out"]

def test_input_pixels_of_x_squared_keep_intervals_around_zero(gurobi_solver):
    exp_set = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    pixel_lb = np.array([[-3.0], [-1.0], [2.0]])
    pixel_ub = np.array([[-2.0], [1.0], [3.0]])
    assert _pixelize(
        gurobi_solver,
        "input_pixelization",
        exp_set,
        _quadratic_function(exp_set),
        pixel_lb,
        pixel_ub,
        "single-objective-optimality",
        [["convexity"]],
    ) == ["screened_out", "returned", "screened_out"]

def test_output_pixels_of_two_quadratics_keep_the_pareto_front(gurobi_solver):

    exp_set = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    values = np.hstack([
        _quadratic_function(exp_set, center=-1.0),
        _quadratic_function(exp_set, center=1.0),
    ])
    pixel_lb = np.array([[0.0, 3.0], [1.0, 1.0], [4.0, 0.0], [8.0, 8.0], [-9.0, -9.0]])
    pixel_ub = np.array([[1.0, 4.0], [2.0, 2.0], [5.0, 1.0], [9.0, 9.0], [-8.0, -8.0]])
    assert _pixelize(
        gurobi_solver,
        "output_pixelization",
        exp_set,
        values,
        pixel_lb,
        pixel_ub,
        "Pareto-optimality",
        [["convexity"], ["convexity"]],
    ) == ["returned", "returned", "returned", "screened_out", "screened_out"]
