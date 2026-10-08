import numpy as np

from plausible_inference.execution_engine import ExecutionConfig, run_execution_engine
from plausible_inference.model_construction import model_construction

def _interval(gurobi_solver, exp_set, values, candidate, functional_properties, parameters=None):
    values = np.asarray(values, dtype=float)
    bounds = {}
    for inference_type in ("lower_plausible_interval", "upper_plausible_interval"):
        model = model_construction(
            np.asarray(exp_set, dtype=float),
            num_objectives=1,
            inference_type=inference_type,
            functional_properties_list=[functional_properties],
            functional_properties_parameters=parameters,
            acceptability=None,
            discrepancy_type="confidence_region",
            upper_confidence_bounds=values,
            lower_confidence_bounds=values,
            feasibility_check_solver=None,
        )
        df = run_execution_engine(
            ExecutionConfig(inference_type=inference_type, execution="sequential"),
            model,
            gurobi_solver,
            np.asarray(candidate, dtype=float),
        )
        assert list(df["solve_status"]) == ["completed"]
        bounds[inference_type] = float(df.loc[0, "bound"])
    return bounds["lower_plausible_interval"], bounds["upper_plausible_interval"]

def test_convexity_bounds_a_v_shaped_sample(gurobi_solver):
    exp_set = np.array([[-1.0], [0.0], [1.0]])
    values = np.array([[1.0], [0.0], [1.0]])
    for x in (-0.5, 0.5):
        lower, upper = _interval(gurobi_solver, exp_set, values, [[x]], ["convexity"])
        assert lower == -0.5
        assert upper == 0.5
    for x in (-2.0, 2.0):
        lower, upper = _interval(gurobi_solver, exp_set, values, [[x]], ["convexity"])
        assert lower == 2.0
        assert np.isposinf(upper)

def test_concavity_bounds_an_inverted_v_shaped_sample(gurobi_solver):
    exp_set = np.array([[-1.0], [0.0], [1.0]])
    values = np.array([[-1.0], [0.0], [-1.0]])
    for x in (-0.5, 0.5):
        lower, upper = _interval(gurobi_solver, exp_set, values, [[x]], ["concavity"])
        assert lower == -0.5
        assert upper == 0.5
    for x in (-2.0, 2.0):
        lower, upper = _interval(gurobi_solver, exp_set, values, [[x]], ["concavity"])
        assert np.isneginf(lower)
        assert upper == -2.0

def test_lipschitz_pins_the_midpoint_of_a_unit_slope(gurobi_solver):
    exp_set = np.array([[0.0], [1.0]])
    values = np.array([[0.0], [1.0]])
    parameters = [{"lip_CST": 1.0}]
    for x in (0.25, 0.5, 0.75):
        lower, upper = _interval(
            gurobi_solver, exp_set, values, [[x]], ["Lipschitz_continuity"], parameters
        )
        assert lower == x
        assert upper == x
    lower, upper = _interval(
        gurobi_solver, exp_set, values, [[-1.0]], ["Lipschitz_continuity"], parameters
    )
    assert lower == -1.0
    assert upper == 1.0
    lower, upper = _interval(
        gurobi_solver, exp_set, values, [[2.0]], ["Lipschitz_continuity"], parameters
    )
    assert lower == 0.0
    assert upper == 2.0

def test_directional_lipschitz_freezes_the_zero_constant_direction(gurobi_solver):
    exp_set = np.array([[0.0, 0.0], [1.0, 0.0]])
    values = np.array([[0.0], [0.0]])
    parameters = [{"lip_CST_vector": [2.0, 0.0]}]
    for x in (0.25, 0.75):
        lower, upper = _interval(
            gurobi_solver,
            exp_set,
            values,
            [[x, 0.0]],
            ["directional_Lipschitz_continuity"],
            parameters,
        )
        assert lower == -0.5
        assert upper == 0.5
    for point in ([0.5, 0.0], [0.5, 1.0]):
        lower, upper = _interval(
            gurobi_solver,
            exp_set,
            values,
            [point],
            ["directional_Lipschitz_continuity"],
            parameters,
        )
        assert lower == -1.0
        assert upper == 1.0
    for x in (-1.0, 2.0):
        lower, upper = _interval(
            gurobi_solver,
            exp_set,
            values,
            [[x, 0.0]],
            ["directional_Lipschitz_continuity"],
            parameters,
        )
        assert lower == -2.0
        assert upper == 2.0
    lower, upper = _interval(
        gurobi_solver,
        exp_set,
        values,
        [[0.0, 1.0]],
        ["directional_Lipschitz_continuity"],
        parameters,
    )
    assert lower == 0.0
    assert upper == 0.0
