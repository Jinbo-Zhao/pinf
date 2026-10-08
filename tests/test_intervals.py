import numpy as np

from plausible_inference.execution_engine import ExecutionConfig, run_execution_engine
from plausible_inference.model_construction import model_construction

def test_upper_and_lower_intervals_finite_and_unbounded_paths(gurobi_solver):
    exp_set = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    lower_confidence_bounds = np.array([[1.0], [2.0]], dtype=float)
    upper_confidence_bounds = np.array([[2.0], [3.0]], dtype=float)
    candidates = np.array([[0.5, 0.5]], dtype=float)

    finite_bounds = {}
    for inference_type in ("upper_plausible_interval", "lower_plausible_interval"):
        model = model_construction(
            exp_set,
            num_objectives=1,
            inference_type=inference_type,
            functional_properties_list=[["convexity", "Lipschitz_continuity"]],
            functional_properties_parameters=[{"lip_CST": 2.0}],
            acceptability=None,
            discrepancy_type="confidence_region",
            upper_confidence_bounds=upper_confidence_bounds,
            lower_confidence_bounds=lower_confidence_bounds,
            feasibility_check_solver=None,
        )
        df = run_execution_engine(
            ExecutionConfig(inference_type=inference_type, execution="sequential"),
            model,
            gurobi_solver,
            candidates,
        )
        assert len(df) == 1
        assert df.loc[0, "solve_status"] == "completed"
        bound = float(df.loc[0, "bound"])
        assert np.isfinite(bound)
        finite_bounds[inference_type] = bound

    assert (
        finite_bounds["upper_plausible_interval"]
        >= finite_bounds["lower_plausible_interval"]
    )

    model_unbounded = model_construction(
        exp_set,
        num_objectives=1,
        inference_type="lower_plausible_interval",
        functional_properties_list=[["convexity"]],
        acceptability=None,
        discrepancy_type="confidence_region",
        upper_confidence_bounds=upper_confidence_bounds,
        lower_confidence_bounds=lower_confidence_bounds,
        feasibility_check_solver=None,
    )
    df_u = run_execution_engine(
        ExecutionConfig(inference_type="lower_plausible_interval", execution="sequential"),
        model_unbounded,
        gurobi_solver,
        candidates,
    )
    assert df_u.loc[0, "solve_status"] == "completed"
    assert np.isneginf(float(df_u.loc[0, "bound"]))

def _flat_zero_trial():
    """Constant objective 0. Rectangle corners of (0, 0) and (1, -1), plus the center."""
    exp_set = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [1.0, -1.0],
            [0.5, -0.5],
        ],
        dtype=float,
    )
    bounds = np.zeros((exp_set.shape[0], 1))
    return exp_set, bounds

def _assert_flat_intervals_match_zero(gurobi_solver, functional_properties, parameters=None):
    exp_set, bounds = _flat_zero_trial()
    assert np.all(bounds == 0.0)

    inferred = {}
    for inference_type in ("upper_plausible_interval", "lower_plausible_interval"):
        model = model_construction(
            exp_set,
            num_objectives=1,
            inference_type=inference_type,
            functional_properties_list=[functional_properties],
            functional_properties_parameters=parameters,
            acceptability=None,
            discrepancy_type="confidence_region",
            upper_confidence_bounds=bounds,
            lower_confidence_bounds=bounds,
            feasibility_check_solver=None,
        )
        df = run_execution_engine(
            ExecutionConfig(inference_type=inference_type, execution="sequential"),
            model,
            gurobi_solver,
            exp_set,
        )
        assert list(df["solve_status"]) == ["completed"] * len(exp_set)
        inferred[inference_type] = df["bound"].to_numpy(dtype=float)

    assert np.allclose(inferred["upper_plausible_interval"], bounds.ravel())
    assert np.allclose(inferred["lower_plausible_interval"], bounds.ravel())

def test_flat_function_convex_and_concave_intervals_are_zero(gurobi_solver):
    _assert_flat_intervals_match_zero(gurobi_solver, ["convexity", "concavity"])

def test_flat_function_lipschitz_zero_intervals_are_zero(gurobi_solver):
    _assert_flat_intervals_match_zero(
        gurobi_solver,
        ["Lipschitz_continuity"],
        [{"lip_CST": 0.0}],
    )
