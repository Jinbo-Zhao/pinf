import numpy as np
import pytest

from plausible_inference.execution_engine import ExecutionConfig, run_execution_engine
from plausible_inference.model_construction import model_construction


def _convex_model(values, inference_type, feasibility_check_solver):
    values = np.asarray(values, dtype=float)
    return model_construction(
        np.array([[-1.0], [0.0], [1.0]]),
        num_objectives=1,
        inference_type=inference_type,
        functional_properties_list=[["convexity"]],
        acceptability=None,
        discrepancy_type="confidence_region",
        upper_confidence_bounds=values,
        lower_confidence_bounds=values,
        feasibility_check_solver=feasibility_check_solver,
    )


def _bound(gurobi_solver, model, inference_type, candidate):
    df = run_execution_engine(
        ExecutionConfig(inference_type=inference_type, execution="sequential"),
        model,
        gurobi_solver,
        np.asarray(candidate, dtype=float),
    )
    return df


def test_structural_check_rejects_a_peak_and_keeps_a_valley(gurobi_solver):
    peak = np.array([[0.0], [1.0], [0.0]])
    valley = np.array([[1.0], [0.0], [1.0]])
    with pytest.raises(ValueError, match="infeasible under functional-property and confidence-region"):
        _convex_model(peak, "lower_plausible_interval", "gurobi")

    skipped = _convex_model(peak, "lower_plausible_interval", None)
    df = _bound(gurobi_solver, skipped, "lower_plausible_interval", [[0.5]])
    assert list(df["solve_status"]) == ["error"]

    for inference_type, expected in (
        ("lower_plausible_interval", -0.5),
        ("upper_plausible_interval", 0.5),
    ):
        model = _convex_model(valley, inference_type, "gurobi")
        df = _bound(gurobi_solver, model, inference_type, [[0.5]])
        assert list(df["solve_status"]) == ["completed"]
        assert float(df.loc[0, "bound"]) == expected


def test_interval_objective_index_selects_the_pinned_line(gurobi_solver):
    exp_set = np.array([[0.0], [1.0]])
    values = np.array([[0.0, 2.0], [1.0, 4.0]])
    for index, pinned in ((1, 0.5), (2, 3.0)):
        for inference_type in ("lower_plausible_interval", "upper_plausible_interval"):
            model = model_construction(
                exp_set,
                num_objectives=2,
                inference_type=inference_type,
                functional_properties_list=[["convexity", "concavity"], ["convexity", "concavity"]],
                acceptability=None,
                discrepancy_type="confidence_region",
                upper_confidence_bounds=values,
                lower_confidence_bounds=values,
                feasibility_check_solver="gurobi",
                interval_objective_index=index,
            )
            df = _bound(gurobi_solver, model, inference_type, [[0.5]])
            assert list(df["solve_status"]) == ["completed"]
            assert float(df.loc[0, "bound"]) == pinned


def test_discrepancy_type_required():
    with pytest.raises(ValueError, match="discrepancy_type must be provided"):
        model_construction(
            np.array([[-1.0], [1.0]]),
            num_objectives=1,
            inference_type="screening",
            functional_properties_list=[["convexity"]],
            acceptability="single-objective-optimality",
            feasibility_check_solver=None,
        )
