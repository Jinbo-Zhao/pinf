import numpy as np

from plausible_inference.execution_engine import ExecutionConfig, run_execution_engine
from plausible_inference.model_construction import model_construction

def _screen(gurobi_solver, execution, candidates):
    exp_set = np.array([[0.0], [1.0], [2.0]])
    model = model_construction(
        exp_set,
        num_objectives=1,
        inference_type="screening",
        functional_properties_list=[["convexity", "concavity"]],
        acceptability="feasibility",
        acceptability_parameters={"threshold": 1.5},
        discrepancy_type="confidence_region",
        upper_confidence_bounds=exp_set,
        lower_confidence_bounds=exp_set,
        feasibility_check_solver=None,
    )
    df = run_execution_engine(
        ExecutionConfig(
            inference_type="screening",
            execution=execution,
            pool_processes=2,
        ),
        model,
        gurobi_solver,
        candidates,
    )
    df = df.sort_values("idx")
    assert list(df["solve_status"]) == ["completed"] * len(candidates)
    return list(df["result"])

def test_parallel_screening_matches_sequential(gurobi_solver):
    candidates = np.linspace(-4.0, 8.0, 25).reshape(-1, 1)
    expected = ["returned" if x <= 1.5 else "screened_out" for x in candidates[:, 0]]
    assert _screen(gurobi_solver, "sequential", candidates) == expected
    assert _screen(gurobi_solver, "parallel", candidates) == expected
