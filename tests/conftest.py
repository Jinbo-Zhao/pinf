import pytest


def _require_plausibility_solver(solver_name: str, install_hint: str):
    try:
        from plausible_inference.utils.solver_config import get_plausibility_solver
    except ImportError as exc:
        pytest.fail(
            "Pyomo is required but could not be imported. "
            f"Install the project dependencies, then re-run pytest. ({exc})"
        )

    try:
        solver = get_plausibility_solver(solver_name)
    except Exception as exc:
        pytest.fail(f"{solver_name} is required but could not be created: {exc}")

    available = getattr(solver, "available", None)
    if callable(available):
        try:
            flag = bool(available(exception_flag=False))
        except TypeError:
            flag = bool(available())
        if not flag:
            pytest.fail(
                f"{solver_name} is required but Pyomo reports the solver as unavailable. "
                f"{install_hint}"
            )
    return solver


@pytest.fixture(scope="session")
def gurobi_solver():
    return _require_plausibility_solver(
        "gurobi",
        "Install/configure Gurobi and gurobipy, then re-run pytest.",
    )


@pytest.fixture(scope="session")
def scip_solver():
    return _require_plausibility_solver(
        "scip",
        "Install the SCIP executable so that `scip` is on PATH, then re-run pytest.",
    )
