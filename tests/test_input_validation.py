import pytest

from plausible_inference.utils.model_construction_inputs import (
    validate_model_construction_core_inputs,
)
from plausible_inference.utils.functional_properties_validation import (
    validate_functional_properties_parameters,
)

def test_rejects_invalid_inference_type():
    with pytest.raises(ValueError, match="Invalid inference type"):
        validate_model_construction_core_inputs(
            1, [["convexity"]], "not-a-inference-type", acceptability="single-objective-optimality"
        )

def test_rejects_invalid_single_objective_acceptability():
    with pytest.raises(ValueError, match="Invalid acceptability type"):
        validate_model_construction_core_inputs(
            1, [["convexity"]], "screening", acceptability="Pareto-optimality"
        )

def test_rejects_invalid_multi_objective_acceptability():
    with pytest.raises(ValueError, match="Invalid acceptability type"):
        validate_model_construction_core_inputs(
            2,
            [["convexity"], ["convexity"]],
            "screening",
            acceptability="delta-optimality",
        )

def test_rejects_invalid_discrepancy_type():
    with pytest.raises(ValueError, match="Invalid discrepancy_type"):
        validate_model_construction_core_inputs(
            1,
            [["convexity"]],
            "screening",
            acceptability="single-objective-optimality",
            discrepancy_type="not-a-discrepancy-type",
        )

def test_rejects_invalid_functional_property():
    with pytest.raises(ValueError, match="invalid functional property"):
        validate_model_construction_core_inputs(
            1, [["not-a-functional-property"]], "screening", acceptability="single-objective-optimality"
        )

def test_acceptability_requires_parameters():
    with pytest.raises(ValueError, match="requires acceptability_parameters"):
        validate_model_construction_core_inputs(
            1, [["convexity"]], "screening", acceptability="delta-optimality"
        )

def test_feasibility_requires_threshold():
    with pytest.raises(ValueError, match="threshold"):
        validate_model_construction_core_inputs(
            1,
            [["convexity"]],
            "screening",
            acceptability="feasibility",
            acceptability_parameters={},
        )

def test_closeness_to_target_requires_threshold_and_delta():
    with pytest.raises(ValueError, match="threshold"):
        validate_model_construction_core_inputs(
            1,
            [["convexity"]],
            "screening",
            acceptability="closeness-to-target",
            acceptability_parameters={},
        )
    with pytest.raises(ValueError, match="delta"):
        validate_model_construction_core_inputs(
            1,
            [["convexity"]],
            "screening",
            acceptability="closeness-to-target",
            acceptability_parameters={"threshold": 1.5},
        )
    with pytest.raises(ValueError, match="threshold"):
        validate_model_construction_core_inputs(
            1,
            [["convexity"]],
            "screening",
            acceptability="closeness-to-target",
            acceptability_parameters={"delta": 0.5},
        )

def test_delta_must_be_nonnegative():
    with pytest.raises(ValueError, match="must be >= 0"):
        validate_model_construction_core_inputs(
            1,
            [["convexity"]],
            "screening",
            acceptability="delta-optimality",
            acceptability_parameters={"delta": -0.1},
        )

def test_interval_inference_forbids_acceptability():
    with pytest.raises(ValueError, match="acceptability must be None"):
        validate_model_construction_core_inputs(
            1,
            [["convexity"]],
            "upper_plausible_interval",
            acceptability="single-objective-optimality",
        )

def test_num_objectives_must_be_positive_int():
    with pytest.raises(TypeError, match="num_objectives"):
        validate_model_construction_core_inputs(
            0, [["convexity"]], "screening", acceptability="single-objective-optimality"
        )

def test_lipschitz_requires_parameters():
    with pytest.raises(ValueError, match="functional_properties_parameters must be provided"):
        validate_functional_properties_parameters([["Lipschitz_continuity"]], None)

def test_lipschitz_requires_lip_CST_key():
    with pytest.raises(ValueError, match="lip_CST"):
        validate_functional_properties_parameters(
            [["Lipschitz_continuity"]], [{}]
        )
