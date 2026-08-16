"""Validate arguments to ``model_construction`` before building the Pyomo model."""

from typing import Optional

ALLOWED_INFERENCE_TYPES = frozenset(
    {
        "screening",
        "upper_plausible_interval",
        "lower_plausible_interval",
        "input_pixelization",
        "output_pixelization",
    }
)
ALLOWED_FUNCTIONAL_PROPERTIES = frozenset(
    {
        "convexity",
        "Lipschitz_continuity",
        "concavity",
        # "linearity",  # dormant: implementation kept in functional_structure.py; uncomment to re-enable
        "directional_Lipschitz_continuity",
    }
)
ALLOWED_SINGLE_OBJECTIVE_ACCEPTABILITIES = frozenset(
    {
        "single-objective-optimality",
        "delta-optimality",
        "feasibility",
        "closeness-to-target",
    }
)
# feasibility and closeness-to-target are per-objective, so they work for any objective count.
ALLOWED_MULTI_OBJECTIVE_ACCEPTABILITIES = frozenset(
    {"Pareto-optimality", "feasibility", "closeness-to-target"}
)
ALLOWED_DISCREPANCY_TYPES = frozenset(
    {"norm_infinite", "norm_1", "norm_2", "CRN", "confidence_region"}
)

# Required numeric entries in acceptability_parameters, per acceptability.
ACCEPTABILITY_REQUIRED_PARAMS = {
    "delta-optimality": ("delta",),
    "feasibility": ("threshold",),
    "closeness-to-target": ("threshold", "delta"),
}
# Parameters allowed to be per-objective (scalar broadcast, or length-num_objectives sequence).
PER_OBJECTIVE_PARAMS = {
    "feasibility": ("threshold",),
    "closeness-to-target": ("threshold", "delta"),
}
# Parameters that must be non-negative when present.
NONNEGATIVE_ACCEPTABILITY_PARAMS = frozenset({"delta"})


def _is_real(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_real_sequence(value, num_objectives: int, acceptability: str, key: str) -> list:
    """Validate a per-objective parameter: a scalar (broadcast) or a length-num_objectives
    sequence of reals. Returns the resolved list of floats."""
    if _is_real(value):
        return [float(value)] * num_objectives
    seq = (
        list(value)
        if isinstance(value, (list, tuple))
        else (value.tolist() if hasattr(value, "tolist") else None)
    )
    if seq is None or len(seq) != num_objectives or not all(_is_real(x) for x in seq):
        raise ValueError(
            f"{acceptability} '{key}' must be a real number or a length-{num_objectives} "
            f"sequence of reals, got {value!r}"
        )
    return [float(x) for x in seq]


def _validate_acceptability_parameters(
    acceptability: str, acceptability_parameters: Optional[dict], num_objectives: int
) -> None:
    """Check that ``acceptability_parameters`` carries the numeric constants the
    given ``acceptability`` needs (delta-optimality / feasibility / closeness-to-target).

    feasibility's ``threshold`` is per-objective: a scalar (broadcast to all objectives)
    or a length-``num_objectives`` sequence. Acceptabilities with no numeric parameters
    ignore ``acceptability_parameters``.
    """
    required = ACCEPTABILITY_REQUIRED_PARAMS.get(acceptability, ())
    if not required:
        return
    if acceptability_parameters is None or not isinstance(acceptability_parameters, dict):
        raise ValueError(
            f"acceptability '{acceptability}' requires acceptability_parameters dict with keys {required}"
        )
    per_obj_keys = PER_OBJECTIVE_PARAMS.get(acceptability, ())
    for key in required:
        if key not in acceptability_parameters or acceptability_parameters[key] is None:
            raise ValueError(
                f"acceptability '{acceptability}' requires acceptability_parameters['{key}']"
            )
        value = acceptability_parameters[key]
        nonneg = key in NONNEGATIVE_ACCEPTABILITY_PARAMS
        if key in per_obj_keys:
            # scalar (broadcast) or one value per objective
            values = _as_real_sequence(value, num_objectives, acceptability, key)
            if nonneg and any(v < 0 for v in values):
                raise ValueError(f"{acceptability} '{key}' must be >= 0 for every objective, got {value!r}")
            continue
        if not _is_real(value):
            raise TypeError(
                f"acceptability_parameters['{key}'] must be a real number, got {type(value).__name__}"
            )
        if nonneg and value < 0:
            raise ValueError(f"acceptability_parameters['{key}'] must be >= 0, got {value}")


def validate_model_construction_core_inputs(
    num_objectives: int,
    functional_properties_list: list,
    inference_type: str,
    acceptability: Optional[str] = None,
    acceptability_parameters: Optional[dict] = None,
    discrepancy_type: Optional[str] = None,
) -> None:
    """
    Check ``num_objectives``, ``functional_properties_list``, ``inference_type``,
    ``acceptability``, ``acceptability_parameters``, and ``discrepancy_type``.

    ``discrepancy_type`` is only checked for being a legal value when provided (not
    ``None``); the requirement that it be present is enforced by the caller.

    Raises ``TypeError`` or ``ValueError`` with a clear message on failure.
    """
    if not isinstance(num_objectives, int) or num_objectives <= 0:
        raise TypeError("num_objectives must be a positive integer")

    if acceptability is not None:
        if num_objectives == 1:
            if acceptability not in ALLOWED_SINGLE_OBJECTIVE_ACCEPTABILITIES:
                raise ValueError(
                    f"Invalid acceptability type: {acceptability} for single-objective case"
                )
        else:
            if acceptability not in ALLOWED_MULTI_OBJECTIVE_ACCEPTABILITIES:
                raise ValueError(
                    f"Invalid acceptability type: {acceptability} for multi-objective case"
                )
        _validate_acceptability_parameters(acceptability, acceptability_parameters, num_objectives)

    if not isinstance(functional_properties_list, list):
        raise TypeError("functional_properties_list must be a list")

    for enum, functional_properties in enumerate(functional_properties_list):
        if not isinstance(functional_properties, (list, tuple, set)):
            raise TypeError(
                f"The ({enum + 1})th element of functional_properties_list is not a list/tuple/set"
            )

    for enum, functional_properties in enumerate(functional_properties_list):
        if not set(functional_properties).issubset(ALLOWED_FUNCTIONAL_PROPERTIES):
            invalid_properties = [
                prop for prop in functional_properties if prop not in ALLOWED_FUNCTIONAL_PROPERTIES
            ]
            raise ValueError(
                f"{enum + 1}th objective's functional properties contains invalid functional property: {invalid_properties}"
            )

    if not isinstance(inference_type, str):
        raise TypeError("inference_type must be a string")
    if acceptability is not None and not isinstance(acceptability, str):
        raise TypeError("acceptability must be a string or None")

    if inference_type not in ALLOWED_INFERENCE_TYPES:
        raise ValueError(f"Invalid inference type: {inference_type}")

    if discrepancy_type is not None and discrepancy_type not in ALLOWED_DISCREPANCY_TYPES:
        raise ValueError(f"Invalid discrepancy_type: {discrepancy_type}")

    if inference_type in ("upper_plausible_interval", "lower_plausible_interval"):
        if acceptability is not None:
            raise ValueError(
                "acceptability must be None when inference type is upper_plausible_interval or lower_plausible_interval"
            )
