from typing import Optional

import pyomo.environ as pyo

def constraint_g0_zero_rule(model, s):
    """Constraint: g[0, s] == 0 for all s in S_set
    
        For single-objective optimization, the gradient of the objective function at an acceptable solution should be zero.
    """
    d=1 # as this is for single-objective optimization, d=1 is the only possible value.
    return model.g[0, d,s] == 0

def constraint_single_objective_optimality_rule(model, i):
    """Constraint single-objective optimality: -m_i + m0 <= 0"""
    d=1 # as this is for single-objective optimization, d=1 is the only possible value.
    return -model.m[i,d] + model.m[0,d] <= 0


def constraint_delta_optimality_rule(model, i, delta):
    """Constraint delta-optimality: m0 - m_i <= delta (x0 within delta of the best simulated point)."""
    d=1 # as this is for single-objective optimization, d=1 is the only possible value.
    return model.m[0,d] - model.m[i,d] <= delta


def constraint_feasibility_rule(model, d, thresholds):
    """Constraint feasibility (per objective): m[0,d] <= thresholds[d].

    Indexed over ``D_set`` so it supports both single- and multi-objective. ``thresholds``
    is a {d: value} dict forwarded as a Constraint keyword.
    """
    return model.m[0,d] <= thresholds[d]


def constraint_closeness_to_target_rule(model, d, thresholds, deltas):
    """Constraint closeness-to-target (per objective):
    thresholds[d] - deltas[d] <= m[0,d] <= thresholds[d] + deltas[d].

    Indexed over ``D_set`` (single- or multi-objective). ``thresholds`` and ``deltas`` are
    {d: value} dicts forwarded as Constraint keywords.
    """
    return (thresholds[d] - deltas[d], model.m[0,d], thresholds[d] + deltas[d])


def constraint_dimension_logic_rule(model, i, r):
    """
    Dimensional logic: (m0 - mi) * indicator <= 0
    If obj_indicator[i, r] = 1, then m[i, r] >= m[0, r] (x0 is not dominated)
    If obj_indicator[i, r] = 0, the constraint is 0 <= 0 (always satisfied).
    """
    return (model.m[0, r] - model.m[i, r]) * model.obj_indicator[i, r] <= 0

def constraint_Pareto_optimality_rule(model, i):
    #NOTE: Consider moving this rule to auxiliary_constraints.py
    """
    Pareto condition: Point i must not be dominated by point 0.
    This requires at least one objective r to satisfy m[i, r] >= m[0, r].
    Which is equivalent to x0 is not dominated by x_i
    """
    return sum(model.obj_indicator[i, r] for r in model.D_set) >= 1


def _require_acceptability_param(
    acceptability: str, acceptability_parameters: Optional[dict], key: str
) -> float:
    """Return float(acceptability_parameters[key]); raise ValueError if missing/None.

    Used for delta-optimality's scalar ``delta`` (feasibility and closeness-to-target use
    the per-objective ``_normalize_per_objective`` instead). Defensive: the same checks
    also run in validate_model_construction_core_inputs, but keep this so the rule builder
    is safe when called directly.
    """
    if not acceptability_parameters or acceptability_parameters.get(key) is None:
        raise ValueError(
            f"acceptability '{acceptability}' requires acceptability_parameters['{key}']"
        )
    return float(acceptability_parameters[key])


def _normalize_per_objective(acceptability, acceptability_parameters, key, D_set) -> dict:
    """Return a {d: float} map over ``D_set`` for ``acceptability_parameters[key]``.

    The value may be a scalar (broadcast to every objective) or a length-``len(D_set)``
    sequence (one value per objective). Used for the per-objective threshold / delta of
    feasibility and closeness-to-target.
    """
    if not acceptability_parameters or acceptability_parameters.get(key) is None:
        raise ValueError(
            f"acceptability '{acceptability}' requires acceptability_parameters['{key}']"
        )
    raw = acceptability_parameters[key]
    dims = list(D_set)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return {d: float(raw) for d in dims}
    seq = list(raw)
    if len(seq) != len(dims):
        raise ValueError(
            f"{acceptability} '{key}' sequence length {len(seq)} != num_objectives {len(dims)}"
        )
    return {d: float(v) for d, v in zip(dims, seq)}


def add_acceptability_constraints(
    model,
    acceptability: str,
    functional_properties_list: list,
    acceptability_parameters: Optional[dict] = None,
):
    """Add acceptability constraints to the model.

    ``acceptability_parameters`` carries the numeric constants for the
    threshold-based acceptabilities (single-objective, minimization, on m[0,1]):
      - ``delta-optimality``: needs ``delta`` (>=0); adds m[0,1] - m[i,1] <= delta
        for all i (x0 within delta of the best simulated objective). No gradient
        shortcut is used even under convexity/concavity.
      - ``feasibility`` (single- or multi-objective): needs ``threshold`` (scalar broadcast
        to all objectives, or a length-num_objectives sequence); adds m[0,d] <= threshold[d]
        for every objective d.
      - ``closeness-to-target`` (single- or multi-objective): needs ``threshold`` and
        ``delta`` (>=0), each a scalar (broadcast) or a length-num_objectives sequence;
        adds threshold[d] - delta[d] <= m[0,d] <= threshold[d] + delta[d] for every d.
    Numeric constants are forwarded to module-level rule functions as keyword
    arguments (the same pattern as the Lipschitz / convexity rules), so the
    constructed model stays picklable for the multiprocessing pool.
    """
    if acceptability == 'single-objective-optimality':
        # as this is for single-objective optimization, there is only one component in the functional properties list.
        functional_properties = functional_properties_list[0]
        if 'convexity' in functional_properties or 'concavity' in functional_properties:
            # for convexity or concavity, the single-objective optimality is equivalent to
            # that the gradient of the objective function at x0 should be zero.
            model.Constraint_acceptability = pyo.Constraint(model.S_set, rule=constraint_g0_zero_rule)
        else:
            # for other functional properties, call single-objective optimality rule directly.
            model.Constraint_acceptability = pyo.Constraint(model.K_set, rule=constraint_single_objective_optimality_rule)
    elif acceptability == 'delta-optimality':
        # Near-optimal: x0's objective is within delta of the best simulated point.
        # m[0,1] - m[i,1] <= delta for all i  <=>  m[0,1] <= min_i m[i,1] + delta.
        # delta == 0 recovers single-objective-optimality. No gradient shortcut.
        delta = _require_acceptability_param(acceptability, acceptability_parameters, 'delta')
        model.Constraint_acceptability = pyo.Constraint(
            model.K_set, rule=constraint_delta_optimality_rule, delta=delta
        )
    elif acceptability == 'feasibility':
        # Per-objective: m[0,d] <= threshold[d] for every objective d. threshold may be a
        # scalar (broadcast to all objectives) or a length-num_objectives sequence.
        thresholds = _normalize_per_objective(acceptability, acceptability_parameters, 'threshold', model.D_set)
        model.Constraint_acceptability = pyo.Constraint(
            model.D_set, rule=constraint_feasibility_rule, thresholds=thresholds
        )
    elif acceptability == 'closeness-to-target':
        # Per-objective band: threshold[d] - delta[d] <= m[0,d] <= threshold[d] + delta[d].
        # threshold and delta may each be a scalar (broadcast) or a length-num_objectives sequence.
        thresholds = _normalize_per_objective(acceptability, acceptability_parameters, 'threshold', model.D_set)
        deltas = _normalize_per_objective(acceptability, acceptability_parameters, 'delta', model.D_set)
        model.Constraint_acceptability = pyo.Constraint(
            model.D_set, rule=constraint_closeness_to_target_rule, thresholds=thresholds, deltas=deltas
        )
    elif acceptability == 'Pareto-optimality':
        #indicator variable for the objective function
        model.obj_indicator = pyo.Var(model.K_set, model.D_set, domain=pyo.Binary)
        # Apply the logic for each objective dimension
        model.constraint_dimension_logic = pyo.Constraint(
            model.K_set, 
            model.D_set, 
            rule=constraint_dimension_logic_rule
        )
        
        # Apply the Pareto non-domination requirement for each point i
        model.constraint_Pareto_optimality = pyo.Constraint(
            model.K_set, 
            rule=constraint_Pareto_optimality_rule
        )
    else:
        raise ValueError(f"Invalid acceptability type: {acceptability}")
    return model