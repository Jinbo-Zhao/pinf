import pyomo.environ as pyo
import numpy as np
from .auxiliary_constraints import constraint_abs_pos_rule, constraint_abs_neg_rule, constraint_max_rule
from ..utils.cutoff_calculation import calc_cutoff


# ============================================================
# Input validation
# ============================================================

def _validate_discrepancy_inputs(sample_mean, sample_var, sample_size, k, d, discrepancy_type):
    """Validate and normalize discrepancy inputs to canonical shapes.

    For non-CRN discrepancies (norm_infinite, norm_1, norm_2):
        sample_mean : (k, d)
        sample_var  : (k, d)
        sample_size : (k, d)  or scalar / (k,) when d=1 (auto-promoted)

    For CRN:
        sample_mean : (k, d)
        sample_var  : (k, k, d)  or (k, k) when d=1 or shared across objectives
        sample_size : (d,)       or scalar (broadcast to all objectives)

    Raises ValueError / TypeError on shape mismatch.
    """
    sample_mean = np.asarray(sample_mean, dtype=float)
    if sample_mean.shape == (k,):
        if d != 1:
            raise ValueError(
                f"sample_mean has shape ({k},) but num_objectives={d}; expected ({k}, {d})"
            )
        sample_mean = sample_mean.reshape(k, 1)
    elif sample_mean.shape != (k, d):
        raise ValueError(
            f"sample_mean must have shape ({k},) for single-objective or ({k}, {d}), got {sample_mean.shape}"
        )

    if discrepancy_type == 'CRN':
        sample_var = np.asarray(sample_var, dtype=float)
        if sample_var.shape == (k, k):
            sample_var = np.repeat(sample_var[:, :, np.newaxis], d, axis=2)
        elif sample_var.shape != (k, k, d):
            raise ValueError(
                f"For CRN, sample_var must have shape ({k}, {k}) or ({k}, {k}, {d}), got {sample_var.shape}"
            )
        sample_size = np.asarray(sample_size, dtype=float).flatten()
        if sample_size.size == 1:
            sample_size = np.full(d, float(sample_size[0]))
        elif sample_size.shape != (d,):
            raise ValueError(
                f"For CRN, sample_size must be scalar or shape ({d},), got {sample_size.shape}"
            )
    else:
        sample_var = np.asarray(sample_var, dtype=float)
        if sample_var.shape == (k,):
            if d != 1:
                raise ValueError(
                    f"sample_var has shape ({k},) but num_objectives={d}; expected ({k}, {d})"
                )
            sample_var = sample_var.reshape(k, 1)
        elif sample_var.shape != (k, d):
            raise ValueError(
                f"sample_var must have shape ({k},) for single-objective or ({k}, {d}), got {sample_var.shape}"
            )

        # Variances drive C_constants = sqrt(n)/sqrt(var); zero/negative/non-finite
        # values would silently produce inf/nan in the model, so reject them up front.
        if not np.all(np.isfinite(sample_var)) or np.any(sample_var <= 0):
            raise ValueError(
                "sample_var must be finite and strictly positive (variance > 0) for "
                f"{discrepancy_type} discrepancy"
            )

        sample_size = np.asarray(sample_size, dtype=float)
        if sample_size.size == 1:
            sample_size = np.full((k, d), float(sample_size.flat[0]))
        elif sample_size.shape in ((k,), (k, 1)):
            sample_size = np.broadcast_to(sample_size.reshape(k, 1), (k, d)).copy()
        elif sample_size.shape != (k, d):
            raise ValueError(
                f"sample_size must be scalar, shape ({k},), ({k}, 1), or ({k}, {d}), got {sample_size.shape}"
            )

    return sample_mean, sample_var, sample_size


# ============================================================
# Discrepancy setup functions (unified for single and multi-objective)
# All expect validated (k, d) arrays; use D_set-indexed model components.
# model.m[i, d], model.Y[d, i], model.Z[d], model.discrepancy[d].
# ============================================================

def set_discrepancy_infinite(model, sample_mean, sample_var, sample_size):
    """Norm-infinite discrepancy: max_i C_i |m[i,d] - mu[i,d]| per objective d.

    Args:
        sample_mean : (k, d)
        sample_var  : (k, d)
        sample_size : (k, d)
    """
    C_constants = np.sqrt(sample_size) / np.sqrt(sample_var)
    model._sample_mean_data = sample_mean
    model._C_constants_data = C_constants
    model.Z = pyo.Var(model.D_set, domain=pyo.NonNegativeReals)
    model.Y = pyo.Var(model.D_set, model.K_set, domain=pyo.NonNegativeReals)
    model.C_abs_pos = pyo.Constraint(model.D_set, model.K_set, rule=constraint_abs_pos_rule)
    model.C_abs_neg = pyo.Constraint(model.D_set, model.K_set, rule=constraint_abs_neg_rule)
    model.C_max = pyo.Constraint(model.D_set, model.K_set, rule=constraint_max_rule)
    model.discrepancy = pyo.Expression(model.D_set, rule=_discrepancy_infinite_rule)


def set_discrepancy_norm_1(model, sample_mean, sample_var, sample_size):
    """Norm-1 discrepancy: sum_i C_i |m[i,d] - mu[i,d]| per objective d.

    Args:
        sample_mean : (k, d)
        sample_var  : (k, d)
        sample_size : (k, d)
    """
    C_constants = np.sqrt(sample_size) / np.sqrt(sample_var)
    model._sample_mean_data = sample_mean
    model._C_constants_data = C_constants
    model.Y = pyo.Var(model.D_set, model.K_set, domain=pyo.NonNegativeReals)
    model.C_abs_pos = pyo.Constraint(model.D_set, model.K_set, rule=constraint_abs_pos_rule)
    model.C_abs_neg = pyo.Constraint(model.D_set, model.K_set, rule=constraint_abs_neg_rule)
    model.discrepancy = pyo.Expression(model.D_set, rule=_discrepancy_norm_1_rule)


def set_discrepancy_norm_2(model, sample_mean, sample_var, sample_size):
    """Norm-2 discrepancy: sum_i C_i (m[i,d] - mu[i,d])^2 per objective d.

    Args:
        sample_mean : (k, d)
        sample_var  : (k, d)
        sample_size : (k, d)
    """
    C_constants = sample_size / sample_var
    model._sample_mean_data = sample_mean
    model._C_constants_data = C_constants
    model.discrepancy = pyo.Expression(model.D_set, rule=_discrepancy_norm_2_rule)


def set_discrepancy_CRN(model, sample_mean, sample_var, sample_size):
    """CRN discrepancy per objective d: n_d * (mu_d - m_d)^T Sigma_d^{-1} (mu_d - m_d).

    Args:
        sample_mean : (k, d)
        sample_var  : (k, k, d)  per-objective covariance matrices
        sample_size : (d,)       scalar sample size per objective
    """
    Sigma_inv = np.zeros_like(sample_var)
    for d_idx in range(sample_var.shape[2]):
        Sigma = sample_var[:, :, d_idx]
        Sigma_inv[:, :, d_idx] = np.linalg.solve(Sigma, np.eye(Sigma.shape[0]))
    model._sample_mean_data = sample_mean
    model._Sigma_inv_data = Sigma_inv
    model._sample_size_data = sample_size
    model.discrepancy = pyo.Expression(model.D_set, rule=_discrepancy_CRN_rule)


# ============================================================
# Confidence region constraints for pre-computed bounds
# ============================================================

# ============================================================
# Module-level rule functions for discrepancy expressions.
# Must be defined at module scope (not as lambdas) so that they
# can be pickled by multiprocessing.Pool.
# ============================================================

def _discrepancy_infinite_rule(model, d):
    return model.Z[d]


def _discrepancy_norm_1_rule(model, d):
    return pyo.quicksum(
        model._C_constants_data[i-1, d-1] * model.Y[d, i] for i in model.K_set
    )


def _discrepancy_norm_2_rule(model, d):
    return pyo.quicksum(
        model._C_constants_data[i-1, d-1] * (model.m[i, d] - model._sample_mean_data[i-1, d-1])**2
        for i in model.K_set
    )


def _discrepancy_CRN_rule(model, d):
    return model._sample_size_data[d-1] * pyo.quicksum(
        (model._sample_mean_data[i-1, d-1] - model.m[i, d])
        * model._Sigma_inv_data[i-1, j-1, d-1]
        * (model._sample_mean_data[j-1, d-1] - model.m[j, d])
        for i in model.K_set for j in model.K_set
    )


def _confidence_region_cutoff_rule(model, obj_d):
    return model.discrepancy[obj_d] <= model._cutoff_data[obj_d]


def constraint_confidence_region_upper_rule(model, i, d):
    """constraint: m[i,d] <= upper_confidence_bounds[i-1,d-1]
    Only applied to K_set (experimental points), not x0 (index 0).
    """
    return model.m[i, d] <= model._upper_confidence_bounds_data[i-1, d-1]

def constraint_confidence_region_lower_rule(model, i, d):
    """constraint: m[i,d] >= lower_confidence_bounds[i-1,d-1]
    Only applied to K_set (experimental points), not x0 (index 0).
    """
    return model.m[i, d] >= model._lower_confidence_bounds_data[i-1, d-1]


# ============================================================
# Main entry point
# ============================================================

def add_confidence_region_constraints(model, discrepancy_type, sample_mean=None, sample_var=None,
                                       sample_size=None, confidence_level=None,
                                       upper_confidence_bounds=None, lower_confidence_bounds=None,
                                       num_objectives=1):
    """Add confidence region constraints with Bonferroni correction for multi-objective.

    For discrepancy_type in {'norm_infinite', 'norm_1', 'norm_2', 'CRN'}:
      - Input shapes are validated upfront; errors are raised on mismatch.
      - Bonferroni correction: alpha_d = (1 - confidence_level) / num_objectives per objective.
      - The same discrepancy setup applies for both single and multi-objective.

    For discrepancy_type == 'confidence_region':
      - Pre-computed bounds upper_confidence_bounds and lower_confidence_bounds are used directly.
      - Applies to K_set x D_set (experimental points, all objectives).
    """
    k = len(model.K_set)
    d = num_objectives
    discrep_str_map = {'norm_infinite': 'ellinf', 'norm_1': 'ell1', 'norm_2': 'ell2', 'CRN': 'CRN'}

    if discrepancy_type in discrep_str_map:
        # alpha_d = (1 - confidence_level)/d is fed to a quantile; an out-of-range
        # level would silently yield a meaningless cutoff, so guard it here.
        if confidence_level is None or not (0 < confidence_level < 1):
            raise ValueError(
                f"confidence_level must be in the open interval (0, 1), got {confidence_level!r}"
            )
        sample_mean, sample_var, sample_size = _validate_discrepancy_inputs(
            sample_mean, sample_var, sample_size, k, d, discrepancy_type
        )

        # Bonferroni: split total alpha equally across objectives
        alpha_d = (1 - confidence_level) / d
        discrep_str = discrep_str_map[discrepancy_type]

        # Set up discrepancy variables and expressions (single code path for all objectives)
        if discrepancy_type == 'norm_infinite':
            set_discrepancy_infinite(model, sample_mean, sample_var, sample_size)
        elif discrepancy_type == 'norm_1':
            set_discrepancy_norm_1(model, sample_mean, sample_var, sample_size)
        elif discrepancy_type == 'norm_2':
            set_discrepancy_norm_2(model, sample_mean, sample_var, sample_size)
        else:
            set_discrepancy_CRN(model, sample_mean, sample_var, sample_size)

        # Precompute per-objective cutoffs (Bonferroni: same alpha_d for each d)
        cutoff_values = {}
        for obj_d in range(1, d + 1):
            ss_d = sample_size[obj_d-1] if discrepancy_type == 'CRN' else sample_size[:, obj_d-1]
            cutoff_values[obj_d] = calc_cutoff(k, ss_d, alpha_d, discrep_str)
        model._cutoff_data = cutoff_values

        model.confidence_region = pyo.Constraint(
            model.D_set, rule=_confidence_region_cutoff_rule
        )

    elif discrepancy_type == 'confidence_region':
        if upper_confidence_bounds is None or lower_confidence_bounds is None:
            raise ValueError(
                "upper_confidence_bounds and lower_confidence_bounds must be provided "
                "when discrepancy_type is 'confidence_region'"
            )
        model._upper_confidence_bounds_data = upper_confidence_bounds
        model._lower_confidence_bounds_data = lower_confidence_bounds
        model.C_confidence_region_upper = pyo.Constraint(
            model.K_set, model.D_set, rule=constraint_confidence_region_upper_rule
        )
        model.C_confidence_region_lower = pyo.Constraint(
            model.K_set, model.D_set, rule=constraint_confidence_region_lower_rule
        )
    else:
        raise ValueError(f"Invalid discrepancy type: {discrepancy_type}")

    return model
