import pyomo.environ as pyo
import numpy as np
#============================================
# Auxiliary constraints for the linearization of the abs operator and max operator
#============================================
def constraint_abs_pos_rule(model, d, i):
    """constraint: Y[d,i] >= sample_mean[i-1, d-1] - m[i, d]"""
    return model.Y[d, i] - model._sample_mean_data[i-1, d-1] + model.m[i, d] >= 0


def constraint_abs_neg_rule(model, d, i):
    """constraint: Y[d,i] >= m[i, d] - sample_mean[i-1, d-1]"""
    return model.Y[d, i] + model._sample_mean_data[i-1, d-1] - model.m[i, d] >= 0

def constraint_max_rule(model, d, i):
    """constraint: Z[d] >= C_constants[i-1, d-1] * Y[d, i]"""
    return model.Z[d] - model._C_constants_data[i-1, d-1] * model.Y[d, i] >= 0

# TODO(abs-linearization): The functions below (abs_logic_cases / abs_disjunction_choice)
# are the disjunctive (big-M) attempt to linearize the abs operator used by the
# directional-Lipschitz constraint (see constraint_directional_Lipschitz_rule and the
# commented-out block in functional_structure.add_directional_Lipschitz_constraints).
# They are currently NOT wired in: the directional-Lipschitz rule falls back to Python
# abs(), which is non-smooth when x0 is a decision variable (input/output pixelization).
# A proper linearization of abs is still needed here. Note: Gurobi has an excellent native
# abs (general constraint), but Pyomo cannot emit/use it even when the solver is Gurobi,
# so we must linearize abs ourselves before re-enabling this path.
def abs_logic_cases(model, i, j, s, case):
    """
    Rule function for Disjunct: |x_is - x_js|
    
    Note: In Pyomo Disjunct rule functions, the first parameter 'model' is actually
    the DisjunctData object, not the main model. We need to access the main model
    via model.model() to get model-level data like _exp_set_data.
    """
    # Only process cases where at least one index is 0 (variable case)
    # When both indices are non-zero, the difference is a constant and handled elsewhere
    if i != 0 and j != 0:
        # Both are non-zero, skip creating disjuncts (constant case)
        return
    if i == j:
        # When i==j, the diff must be zero, the constraint also provides no info
        return
    
    # Get the main model from the disjunct
    main_model = model.model()
    
    # Access exp_set data (s is 1-indexed, array is 0-indexed)
    exp_set = main_model._exp_set_data
    if i == 0:
        # x0 is a variable, j is a constant
        x0_expr = main_model.x0[s] if hasattr(main_model, 'x0') else main_model.x0_param[s]
        diff = x0_expr - exp_set[j, s - 1]
    elif j == 0:
        # i is a constant, x0 is a variable
        x0_expr = main_model.x0[s] if hasattr(main_model, 'x0') else main_model.x0_param[s]
        diff = exp_set[i, s - 1] - x0_expr
    else:
        # Should not reach here due to check above
        return
    
    # Access the v_diff_abs_x variable from the main model
    if case == 0:  # case 0: diff is positive
        model.c_pos = pyo.Constraint(expr=diff >= 0)
        model.c_val = pyo.Constraint(expr=main_model.v_diff_abs_x[i, j, s] == diff)
    else:          # case 1: diff is negative
        model.c_neg = pyo.Constraint(expr=diff <= 0)
        model.c_val = pyo.Constraint(expr=main_model.v_diff_abs_x[i, j, s] == -diff)

def abs_disjunction_choice(model, i, j, s):
    """constraint: in the positive and negative constraints, only one can be satisfied"""
    if i == j: 
        return pyo.Constraint.Skip
    # Only create disjunction when one index is 0 (variable case)
    # When both indices are non-zero, the difference is a constant and handled elsewhere
    if i != 0 and j != 0:
        return pyo.Constraint.Skip
    return [model.abs_disjuncts[i, j, s, 0], model.abs_disjuncts[i, j, s, 1]]
