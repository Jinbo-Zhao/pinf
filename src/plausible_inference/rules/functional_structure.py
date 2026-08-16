import pyomo.environ as pyo
import pyomo.gdp as gdp
from .auxiliary_constraints import abs_logic_cases, abs_disjunction_choice
from ..utils.functional_properties_validation import validate_functional_properties_parameters
#============================================
# Rules describing the functional properties.
def constraint_convexity_rule(model,i,j,d):
    """Constraint convexity: m_i - m_j - (x_i - x_j)^T * g_i <= 0"""
    if i!=j:
        exp_set = model._exp_set_data
        vector_diff = exp_set[i, :] - exp_set[j, :]
        dot_expr = sum(vector_diff[s_idx - 1] * model.g[i, d, s_idx] for s_idx in model.S_set)
        return model.m[i,d] - model.m[j,d] - dot_expr <= 0
    else:
        return pyo.Constraint.Skip


def constraint_Lipschitz_rule(model,i,j,d,lip_CST):
    """Constraint Lipschitz: m_i - m_j - lip_CST * ||x_i - x_j||_2 <= 0"""
    if i!=j:
        exp_set = model._exp_set_data
        vector_diff = exp_set[i, :] - exp_set[j, :]
        norm_2 = pyo.sqrt(
                sum(vector_diff[k]**2 for k in range(len(vector_diff)))
                )
        return model.m[i,d] - model.m[j,d] - lip_CST * norm_2 <= 0
    else:
        return pyo.Constraint.Skip


def constraint_directional_Lipschitz_rule(model, i, j, d, lip_CST_vector):
    """Constraint directional Lipschitz: m_i - m_j <= \sum_{s} lip_CST_vector(s) * |x_is - x_js|"""
    if i == j: 
        return pyo.Constraint.Skip
    
    exp_set = model._exp_set_data

    dot_expr = sum(abs(exp_set[i, s_idx - 1] - exp_set[j, s_idx - 1]) * lip_CST_vector[s_idx - 1] 
                    for s_idx in model.S_set)
    
    return model.m[i, d] - model.m[j, d] <= dot_expr
def constraint_concavity_rule(model,i,j,d):
    """Constraint concavity: m_i - m_j - (x_i - x_j)^T * g_i >= 0"""
    if i!=j:
        exp_set = model._exp_set_data
        vector_diff = exp_set[i, :] - exp_set[j, :]
        dot_expr = sum(vector_diff[s_idx - 1] * model.g[i, d, s_idx] for s_idx in model.S_set)
        return model.m[i,d] - model.m[j,d] - dot_expr >= 0
    else:
        return pyo.Constraint.Skip

def constraint_linearity_additional_rule(model,i,j,s,d):
    """Constraint linearity additional: g_i,s = g_j,s for all i,j in K_ext_set and all s in S_set

    Index order matches how the constraint is built: index sets (K_ext, K_ext, S_set)
    are passed positionally as (i, j, s) and the objective dimension d is passed as a
    keyword (d=d). The signature must therefore be (model, i, j, s, d).
    """
    if i != j:
        return model.g[i, d, s] == model.g[j, d, s]
    else:
        return pyo.Constraint.Skip

#============================================
# The following functions are used to add functional properties constraints to the model.

def add_convexity_constraints(model,d):
    """
        add convexity constraints to the model.
    """
    constraint_name = f"Constraint_convexity_d{d}"
    new_convexity_con = pyo.Constraint(model.K_ext_set, model.K_ext_set, rule=constraint_convexity_rule, d=d)
    model.add_component(constraint_name, new_convexity_con)
    return model

def add_Lipschitz_constraints(model,d, lip_CST):
    """
        add Lipschitz constraints to the model.
    """
    constraint_name = f"Constraint_Lipschitz_d{d}"
    new_Lipschitz_con = pyo.Constraint(model.K_ext_set, model.K_ext_set, rule=constraint_Lipschitz_rule, d=d, lip_CST=lip_CST)
    model.add_component(constraint_name, new_Lipschitz_con)
    return model

def add_concavity_constraints(model,d):
    """
        add concavity constraints to the model.
    """
    constraint_name = f"Constraint_concavity_d{d}"
    new_concavity_con = pyo.Constraint(model.K_ext_set, model.K_ext_set, rule=constraint_concavity_rule, d=d)
    model.add_component(constraint_name, new_concavity_con)
    return model

def add_linearity_additional_constraints(model,d):
    # NOTE: We need to avoid the appearance of convexity or concavity, if linearity is specified.
    #It would cause the previous constraints to be convered and pop up an warning. Though, I think it is not a big issue.
    """
        add linearity additional constraints to the model.
        For linearity, we need both convexity and concavity constraints (which together imply linearity),
        plus the additional constraint that all gradients are equal: g_i,s = g_j,s for all i,j,s.
    """
    # Use dimension-specific names to avoid overwriting when multiple objectives are used
    constraint_name_convexity = f"Constraint_convexity_d{d}"
    constraint_name_concavity = f"Constraint_concavity_d{d}"
    constraint_name_linearity = f"Constraint_linearity_additional_d{d}"
    
    new_convexity_con = pyo.Constraint(model.K_ext_set, model.K_ext_set, rule=constraint_convexity_rule, d=d)
    new_concavity_con = pyo.Constraint(model.K_ext_set, model.K_ext_set, rule=constraint_concavity_rule, d=d)
    new_linearity_con = pyo.Constraint(model.K_ext_set, model.K_ext_set, model.S_set, rule=constraint_linearity_additional_rule, d=d)
    
    model.add_component(constraint_name_convexity, new_convexity_con)
    model.add_component(constraint_name_concavity, new_concavity_con)
    # NOTE: Experiments indicates that the additional constraint provides no additional information for the model, at least for one dimensional case.
    model.add_component(constraint_name_linearity, new_linearity_con)
    return model

def add_directional_Lipschitz_constraints(model, d, lip_CST_vector):
    """
        add directional Lipschitz constraints to the model.
    """

    # add the main constraint for the current dimension d
    constraint_name = f"Constraint_directional_Lipschitz_d{d}"
    
    new_directional_Lipschitz_con = pyo.Constraint(
        model.K_ext_set, 
        model.K_ext_set, 
        rule=constraint_directional_Lipschitz_rule, 
        d=d, 
        lip_CST_vector=lip_CST_vector
    )
    
    # add the constraint to the model
    model.add_component(constraint_name, new_directional_Lipschitz_con)
    
    return model
#============================================
# The following function is used to add functional properties constraints to the model.
def add_functional_properties_constraints(model, functional_properties_list, functional_properties_parameters):
    validate_functional_properties_parameters(functional_properties_list, functional_properties_parameters)
    num_objectives = len(functional_properties_list)
    for d in range(1, num_objectives + 1):
        # For each objective d, use only the functional_properties for that objective
        functional_properties = functional_properties_list[d-1]
        if 'convexity' in functional_properties:
            model = add_convexity_constraints(model, d)
        if 'Lipschitz_continuity' in functional_properties:
            lip_CST = functional_properties_parameters[d-1]['lip_CST']
            model = add_Lipschitz_constraints(model, d, lip_CST)
        if 'concavity' in functional_properties:
            model = add_concavity_constraints(model, d)
        if 'linearity' in functional_properties:
            model = add_linearity_additional_constraints(model, d)
        if 'directional_Lipschitz_continuity' in functional_properties:
            lip_CST_vector = functional_properties_parameters[d-1]['lip_CST_vector']
            model = add_directional_Lipschitz_constraints(model, d, lip_CST_vector)
    return model