"""Preprocessing and validation for functional_properties_parameters."""


def validate_functional_properties_parameters(functional_properties_list, functional_properties_parameters):
    """
    Validate functional_properties_parameters when any objective uses Lipschitz or directional_Lipschitz.
    Raises TypeError or ValueError with a clear message on failure.
    """
    n = len(functional_properties_list)
    needs_params = any(
        'Lipschitz_continuity' in fp or 'directional_Lipschitz_continuity' in fp
        for fp in functional_properties_list
    )
    if not needs_params:
        return

    if functional_properties_parameters is None:
        raise ValueError(
            "functional_properties_parameters must be provided when any objective has "
            "'Lipschitz_continuity' or 'directional_Lipschitz_continuity'"
        )
    if not isinstance(functional_properties_parameters, list):
        raise TypeError("functional_properties_parameters must be a list")
    if len(functional_properties_parameters) != n:
        raise ValueError(
            f"functional_properties_parameters must have length {n} (num_objectives), "
            f"got {len(functional_properties_parameters)}"
        )

    for d in range(n):
        fp = functional_properties_list[d]
        params_d = functional_properties_parameters[d]
        if not isinstance(params_d, dict):
            raise TypeError(
                f"functional_properties_parameters[{d}] must be a dict for objective {d+1}, "
                f"got {type(params_d).__name__}"
            )
        if 'Lipschitz_continuity' in fp and 'lip_CST' not in params_d:
            raise ValueError(
                f"functional_properties_parameters[{d}] must contain key 'lip_CST' when objective {d+1} has 'Lipschitz_continuity'"
            )
        if 'directional_Lipschitz_continuity' in fp and 'lip_CST_vector' not in params_d:
            raise ValueError(
                f"functional_properties_parameters[{d}] must contain key 'lip_CST_vector' when objective {d+1} has 'directional_Lipschitz_continuity'"
            )
