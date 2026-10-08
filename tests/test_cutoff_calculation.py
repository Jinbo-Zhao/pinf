import numpy as np
import pytest

from plausible_inference.utils.cutoff_calculation import calc_cutoff

def test_crn_closed_form():
    k, n, alpha = 2, 50.0, 0.05
    got = calc_cutoff(k, n, alpha, "CRN")
    from scipy.stats import f

    expected = k * (n - 1) / (n - k) * f.ppf(1 - alpha, k, n - k)
    assert np.isclose(got, expected)

def test_ellinf_reproducible_with_random_state():
    a = calc_cutoff(2, np.array([40.0, 40.0]), 0.05, "ellinf", random_state=0)
    b = calc_cutoff(2, np.array([40.0, 40.0]), 0.05, "ellinf", random_state=0)
    assert np.isclose(a, b)
    assert a > 0

def test_n_vec_length_mismatch_raises():
    with pytest.raises(ValueError, match="size of n_vec"):
        calc_cutoff(2, np.array([10.0]), 0.05, "ell1", random_state=0)

def test_invalid_discrepancy_raises():
    with pytest.raises(ValueError, match="valid discrepancy"):
        calc_cutoff(2, np.array([10.0, 10.0]), 0.05, "not-a-discrepancy-type", random_state=0)
