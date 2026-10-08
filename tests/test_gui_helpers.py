from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import GUI.gui_helpers as gh

def test_project_paths():
    root = gh.project_root()
    assert root == Path(__file__).resolve().parents[1]
    assert (root / "pyproject.toml").is_file()
    seeded = gh.seeded_data_dir()
    assert seeded == root / "GUI" / "seeded_data"
    assert seeded.is_dir()

def test_ensure_package_on_path():
    root = gh.ensure_package_on_path()
    assert root == gh.project_root()
    assert str(root / "src") in sys.path

def test_simulation_template_stats_roundtrip():
    csv_text = gh.simulation_template_csv_stats(2, 1, num_example_rows=2)
    cleaned = gh._strip_comment_lines(csv_text)
    df = pd.read_csv(io.StringIO(cleaned))
    assert list(df.columns) == ["solution_id", "x_1", "x_2", "sample_size", "mean_1", "var_1"]
    assert len(df) == 2
    parsed = gh.parse_simulation_dataframe(df, num_decision_dims=2, num_objectives=1)
    np.testing.assert_array_equal(parsed.exp_set, df[["x_1", "x_2"]].to_numpy(dtype=float))
    np.testing.assert_array_equal(parsed.sample_mean, df[["mean_1"]].to_numpy(dtype=float))
    np.testing.assert_array_equal(parsed.sample_var, df[["var_1"]].to_numpy(dtype=float))
    np.testing.assert_array_equal(parsed.sample_size, df[["sample_size"]].to_numpy(dtype=float))
    assert parsed.lower_confidence_bounds is None
    assert parsed.upper_confidence_bounds is None
    assert parsed.uses_bounds is False

def test_simulation_template_bounds_roundtrip():
    csv_text = gh.simulation_template_csv_bounds(2, 1, num_example_rows=2)
    cleaned = gh._strip_comment_lines(csv_text)
    df = pd.read_csv(io.StringIO(cleaned))
    assert list(df.columns) == ["solution_id", "x_1", "x_2", "lower_1", "upper_1"]
    assert len(df) == 2
    parsed = gh.parse_simulation_dataframe(df, num_decision_dims=2, num_objectives=1)
    np.testing.assert_array_equal(parsed.exp_set, df[["x_1", "x_2"]].to_numpy(dtype=float))
    np.testing.assert_array_equal(parsed.lower_confidence_bounds, df[["lower_1"]].to_numpy(dtype=float))
    np.testing.assert_array_equal(parsed.upper_confidence_bounds, df[["upper_1"]].to_numpy(dtype=float))
    assert parsed.sample_mean is None
    assert parsed.sample_var is None
    assert parsed.sample_size is None
    assert parsed.uses_bounds is True

def test_build_input_pixel_grid_3x3():
    lb, ub = gh.build_input_pixel_grid([(0.0, 3.0), (0.0, 3.0)], [3, 3])
    expected_lb = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [2.0, 2.0],
        ]
    )
    expected_ub = expected_lb + 1.0
    np.testing.assert_array_equal(lb, expected_lb)
    np.testing.assert_array_equal(ub, expected_ub)

def test_build_output_pixel_grid_3x3():
    lb, ub = gh.build_output_pixel_grid([(0.0, 1.0), (0.0, 1.0)], [3, 3])
    edges = np.linspace(0.0, 1.0, 4)
    expected_lb = np.array([[edges[i], edges[j]] for i in range(3) for j in range(3)])
    expected_ub = expected_lb + (edges[1] - edges[0])
    np.testing.assert_array_equal(lb, expected_lb)
    np.testing.assert_array_equal(ub, expected_ub)

def test_unit_box_A_b():
    A, b = gh.unit_box_A_b(2)
    expected_A = np.array(
        [
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    expected_b = np.array([0.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(A, expected_A)
    np.testing.assert_array_equal(b, expected_b)

def test_template_filenames():
    assert gh.template_filename_stats(1, 2) == "pi_simulation_template_stats_m1_s2.csv"
    assert gh.template_filename_bounds(1, 2) == "pi_simulation_template_bounds_m1_s2.csv"
