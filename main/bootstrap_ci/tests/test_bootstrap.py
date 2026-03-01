"""Tests for bootstrap confidence interval estimation."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bootstrap import bootstrap_ci


def test_bootstrap_ci_returns_required_keys():
    data = [1, 2, 3, 4, 5]
    result = bootstrap_ci(data, seed=42)
    assert "mean" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert "se" in result


def test_bootstrap_ci_mean_is_correct():
    data = [2.0, 4.0, 6.0, 8.0, 10.0]
    result = bootstrap_ci(data, seed=42)
    assert abs(result["mean"] - 6.0) < 1e-10


def test_bootstrap_ci_interval_contains_true_mean():
    rng = np.random.default_rng(123)
    data = rng.normal(loc=10.0, scale=1.0, size=200)
    result = bootstrap_ci(data, n_bootstrap=5000, seed=123)
    assert result["ci_lower"] < 10.0 < result["ci_upper"]


def test_bootstrap_ci_lower_less_than_upper():
    data = np.arange(1, 51, dtype=float)
    result = bootstrap_ci(data, seed=0)
    assert result["ci_lower"] < result["ci_upper"]


def test_bootstrap_ci_se_is_positive():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = bootstrap_ci(data, seed=42)
    assert result["se"] > 0


def test_bootstrap_ci_narrow_with_large_sample():
    rng = np.random.default_rng(99)
    data = rng.normal(loc=0, scale=1, size=10000)
    result = bootstrap_ci(data, n_bootstrap=2000, seed=99)
    ci_width = result["ci_upper"] - result["ci_lower"]
    assert ci_width < 0.1
