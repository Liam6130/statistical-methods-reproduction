"""Bootstrap confidence interval estimation.

Implements the non-parametric bootstrap method for estimating
confidence intervals of the sample mean, following Efron & Tibshirani (1993).
"""

import numpy as np


def bootstrap_ci(data, n_bootstrap=1000, confidence=0.95, seed=None):
    """Compute bootstrap confidence interval for the mean.

    Args:
        data: array-like of observed values.
        n_bootstrap: number of bootstrap resamples.
        confidence: confidence level (e.g. 0.95 for 95% CI).
        seed: random seed for reproducibility.

    Returns:
        dict with keys 'mean', 'ci_lower', 'ci_upper', 'se'.
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=float)
    n = len(data)

    bootstrap_means = np.array([
        rng.choice(data, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])

    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return {
        "mean": float(np.mean(data)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "se": float(np.std(bootstrap_means, ddof=1)),
    }


if __name__ == "__main__":
    np.random.seed(42)
    sample_data = np.random.normal(loc=5.0, scale=2.0, size=100)
    result = bootstrap_ci(sample_data, n_bootstrap=2000, seed=42)
    print(f"Sample mean: {result['mean']:.3f}")
    print(f"95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
    print(f"Bootstrap SE: {result['se']:.3f}")
