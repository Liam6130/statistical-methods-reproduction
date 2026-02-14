"""
Bayesian Network Markov Check (BNMC) Analysis

This script demonstrates the Bayesian Network Markov Check method used in
McNally et al. (2017) to test whether symptoms are conditionally independent
given a latent factor.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import FactorAnalysis
from itertools import combinations
from pathlib import Path


def partial_correlation(
    data: pd.DataFrame,
    var1: str,
    var2: str,
    controls: list[str]
) -> float:
    """
    Calculate partial correlation between var1 and var2,
    controlling for other variables.

    Args:
        data: DataFrame containing the variables
        var1: First variable name
        var2: Second variable name
        controls: List of control variable names

    Returns:
        Partial correlation coefficient
    """
    if not controls:
        return data[var1].corr(data[var2])

    # Residualize var1 by regressing on control variables
    X = data[controls].values
    y1 = data[var1].values
    y2 = data[var2].values

    # Add intercept
    X_with_intercept = np.column_stack([np.ones(len(X)), X])

    # Calculate residuals using least squares
    beta1 = np.linalg.lstsq(X_with_intercept, y1, rcond=None)[0]
    residuals1 = y1 - X_with_intercept @ beta1

    beta2 = np.linalg.lstsq(X_with_intercept, y2, rcond=None)[0]
    residuals2 = y2 - X_with_intercept @ beta2

    return np.corrcoef(residuals1, residuals2)[0, 1]


def conditional_independence_test(
    data: pd.DataFrame,
    var1: str,
    var2: str,
    conditioning_vars: list[str],
    n_permutations: int = 1000
) -> dict:
    """
    Test conditional independence using permutation test.

    H0: var1 ⊥ var2 | conditioning_vars

    Args:
        data: DataFrame containing the variables
        var1: First variable name
        var2: Second variable name
        conditioning_vars: List of conditioning variable names
        n_permutations: Number of permutations for the test

    Returns:
        Dictionary with partial correlation, p-value, and permutation stats
    """
    if not conditioning_vars:
        observed_corr = data[var1].corr(data[var2])
        return {
            'partial_correlation': observed_corr,
            'p_value': 0.0,
            'permutation_mean': 0.0,
            'permutation_std': 0.0
        }

    # Calculate observed partial correlation
    obs_pc = partial_correlation(data, var1, var2, conditioning_vars)

    # Permutation test
    perm_corrs = []
    for _ in range(n_permutations):
        perm_data = data.copy()
        perm_data[var1] = np.random.permutation(data[var1])
        perm_pc = partial_correlation(perm_data, var1, var2, conditioning_vars)
        perm_corrs.append(perm_pc)

    # Two-tailed p-value
    p_value = np.mean(np.abs(perm_corrs) >= np.abs(obs_pc))

    return {
        'partial_correlation': obs_pc,
        'p_value': p_value,
        'permutation_mean': np.mean(perm_corrs),
        'permutation_std': np.std(perm_corrs)
    }


def bnmc_analysis(
    data: pd.DataFrame,
    sample_name: str,
    n_permutations: int = 1000,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Run the Bayesian Network Markov Check analysis.

    Tests all pairs of symptoms for conditional independence
    given the latent factor estimated by Factor Analysis.

    Args:
        data: DataFrame with symptom variables
        sample_name: Name for the sample being analyzed
        n_permutations: Number of permutations for independence test
        alpha: Significance level

    Returns:
        DataFrame with test results for all symptom pairs
    """
    print(f"\n{'='*60}")
    print(f"BNMC Analysis for {sample_name}")
    print(f"{'='*60}")
    print(f"Sample size: {len(data)}")

    # Step 1: Identify symptom columns (exclude non-symptom columns)
    exclude_cols = ['sample', 'ID', 'latent_factor']
    symptoms = [col for col in data.columns if col not in exclude_cols]
    print(f"Number of symptoms: {len(symptoms)}")

    # Step 2: Fit Factor Analysis to estimate latent factor
    print("\n1. Fitting Factor Analysis (n_factors=1)...")
    symptom_data = data[symptoms].values

    fa = FactorAnalysis(n_components=1, random_state=42)
    latent_scores = fa.fit_transform(symptom_data)

    # Add latent factor to data
    data_with_latent = data.copy()
    data_with_latent['latent_factor'] = latent_scores.flatten()

    # Get factor loadings
    loadings = fa.components_.T.flatten()
    print("\nFactor loadings:")
    for sym, load in zip(symptoms, loadings):
        print(f"  {sym}: {load:.3f}")

    # Step 3: Test all symptom pairs
    print(f"\n2. Testing conditional independence for all symptom pairs...")
    symptom_pairs = list(combinations(symptoms, 2))
    print(f"   Total pairs to test: {len(symptom_pairs)}")

    results = []
    for i, (var1, var2) in enumerate(symptom_pairs):
        if (i + 1) % 20 == 0:
            print(f"   Progress: {i + 1}/{len(symptom_pairs)}")

        # Marginal independence test (no conditioning)
        marginal = conditional_independence_test(
            data_with_latent, var1, var2, [], n_permutations
        )

        # Conditional independence test (given latent factor)
        conditional = conditional_independence_test(
            data_with_latent, var1, var2, ['latent_factor'], n_permutations
        )

        # Determine if conditional independence holds
        cond_independent = conditional['p_value'] > alpha

        results.append({
            'var1': var1,
            'var2': var2,
            'marginal_corr': marginal['partial_correlation'],
            'marginal_p': marginal['p_value'],
            'marginal_sig': marginal['p_value'] < alpha,
            'conditional_corr': conditional['partial_correlation'],
            'conditional_p': conditional['p_value'],
            'conditional_sig': conditional['p_value'] < alpha,
            'cond_independent': cond_independent
        })

    results_df = pd.DataFrame(results)

    # Summary statistics
    n_marginal_sig = results_df['marginal_sig'].sum()
    n_conditional_sig = results_df['conditional_sig'].sum()
    n_cond_independent = results_df['cond_independent'].sum()

    print(f"\n3. Results Summary:")
    print(f"   - Marginal correlations significant: {n_marginal_sig}/{len(results_df)}")
    print(f"   - Conditional correlations significant: {n_conditional_sig}/{len(results_df)}")
    print(f"   - Pairs conditionally independent: {n_cond_independent}/{len(results_df)}")

    return results_df


def interpret_results(results: pd.DataFrame) -> str:
    """
    Interpret the BNMC results.

    Args:
        results: DataFrame with BNMC test results

    Returns:
        String with interpretation
    """
    interpretation = """
======================================================================
Interpretation of BNMC Results
======================================================================

The Bayesian Network Markov Check tests whether symptom pairs are
conditionally independent given a latent factor.

If symptoms are CONDITIONALLY INDEPENDENT given the latent factor:
  - Marginal correlations are significant (symptoms correlate)
  - Conditional correlations are NON-significant (p > 0.05)

This supports the LATENT FACTOR MODEL:
  latent --> symptom1
  latent --> symptom2
  (symptom1 ⊥ symptom2 | latent)

If symptoms remain correlated after conditioning:
  - This supports the NETWORK MODEL:
    symptom1 <--> symptom2 (direct causal link)

======================================================================
Results Summary
======================================================================
"""
    for _, row in results.iterrows():
        interpretation += (
            f"\n{row['var1']} - {row['var2']}:\n"
            f"  Marginal:     r={row['marginal_corr']:+.3f}, p={row['marginal_p']:.4f}\n"
            f"  Conditional:   r={row['conditional_corr']:+.3f}, p={row['conditional_p']:.4f}\n"
            f"  Cond. Indep.: {'Yes' if row['cond_independent'] else 'No'}\n"
        )

    return interpretation


if __name__ == '__main__':
    # Load simulated data
    data_dir = Path(__file__).parent.parent / 'simulation' / 'data'
    clinical = pd.read_csv(data_dir / 'clinical_sample.csv')

    print("Bayesian Network Markov Check (BNMC) Analysis")
    print("=" * 50)

    # Run BNMC analysis
    results = bnmc_analysis(clinical, "Clinical Sample", n_permutations=1000)

    # Print results table
    print("\n" + "=" * 60)
    print("Detailed Results")
    print("=" * 60)
    print(results.to_string(index=False))

    # Print interpretation
    print(interpret_results(results))

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    results.to_csv(output_dir / 'bnmc_results.csv', index=False)
    print(f"\nResults saved to: {output_dir / 'bnmc_results.csv'}")
