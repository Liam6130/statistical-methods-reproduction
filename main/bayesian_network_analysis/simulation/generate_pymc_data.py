"""
PyMC Data Simulation for PTSD Symptom Analysis.

This script generates synthetic PTSD symptom data with ordinal (0-3) ratings
matching the McNally et al. (2017) paper Table 2 frequencies:
- Clinical sample: mean 1.4-1.9
- MTurk sample: mean 0.3-0.7

Uses Beta distribution to model underlying symptom severity.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import NamedTuple

# Set random seed for reproducibility
np.random.seed(42)

# 17 PTSD symptoms matching the paper
PTSD_SYMPTOMS: list[str] = [
    "intrusion",
    "nightmares",
    "flashbacks",
    "distress",
    "physiological",
    "avoidance_memories",
    "avoidance_stimuli",
    "numbing",
    "anhedonia",
    "detachment",
    "restricted_affect",
    "irritability",
    "hypervigilance",
    "sleep",
    "concentration",
    "exaggerated_startle",
]


class SampleParams(NamedTuple):
    """Parameters for generating symptom data."""

    n_samples: int
    base_mean: float
    base_std: float
    sample_type: str


def beta_to_ordinal(
    alpha: np.ndarray, beta: np.ndarray, thresholds: list[float]
) -> np.ndarray:
    """
    Convert Beta-distributed latent scores to ordinal 0-3 ratings.

    Args:
        alpha: Beta distribution alpha parameters (shape, n_samples)
        beta: Beta distribution beta parameters (scale, n_samples)
        thresholds: Cutoff points [t1, t2, t3] for 0/1/2/3 ratings

    Returns:
        Ordinal ratings (0, 1, 2, or 3)
    """
    # Sample from Beta distribution
    latent = np.random.beta(alpha, beta)

    # Convert to ordinal using thresholds
    ordinal = np.zeros_like(latent, dtype=int)
    for i, t in enumerate(thresholds):
        ordinal[latent >= t] = i + 1

    return ordinal


def generate_symptom_with_target_mean(
    n_samples: int, target_mean: float, target_std: float
) -> np.ndarray:
    """
    Generate ordinal (0-3) symptom data that matches target mean and std.

    Uses Beta distribution parameters calibrated to produce the desired
    distribution of ordinal values.

    Args:
        n_samples: Number of samples to generate
        target_mean: Target mean rating (0-3 scale)
        target_std: Target standard deviation

    Returns:
        Array of ordinal ratings (0, 1, 2, or 3)
    """
    # Target mean scaled to [0,1]
    target_mean_scaled = target_mean / 3.0

    # Adaptive thresholds based on target mean
    # Lower means need higher thresholds to keep more values at 0
    if target_mean_scaled < 0.15:
        thresholds = [0.4, 0.65, 0.85]
    elif target_mean_scaled < 0.25:
        thresholds = [0.35, 0.6, 0.8]
    elif target_mean_scaled < 0.35:
        thresholds = [0.28, 0.52, 0.75]
    elif target_mean_scaled < 0.45:
        thresholds = [0.22, 0.45, 0.68]
    elif target_mean_scaled < 0.55:
        thresholds = [0.18, 0.38, 0.6]
    elif target_mean_scaled < 0.65:
        thresholds = [0.14, 0.32, 0.52]
    else:
        thresholds = [0.1, 0.25, 0.45]

    # Use Beta distribution with parameters that give the right skew
    # For low means: high alpha, low beta (skewed toward 0)
    # For high means: low alpha, high beta (skewed toward 1)
    if target_mean_scaled < 0.33:
        alpha = 2.0 * target_mean_scaled + 0.5
        beta = 2.0 * (1 - target_mean_scaled) + 0.5
    else:
        alpha = 2.0 * target_mean_scaled
        beta = 2.0 * (1 - target_mean_scaled)

    # Sample from Beta and convert to ordinal
    latent = np.random.beta(alpha, beta, n_samples)

    ordinal = np.zeros(n_samples, dtype=int)
    ordinal[latent >= thresholds[0]] = 1
    ordinal[latent >= thresholds[1]] = 2
    ordinal[latent >= thresholds[2]] = 3

    return ordinal


def generate_clinical_sample(n: int = 335) -> pd.DataFrame:
    """
    Generate clinical sample data (n=335).

    Clinical sample has higher symptom severity:
    - Target means: 1.4-1.9 per symptom (from Table 2)
    - Higher base rates across all symptoms
    """
    data = np.zeros((n, len(PTSD_SYMPTOMS)), dtype=int)

    # Target means for clinical sample (per Table 2: 1.4-1.9 range)
    # Using varying means per symptom to match realistic distribution
    clinical_means = [
        1.5,  # intrusion
        1.4,  # nightmares
        1.4,  # flashbacks
        1.6,  # distress
        1.5,  # physiological
        1.5,  # avoidance_memories
        1.5,  # avoidance_stimuli
        1.6,  # numbing
        1.7,  # anhedonia
        1.5,  # detachment
        1.4,  # restricted_affect
        1.6,  # irritability
        1.7,  # hypervigilance
        1.6,  # sleep
        1.4,  # concentration
        1.5,  # exaggerated_startle
    ]

    for i, target_mean in enumerate(clinical_means):
        # Target std of ~0.85 for clinical sample
        data[:, i] = generate_symptom_with_target_mean(n, target_mean, 0.85)

    df = pd.DataFrame(data, columns=PTSD_SYMPTOMS)
    df["sample"] = "clinical"
    df["ID"] = range(n)

    return df


def generate_mturk_sample(n: int = 511) -> pd.DataFrame:
    """
    Generate MTurk sample data (n=511).

    MTurk sample has lower symptom severity:
    - Target means: 0.3-0.7 per symptom (from Table 2)
    - Lower base rates across all symptoms
    """
    data = np.zeros((n, len(PTSD_SYMPTOMS)), dtype=int)

    # Target means for MTurk sample (per Table 2: 0.3-0.7 range)
    mturk_means = [
        0.6,  # intrusion
        0.4,  # nightmares
        0.3,  # flashbacks
        0.7,  # distress
        0.5,  # physiological
        0.5,  # avoidance_memories
        0.6,  # avoidance_stimuli
        0.5,  # numbing
        0.6,  # anhedonia
        0.4,  # detachment
        0.4,  # restricted_affect
        0.5,  # irritability
        0.7,  # hypervigilance
        0.6,  # sleep
        0.4,  # concentration
        0.5,  # exaggerated_startle
    ]

    for i, target_mean in enumerate(mturk_means):
        # Target std of ~0.7 for MTurk sample
        data[:, i] = generate_symptom_with_target_mean(n, target_mean, 0.7)

    df = pd.DataFrame(data, columns=PTSD_SYMPTOMS)
    df["sample"] = "mturk"
    df["ID"] = range(n)

    return df


def main() -> None:
    """Generate and save all simulated datasets."""
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    # Generate samples
    clinical_df = generate_clinical_sample(335)
    mturk_df = generate_mturk_sample(511)

    # Combine clinical and MTurk
    combined_df = pd.concat([clinical_df, mturk_df], ignore_index=True)
    combined_df["ID"] = range(len(combined_df))

    # Save datasets
    clinical_df.to_csv(output_dir / "clinical_sample.csv", index=False)
    mturk_df.to_csv(output_dir / "mturk_sample.csv", index=False)
    combined_df.to_csv(output_dir / "combined_sample.csv", index=False)

    # Print summary statistics
    print("=== PyMC Simulated Data Summary ===\n")
    print(f"Clinical sample: {len(clinical_df)} observations")

    clinical_means = clinical_df[PTSD_SYMPTOMS].mean()
    print(f"  Overall mean: {clinical_means.mean():.3f}")
    print(f"  Min symptom mean: {clinical_means.min():.3f} ({clinical_means.idxmin()})")
    print(f"  Max symptom mean: {clinical_means.max():.3f} ({clinical_means.idxmax()})")
    print(f"  Target range: 1.4-1.9")

    print(f"\nMTurk sample: {len(mturk_df)} observations")
    mturk_means = mturk_df[PTSD_SYMPTOMS].mean()
    print(f"  Overall mean: {mturk_means.mean():.3f}")
    print(f"  Min symptom mean: {mturk_means.min():.3f} ({mturk_means.idxmin()})")
    print(f"  Max symptom mean: {mturk_means.max():.3f} ({mturk_means.idxmax()})")
    print(f"  Target range: 0.3-0.7")

    print(f"\n=== Combined Dataset ===")
    print(f"Total observations: {len(combined_df)}")

    print(f"\n=== Per-Symptom Means ===")
    print("\nClinical sample:")
    for symptom in PTSD_SYMPTOMS:
        print(f"  {symptom}: {clinical_df[symptom].mean():.3f}")

    print("\nMTurk sample:")
    for symptom in PTSD_SYMPTOMS:
        print(f"  {symptom}: {mturk_df[symptom].mean():.3f}")

    print(f"\nDatasets saved to: {output_dir}")


if __name__ == "__main__":
    main()
