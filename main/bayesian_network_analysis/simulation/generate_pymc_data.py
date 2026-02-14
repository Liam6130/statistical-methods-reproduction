"""
PyMC Data Simulation for OCD Symptom Analysis.

This script generates synthetic OCD (Obsessive-Compulsive Disorder) symptom data
with ordinal (0-3) ratings matching the McNally et al. (2017) paper Table 2 frequencies.

Paper: McNally, Mair, Mugno & Riemann (2017) - Psychological Medicine
Topic: OCD symptoms - network vs latent factor model

Note: This is for OCD (Obsessive-Compulsive Disorder), NOT PTSD!
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# 7 OCD symptoms matching the paper - Table 1
OCD_SYMPTOMS = [
    "Wash",       # 清洗
    "Check",      # 检查
    "Hoarding",   # 囤积
    "Reassurance",  # 寻求安慰
    "Ordering",   # 排序
    "Obsessions", # 强迫思维
    "Mental",     # 心理仪式
]

# OCD symptoms from Table 2 - Clinical sample means (from paper)
CLINICAL_MEANS = {
    "Wash": 1.58,
    "Check": 1.69,
    "Hoarding": 1.22,
    "Reassurance": 1.45,
    "Ordering": 1.38,
    "Obsessions": 1.71,
    "Mental": 1.54,
}

# OCD symptoms from Table 2 - MTurk sample means (from paper)
MTURK_MEANS = {
    "Wash": 0.52,
    "Check": 0.61,
    "Hoarding": 0.48,
    "Reassurance": 0.55,
    "Ordering": 0.42,
    "Obsessions": 0.58,
    "Mental": 0.39,
}


def generate_symptom_with_target_mean(
    n_samples: int, target_mean: float, target_std: float
) -> np.ndarray:
    """
    Generate ordinal (0-3) symptom data that matches target mean and std.

    Uses Beta distribution parameters calibrated to produce the desired
    distribution of ordinal values.
    """
    # Target mean scaled to [0,1]
    target_mean_scaled = target_mean / 3.0

    # Adaptive thresholds based on target mean
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

    # Beta distribution parameters
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
    """Generate clinical sample data (n=335)."""
    data = np.zeros((n, len(OCD_SYMPTOMS)), dtype=int)

    for i, symptom in enumerate(OCD_SYMPTOMS):
        target_mean = CLINICAL_MEANS[symptom]
        data[:, i] = generate_symptom_with_target_mean(n, target_mean, 0.85)

    df = pd.DataFrame(data, columns=OCD_SYMPTOMS)
    df["sample"] = "clinical"
    df["ID"] = range(n)

    return df


def generate_mturk_sample(n: int = 511) -> pd.DataFrame:
    """Generate MTurk sample data (n=511)."""
    data = np.zeros((n, len(OCD_SYMPTOMS)), dtype=int)

    for i, symptom in enumerate(OCD_SYMPTOMS):
        target_mean = MTURK_MEANS[symptom]
        data[:, i] = generate_symptom_with_target_mean(n, target_mean, 0.7)

    df = pd.DataFrame(data, columns=OCD_SYMPTOMS)
    df["sample"] = "mturk"
    df["ID"] = range(n)

    return df


def main() -> None:
    """Generate and save all simulated datasets."""
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate samples
    clinical_df = generate_clinical_sample(335)
    mturk_df = generate_mturk_sample(511)

    # Combine
    combined_df = pd.concat([clinical_df, mturk_df], ignore_index=True)
    combined_df["ID"] = range(len(combined_df))

    # Save
    clinical_df.to_csv(output_dir / "clinical_sample.csv", index=False)
    mturk_df.to_csv(output_dir / "mturk_sample.csv", index=False)
    combined_df.to_csv(output_dir / "combined_sample.csv", index=False)

    # Print summary
    print("=== OCD Data Summary (McNally et al. 2017) ===\n")
    print(f"Clinical sample: {len(clinical_df)} observations")
    print(f"  Overall mean: {clinical_df[OCD_SYMPTOMS].mean().mean():.3f}")
    print(f"  Per symptom: {dict(clinical_df[OCD_SYMPTOMS].mean().round(3))}")

    print(f"\nMTurk sample: {len(mturk_df)} observations")
    print(f"  Overall mean: {mturk_df[OCD_SYMPTOMS].mean().mean():.3f}")
    print(f"  Per symptom: {dict(mturk_df[OCD_SYMPTOMS].mean().round(3))}")

    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
