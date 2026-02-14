"""
Simulated data generation based on McNally et al. (2017) paper.
This script generates synthetic PTSD symptom data that mirrors the structure
of the original study (17 symptoms, two samples).
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# PTSD symptoms from Table 1 (I saw these in the paper)
PTSD_SYMPTOMS = [
    "Intrusion",
    "Nightmares",
    "Flashbacks",
    "Distress",
    "Physiological",
    "Avoidance",
    "Memories",
    "Avoidance2",
    "Numbing",
    "Anhedonia",
    "Detachment",
    "Numbing3",
    "Irritability",
    "Hypervigilance",
    "Sleep",
    "Concentration",
    "Exaggerated"
]


def generate_latent_factor(n_samples: int) -> np.ndarray:
    """
    Generate a latent factor (underlying PTSD severity).
    This represents the 'common cause' model where all symptoms
    are caused by a single latent variable.
    """
    return np.random.normal(0, 1, n_samples)


def generate_symptom_data_latent(
    n_samples: int,
    latent_factor: np.ndarray,
    loadings: np.ndarray | None = None
) -> np.ndarray:
    """
    Generate symptom data from a latent factor model.
    This is what McNally et al. argue is the true data generating process.

    P(S1, S2, ..., S17 | latent) = Π P(Si | latent)

    Symptoms are conditionally independent given the latent factor.
    """
    if loadings is None:
        loadings = np.random.uniform(0.5, 0.9, len(PTSD_SYMPTOMS))

    # Generate symptoms from latent factor with some noise
    symptoms = np.zeros((n_samples, len(PTSD_SYMPTOMS)))

    for i, (loading, symptom) in enumerate(zip(loadings, PTSD_SYMPTOMS)):
        # Linear model: symptom = loading * latent + noise
        noise = np.random.normal(0, 0.3, n_samples)
        symptoms[:, i] = loading * latent_factor + noise

        # Convert to binary (0/1) similar to PTSD diagnosis criteria
        symptoms[:, i] = (symptoms[:, i] > 0).astype(int)

    return symptoms


def generate_symptom_data_network(
    n_samples: int,
    edge_probability: float = 0.1
) -> np.ndarray:
    """
    Generate symptom data from a network model (for comparison).
    In this model, symptoms directly cause each other.

    This is the alternative hypothesis that the network approach assumes.
    """
    symptoms = np.zeros((n_samples, len(PTSD_SYMPTOMS)))

    # Initialize with some random variation
    for i in range(n_samples):
        symptoms[i] = np.random.binomial(1, 0.3, len(PTSD_SYMPTOMS))

    # Simulate network effects (symptoms influencing each other)
    for _ in range(10):  # Multiple passes
        for i in range(len(PTSD_SYMPTOMS)):
            for j in range(len(PTSD_SYMPTOMS)):
                if i != j and np.random.random() < edge_probability:
                    # Symptom j influences symptom i
                    symptoms[:, i] = np.maximum(
                        symptoms[:, i],
                        symptoms[:, j] * np.random.uniform(0.3, 0.6)
                    )

    return (symptoms > 0).astype(int)


def generate_clinical_sample(n: int = 335) -> pd.DataFrame:
    """
    Generate clinical sample data (n=335).
    Higher base rates of symptoms.
    """
    latent = np.random.normal(0.5, 1, n)  # Elevated latent factor
    symptoms = generate_symptom_data_latent(n, latent)

    df = pd.DataFrame(symptoms, columns=PTSD_SYMPTOMS)
    df['sample'] = 'clinical'
    df['ID'] = range(n)

    return df


def generate_mturk_sample(n: int = 511) -> pd.DataFrame:
    """
    Generate MTurk sample data (n=511).
    Lower base rates of symptoms.
    """
    latent = np.random.normal(-0.2, 1, n)  # Lower latent factor
    symptoms = generate_symptom_data_latent(n, latent)

    df = pd.DataFrame(symptoms, columns=PTSD_SYMPTOMS)
    df['sample'] = 'mturk'
    df['ID'] = range(n)

    return df


def generate_network_sample(n: int) -> pd.DataFrame:
    """
    Generate sample from network model (for comparison).
    """
    symptoms = generate_symptom_data_network(n)

    df = pd.DataFrame(symptoms, columns=PTSD_SYMPTOMS)
    df['sample'] = 'network'
    df['ID'] = range(n)

    return df


def main():
    """Generate and save all simulated datasets."""
    output_dir = Path(__file__).parent / 'data'
    output_dir.mkdir(exist_ok=True)

    # Generate samples
    clinical_df = generate_clinical_sample(335)
    mturk_df = generate_mturk_sample(511)
    network_df = generate_network_sample(500)

    # Combine clinical and MTurk (like original paper)
    combined_df = pd.concat([clinical_df, mturk_df], ignore_index=True)

    # Save datasets
    clinical_df.to_csv(output_dir / 'clinical_sample.csv', index=False)
    mturk_df.to_csv(output_dir / 'mturk_sample.csv', index=False)
    combined_df.to_csv(output_dir / 'combined_sample.csv', index=False)
    network_df.to_csv(output_dir / 'network_sample.csv', index=False)

    # Print summary statistics
    print("=== Simulated Data Summary ===\n")
    print(f"Clinical sample: {len(clinical_df)} observations")
    print(f"  Symptom prevalence: {clinical_df[PTSD_SYMPTOMS].mean().mean():.3f}")

    print(f"\nMTurk sample: {len(mturk_df)} observations")
    print(f"  Symptom prevalence: {mturk_df[PTSD_SYMPTOMS].mean().mean():.3f}")

    print(f"\nNetwork sample: {len(network_df)} observations")
    print(f"  Symptom prevalence: {network_df[PTSD_SYMPTOMS].mean().mean():.3f}")

    print(f"\n=== Combined Dataset ===")
    print(f"Total observations: {len(combined_df)}")
    print(f"Symptom prevalence by sample:")
    print(combined_df.groupby('sample')[PTSD_SYMPTOMS].mean().mean(axis=1))

    print(f"\nDatasets saved to: {output_dir}")


if __name__ == '__main__':
    main()
