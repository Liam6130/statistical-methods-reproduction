"""
Visualization module for Bayesian network analysis.

Creates centrality plots and network visualizations similar to research papers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from typing import Any

from network_analysis import (
    build_glasso_network,
    build_networkx_graph,
    calculate_centrality,
)


def standardize_scores(centrality_dict: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """
    Standardize centrality scores using z-score normalization.

    Args:
        centrality_dict: Dictionary containing centrality measures for each node

    Returns:
        Dictionary with standardized (z-score) centrality values
    """
    standardized: dict[str, dict[str, float]] = {}

    # Extract all values for each centrality type
    centrality_types = ["strength", "betweenness", "closeness"]

    for centrality_type in centrality_types:
        values = [node_data[centrality_type] for node_data in centrality_dict.values()]
        mean = np.mean(values)
        std = np.std(values)

        if std > 0:
            for node, node_data in centrality_dict.items():
                if node not in standardized:
                    standardized[node] = {}
                standardized[node][centrality_type] = (
                    node_data[centrality_type] - mean
                ) / std
        else:
            for node in centrality_dict:
                if node not in standardized:
                    standardized[node] = {}
                standardized[node][centrality_type] = 0.0

    return standardized


def plot_centrality(
    data: pd.DataFrame, output_path: Path | str
) -> None:
    """
    Create centrality plot with three horizontal bar charts (strength, betweenness, closeness).

    Similar to Figure 2 in network analysis papers, showing standardized centrality scores.

    Args:
        data: DataFrame containing symptom data
        output_path: Path to save the figure
    """
    # Extract feature columns
    feature_columns = [col for col in data.columns if col not in ["sample", "ID"]]
    data_values = data[feature_columns].values

    # Build network and calculate centrality
    precision_matrix, adjacency_matrix = build_glasso_network(data_values, alpha=0.1)
    G = build_networkx_graph(adjacency_matrix, feature_columns)
    centrality_results = calculate_centrality(G)

    # Standardize scores
    standardized = standardize_scores(centrality_results)

    # Prepare data for plotting
    nodes = list(standardized.keys())
    strength_scores = [standardized[n]["strength"] for n in nodes]
    betweenness_scores = [standardized[n]["betweenness"] for n in nodes]
    closeness_scores = [standardized[n]["closeness"] for n in nodes]

    # Sort by strength for consistent ordering
    sorted_indices = np.argsort(strength_scores)[::-1]
    nodes = [nodes[i] for i in sorted_indices]
    strength_scores = [strength_scores[i] for i in sorted_indices]
    betweenness_scores = [betweenness_scores[i] for i in sorted_indices]
    closeness_scores = [closeness_scores[i] for i in sorted_indices]

    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 8))
    fig.suptitle("Node Centrality Measures (Standardized Scores)", fontsize=14, fontweight="bold")

    # Color scheme
    colors = {
        "strength": "#2E86AB",
        "betweenness": "#A23B72",
        "closeness": "#F18F01"
    }

    y_positions = np.arange(len(nodes))

    # Plot 1: Strength
    ax1 = axes[0]
    ax1.barh(y_positions, strength_scores, color=colors["strength"], alpha=0.8, edgecolor="black", linewidth=0.5)
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(nodes, fontsize=8)
    ax1.set_xlabel("Standardized Score", fontsize=10)
    ax1.set_title("Strength", fontsize=12, fontweight="bold")
    ax1.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    ax1.grid(axis="x", alpha=0.3)
    ax1.invert_yaxis()

    # Plot 2: Betweenness
    ax2 = axes[1]
    ax2.barh(y_positions, betweenness_scores, color=colors["betweenness"], alpha=0.8, edgecolor="black", linewidth=0.5)
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(nodes, fontsize=8)
    ax2.set_xlabel("Standardized Score", fontsize=10)
    ax2.set_title("Betweenness", fontsize=12, fontweight="bold")
    ax2.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    ax2.grid(axis="x", alpha=0.3)
    ax2.invert_yaxis()

    # Plot 3: Closeness
    ax3 = axes[2]
    ax3.barh(y_positions, closeness_scores, color=colors["closeness"], alpha=0.8, edgecolor="black", linewidth=0.5)
    ax3.set_yticks(y_positions)
    ax3.set_yticklabels(nodes, fontsize=8)
    ax3.set_xlabel("Standardized Score", fontsize=10)
    ax3.set_title("Closeness", fontsize=12, fontweight="bold")
    ax3.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    ax3.grid(axis="x", alpha=0.3)
    ax3.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Centrality plot saved to: {output_path}")


def plot_network(G: nx.Graph, output_path: Path | str) -> None:
    """
    Create network visualization using NetworkX.

    Args:
        G: NetworkX Graph object
        output_path: Path to save the figure
    """
    # Calculate node sizes based on strength (weighted degree)
    node_strengths = dict(G.degree(weight="weight"))
    max_strength = max(node_strengths.values()) if node_strengths else 1

    # Scale node sizes between 300 and 1500
    node_sizes = [
        300 + (node_strengths.get(node, 0) / max_strength) * 1200
        for node in G.nodes()
    ]

    # Calculate centrality for edge weight scaling
    betweenness = nx.betweenness_centrality(G)
    max_betweenness = max(betweenness.values()) if betweenness else 1

    # Edge widths based on weight
    edge_weights = [G[u][v].get("weight", 1) * 3 for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [1 + (w / max_weight) * 4 for w in edge_weights]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use spring layout for positioning
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=0.6,
        edge_color="#666666",
        ax=ax
    )

    # Draw nodes
    node_colors = [betweenness.get(node, 0) for node in G.nodes()]
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.YlOrRd,
        alpha=0.9,
        edgecolors="black",
        linewidths=1,
        ax=ax
    )

    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=8,
        font_weight="bold",
        ax=ax
    )

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label("Betweenness Centrality", fontsize=10)

    ax.set_title("Symptom Network\n(Node size = Strength, Color = Betweenness)", fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Network plot saved to: {output_path}")


def main() -> None:
    """Main function to read data and generate visualizations."""
    # Define paths
    base_dir = Path("/Users/liam/Desktop/My_project/main/bayesian_network_analysis")
    data_path = base_dir / "simulation/data/combined_sample.csv"
    figures_dir = base_dir / "figures"

    # Ensure figures directory exists
    figures_dir.mkdir(exist_ok=True)

    # Load data
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path)

    print(f"Data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")

    # Generate centrality plot
    centrality_output = figures_dir / "centrality_plot.png"
    plot_centrality(data, centrality_output)

    # Build network for network plot
    feature_columns = [col for col in data.columns if col not in ["sample", "ID"]]
    data_values = data[feature_columns].values

    precision_matrix, adjacency_matrix = build_glasso_network(data_values, alpha=0.1)
    G = build_networkx_graph(adjacency_matrix, feature_columns)

    print(f"\nNetwork statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    # Generate network plot
    network_output = figures_dir / "network_plot.png"
    plot_network(G, network_output)

    print("\nAll visualizations generated successfully!")


if __name__ == "__main__":
    main()
