"""
Network analysis module using Graphical LASSO for sparse covariance estimation.
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV


def build_glasso_network(data: np.ndarray, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a network using Graphical LASSO for sparse inverse covariance estimation.

    Args:
        data: 2D array of shape (n_samples, n_variables)
        alpha: Regularization parameter (default: 0.1)

    Returns:
        Tuple of (precision_matrix, adjacency_matrix)
    """
    # Use a range of alpha values for cross-validation
    alphas = np.logspace(-2, 1, 20)

    model = GraphicalLassoCV(cv=5, alphas=alphas, max_iter=1000)
    try:
        model.fit(data)
    except Exception:
        # Fallback to single alpha if CV fails
        model = GraphicalLasso(alpha=alpha, max_iter=1000)
        model.fit(data)

    precision_matrix = model.precision_

    # Create adjacency matrix from precision matrix (non-zero entries indicate edges)
    adjacency_matrix = np.abs(precision_matrix) > 1e-6
    adjacency_matrix = adjacency_matrix.astype(float)
    np.fill_diagonal(adjacency_matrix, 0)

    return precision_matrix, adjacency_matrix


def build_networkx_graph(
    adjacency_matrix: np.ndarray, node_names: list[str]
) -> nx.Graph:
    """
    Build a NetworkX graph from an adjacency matrix.

    Args:
        adjacency_matrix: 2D array representing the adjacency matrix
        node_names: List of node names

    Returns:
        NetworkX Graph object
    """
    G = nx.Graph()

    G.add_nodes_from(node_names)

    n = adjacency_matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency_matrix[i, j] > 0:
                weight = adjacency_matrix[i, j]
                G.add_edge(node_names[i], node_names[j], weight=weight)

    return G


def calculate_centrality(G: nx.Graph) -> dict[str, dict[str, float]]:
    """
    Calculate various centrality measures for the network.

    Args:
        G: NetworkX Graph object

    Returns:
        Dictionary containing centrality measures for each node
    """
    centrality_results: dict[str, dict[str, float]] = {}

    # Calculate all centrality measures
    strength = dict(G.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)

    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigenvector = nx.eigenvector_centrality_numpy(G)

    # Compile results
    for node in G.nodes():
        centrality_results[node] = {
            "strength": float(strength.get(node, 0)),
            "betweenness": float(betweenness.get(node, 0)),
            "closeness": float(closeness.get(node, 0)),
            "eigenvector": float(eigenvector.get(node, 0)),
        }

    return centrality_results


def main() -> None:
    """Main function to run the network analysis pipeline."""
    # Load data
    data_path = "/Users/liam/Desktop/My_project/main/bayesian_network_analysis/simulation/data/combined_sample.csv"
    df = pd.read_csv(data_path)

    # Extract feature columns (exclude 'sample' and 'ID' columns)
    feature_columns = [col for col in df.columns if col not in ["sample", "ID"]]
    data = df[feature_columns].values

    print(f"Data shape: {data.shape}")
    print(f"Features: {feature_columns}")

    # Build GLasso network
    alpha = 0.1
    precision_matrix, adjacency_matrix = build_glasso_network(data, alpha=alpha)

    print(f"\nPrecision matrix shape: {precision_matrix.shape}")
    print(f"Adjacency matrix shape: {adjacency_matrix.shape}")

    # Build NetworkX graph
    G = build_networkx_graph(adjacency_matrix, feature_columns)

    print(f"\nGraph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Calculate centrality measures
    centrality_results = calculate_centrality(G)

    # Print results
    print("\nCentrality Measures:")
    print("-" * 80)
    print(
        f"{'Node':<15} {'Strength':>10} {'Betweenness':>12} {'Closeness':>10} {'Eigenvector':>12}"
    )
    print("-" * 80)

    for node, measures in sorted(centrality_results.items()):
        print(
            f"{node:<15} {measures['strength']:>10.4f} {measures['betweenness']:>12.4f} "
            f"{measures['closeness']:>10.4f} {measures['eigenvector']:>12.4f}"
        )


if __name__ == "__main__":
    main()
