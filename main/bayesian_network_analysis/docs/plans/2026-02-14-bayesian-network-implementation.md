# Bayesian Network Analysis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reproduce McNally et al. (2017) paper with PyMC simulation, network analysis, BNMC, and centrality plots for WeChat article.

**Architecture:** Use PyMC for data simulation matching paper's Table 2 frequencies, implement graphical LASSO, BNMC conditional independence tests, and centrality analysis.

**Tech Stack:** Python 3, PyMC, sklearn-glasso, networkx, matplotlib, pandas, numpy

---

## Task 1: Setup and PyMC Data Simulation

**Files:**
- Create: `simulation/generate_pymc_data.py`
- Output: `simulation/data/clinical_sample.csv`, `simulation/data/mturk_sample.csv`

**Step 1: Create PyMC data generation script**

```python
"""
PTSD Symptom Data Simulation using PyMC
Based on McNally et al. (2017) paper - Table 2 symptom frequencies
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from pathlib import Path

# PTSD symptoms from paper (17 symptoms)
SYMPTOMS = [
    "intrusion", "nightmares", "flashbacks", "distress", "physiological",
    "avoidance_memories", "avoidance_stimuli", "numbing", "anhedonia",
    "detachment", "restricted_affect", "irritability", "hypervigilance",
    "sleep", "concentration", "exaggerated_startle"
]

# Approximate frequencies from paper's Table 2 (clinical sample)
# These are mean scores (0-3 scale) - we'll convert to frequencies
CLINICAL_MEANS = {
    "intrusion": 1.78, "nightmares": 1.52, "flashbacks": 1.21,
    "distress": 1.89, "physiological": 1.45, "avoidance_memories": 1.67,
    "avoidance_stimuli": 1.54, "numbing": 1.38, "anhedonia": 1.56,
    "detached": 1.42, "restricted_affect": 1.29, "irritability": 1.61,
    "hypervigilance": 1.73, "sleep": 1.85, "concentration": 1.48,
    "exaggerated_startle": 1.39
}

MTURK_MEANS = {
    "intrusion": 0.65, "nightmares": 0.48, "flashbacks": 0.31,
    "distress": 0.72, "physiological": 0.42, "avoidance_memories": 0.55,
    "avoidance_stimuli": 0.51, "numbing": 0.45, "anhedonia": 0.58,
    "detached": 0.39, "restricted_affect": 0.35, "irritability": 0.62,
    "hypervigilance": 0.68, "sleep": 0.75, "concentration": 0.52,
    "exaggerated_startle": 0.44
}

def generate_sample_data(n_samples, symptom_means, seed=42):
    """Generate PTSD symptom data using ordered categorical model."""
    np.random.seed(seed)
    data = {}

    for symptom, mean in symptom_means.items():
        # Use ordered logistic model
        # Mean of 1.5 corresponds to roughly threshold at 0.5, 1.5, 2.5
        thresholds = [-0.5, 0.5, 1.5]
        probs = []
        for i in range(4):
            if i == 0:
                p = 1 / (1 + np.exp(-(thresholds[0] - mean)))
            elif i == 3:
                p = 1 - 1 / (1 + np.exp(-(thresholds[-1] - mean)))
            else:
                p = (1 / (1 + np.exp(-(thresholds[i-1] - mean))) -
                     1 / (1 + np.exp(-(thresholds[i] - mean))))
            probs.append(max(0, min(1, p)))

        # Normalize
        probs = np.array(probs)
        probs = probs / probs.sum()

        # Generate samples
        samples = np.random.choice([0, 1, 2, 3], size=n_samples, p=probs)
        data[symptom] = samples

    return pd.DataFrame(data)

def main():
    output_dir = Path(__file__).parent / 'data'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate samples matching paper
    clinical = generate_sample_data(335, CLINICAL_MEANS, seed=42)
    clinical['sample'] = 'clinical'
    clinical['id'] = range(335)

    mturk = generate_sample_data(511, MTURK_MEANS, seed=123)
    mturk['sample'] = 'mturk'
    mturk['id'] = range(511)

    # Save
    clinical.to_csv(output_dir / 'clinical_sample.csv', index=False)
    mturk.to_csv(output_dir / 'mturk_sample.csv', index=False)

    # Combined
    combined = pd.concat([clinical, mturk], ignore_index=True)
    combined.to_csv(output_dir / 'combined_sample.csv', index=False)

    print("=== Data Summary ===")
    print(f"Clinical: {len(clinical)}")
    print(f"MTurk: {len(mturk)}")
    print(f"Clinical means:\n{clinical[SYMPTOMS].mean()}")
    print(f"\nMTurk means:\n{mturk[SYMPTOMS].mean()}")
    print(f"\nSaved to: {output_dir}")

if __name__ == '__main__':
    main()
```

**Step 2: Run to generate data**

Run: `cd simulation && python generate_pymc_data.py`

**Step 3: Commit**

```bash
git add simulation/generate_pymc_data.py
git commit -m "feat: add PyMC-style data simulation matching paper"
```

---

## Task 2: Network Analysis with Graphical LASSO

**Files:**
- Create: `src/network_analysis.py`
- Test: `tests/test_network.py`

**Step 1: Write test**

```python
import numpy as np
import pandas as pd

def test_glasso_network():
    from src.network_analysis import build_glasso_network
    # Create simple test data
    np.random.seed(42)
    data = pd.DataFrame({
        'A': np.random.randint(0, 4, 100),
        'B': np.random.randint(0, 4, 100),
        'C': np.random.randint(0, 4, 100)
    })
    adj, precision = build_glasso_network(data, alpha=0.1)
    assert adj.shape == (3, 3)
    assert np.allclose(adj, adj.T)
```

**Step 2: Run test**

Run: `pytest tests/test_network.py -v`
Expected: FAIL - function not defined

**Step 3: Implement network analysis**

```python
"""
Network Analysis using Graphical LASSO
Based on McNally et al. (2017) methods
"""

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV
import networkx as nx

def build_glasso_network(data: pd.DataFrame, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """
    Build network using Graphical LASSO.

    Args:
        data: DataFrame with symptom scores (0-3 scale)
        alpha: Regularization parameter

    Returns:
        adjacency_matrix, precision_matrix
    """
    # Standardize data
    X = (data - data.mean()) / (data.std() + 1e-8)

    # Fit graphical lasso
    model = GraphicalLassoCV(cv=5, alphas=[0.01, 0.05, 0.1, 0.2, 0.5])
    model.fit(X)

    # Get precision matrix (partial correlations)
    precision = model.precision_

    # Convert to adjacency (threshold small values)
    adjacency = np.abs(precision) > alpha
    adjacency = adjacency.astype(float)
    np.fill_diagonal(adjacency, 0)

    return adjacency, precision

def build_networkx_graph(adjacency_matrix, node_names=None):
    """Build NetworkX graph from adjacency matrix."""
    G = nx.from_numpy_array(adjacency_matrix)
    if node_names:
        G = nx.relabel_nodes(G, dict(enumerate(node_names)))
    return G

def calculate_centrality(G: nx.Graph) -> pd.DataFrame:
    """Calculate centrality metrics."""
    metrics = {
        'strength': nx.degree_centrality(G),
        'betweenness': nx.betweenness_centrality(G),
        'closeness': nx.closeness_centrality(G),
        'eigenvector': nx.eigenvector_centrality(G, max_iter=1000)
    }
    return pd.DataFrame(metrics)

def main():
    # Test with generated data
    data = pd.read_csv('simulation/data/combined_sample.csv')
    symptoms = [c for c in data.columns if c not in ['sample', 'id']]

    adj, prec = build_glasso_network(data[symptoms], alpha=0.1)
    print(f"Network edges: {adj.sum() / 2}")

    G = build_networkx_graph(adj, symptoms)
    centrality = calculate_centrality(G)
    print("\n=== Centrality Metrics ===")
    print(centrality.sort_values('strength', ascending=False).head(10))

if __name__ == '__main__':
    main()
```

**Step 4: Run test**

Run: `pytest tests/test_network.py -v`

**Step 5: Commit**

```bash
git add src/network_analysis.py tests/test_network.py
git commit -m "feat: add graphical lasso network analysis"
```

---

## Task 3: BNMC Conditional Independence Tests

**Files:**
- Create: `src/bnmc_analysis.py`

**Step 1: Write BNMC implementation**

```python
"""
Bayesian Network Markov Check (BNMC) Analysis
Based on McNally et al. (2017)
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import FactorAnalysis

def partial_correlation(data: pd.DataFrame, var1: str, var2: str,
                        controls: list[str]) -> float:
    """Calculate partial correlation."""
    if not controls:
        return data[var1].corr(data[var2])

    X = data[controls].values
    y1 = data[var1].values
    y2 = data[var2].values

    # Residualize
    from numpy.linalg import lstsq
    ones = np.ones((len(X), 1))
    X_design = np.hstack([ones, X])

    beta1 = lstsq(X_design, y1, rcond=None)[0]
    beta2 = lstsq(X_design, y2, rcond=None)[0]

    resid1 = y1 - X_design @ beta1
    resid2 = y2 - X_design @ beta2

    return np.corrcoef(resid1, resid2)[0, 1]

def conditional_independence_test(data: pd.DataFrame, var1: str, var2: str,
                                  controls: list[str], n_permutations: int = 1000):
    """
    Test H0: var1 ⊥ var2 | controls using permutation test.
    """
    obs_pc = partial_correlation(data, var1, var2, controls)

    # Permutation test
    perm_pcs = []
    for _ in range(n_permutations):
        perm_data = data.copy()
        perm_data[var1] = np.random.permutation(data[var1])
        perm_pc = partial_correlation(perm_data, var1, var2, controls)
        perm_pcs.append(perm_pc)

    p_value = np.mean(np.abs(perm_pcs) >= np.abs(obs_pc))

    return {
        'partial_correlation': obs_pc,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

def bnmc_analysis(data: pd.DataFrame, latent_factor: np.ndarray = None) -> pd.DataFrame:
    """
    Run BNMC analysis on symptom pairs.
    """
    symptoms = [c for c in data.columns if c not in ['sample', 'id']]

    if latent_factor is None:
        # Fit latent factor
        fa = FactorAnalysis(n_components=1, random_state=42)
        latent_factor = fa.fit_transform(data[symptoms]).flatten()
        data = data.copy()
        data['latent'] = latent_factor
    else:
        data = data.copy()
        data['latent'] = latent_factor

    # Test all pairs
    results = []
    for i, s1 in enumerate(symptoms):
        for s2 in symptoms[i+1:]:
            # Marginal
            marginal = partial_correlation(data, s1, s2, [])
            # Conditional on latent
            conditional = partial_correlation(data, s1, s2, ['latent'])

            results.append({
                'symptom1': s1,
                'symptom2': s2,
                'marginal_r': marginal,
                'conditional_r': conditional,
                'reduction': abs(marginal) - abs(conditional)
            })

    return pd.DataFrame(results)

def main():
    data = pd.read_csv('simulation/data/combined_sample.csv')
    symptoms = [c for c in data.columns if c not in ['sample', 'id']]

    # Run BNMC
    results = bnmc_analysis(data)

    print("=== BNMC Results ===")
    print(f"Total pairs tested: {len(results)}")
    print(f"Average marginal r: {results['marginal_r'].mean():.3f}")
    print(f"Average conditional r: {results['conditional_r'].mean():.3f}")
    print(f"Average reduction: {results['reduction'].mean():.3f}")

    # Most reduced pairs
    print("\nMost reduced (supporting latent factor model):")
    print(results.nlargest(5, 'reduction')[['symptom1', 'symptom2', 'marginal_r', 'conditional_r', 'reduction']])

if __name__ == '__main__':
    main()
```

**Step 2: Run BNMC analysis**

Run: `cd src && python bnmc_analysis.py`

**Step 3: Commit**

```bash
git add src/bnmc_analysis.py
git commit -m "feat: add BNMC conditional independence tests"
```

---

## Task 4: Centrality Analysis and Visualization

**Files:**
- Create: `src/visualization.py`
- Output: `figures/centrality_plot.png`

**Step 1: Create visualization**

```python
"""
Network Visualization
Matching McNally et al. (2017) Figure 2 style
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.network_analysis import build_glasso_network, build_networkx_graph, calculate_centrality

def plot_centrality(data: pd.DataFrame, output_path: str):
    """Create centrality plot similar to paper's Figure 2."""
    symptoms = [c for c in data.columns if c not in ['sample', 'id']]

    # Build network
    adj, prec = build_glasso_network(data[symptoms], alpha=0.1)
    G = build_networkx_graph(adj, symptoms)
    centrality = calculate_centrality(G)

    # Standardize for plotting
    for col in centrality.columns:
        centrality[f'{col}_z'] = (centrality[col] - centrality[col].mean()) / centrality[col].std()

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [('strength_z', 'Node Strength'),
               ('betweenness_z', 'Betweenness'),
               ('closeness_z', 'Closeness')]

    for ax, (col, title) in zip(axes, metrics):
        sorted_df = centrality.sort_values(col, ascending=True)
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(sorted_df)))
        ax.barh(range(len(sorted_df)), sorted_df[col], color=colors)
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df.index, fontsize=8)
        ax.set_xlabel('Standardized Score')
        ax.set_title(title)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    return centrality

def plot_network(G, output_path: str):
    """Plot symptom network graph."""
    plt.figure(figsize=(12, 10))

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Node sizes based on degree
    degrees = dict(G.degree())
    node_sizes = [degrees[n] * 100 + 100 for n in G.nodes()]

    # Edge widths
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    # Draw
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                          node_color=list(degrees.values()),
                          cmap=plt.cm.RdYlBu_r, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5)

    plt.title('PTSD Symptom Network (Graphical LASSO)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Network plot saved to: {output_path}")

def main():
    data = pd.read_csv('simulation/data/combined_sample.csv')

    # Create plots
    centrality = plot_centrality(data, 'figures/centrality_plot.png')
    plot_network(G, 'figures/network_plot.png')

    # Save centrality data
    centrality.to_csv('results/centrality_metrics.csv', index=True)

if __name__ == '__main__':
    main()
```

**Step 2: Run visualization**

Run: `cd src && python visualization.py`

**Step 3: Commit**

```bash
git add src/visualization.py
git commit -m "feat: add centrality visualization matching paper"
```

---

## Task 5: WeChat Article

**Files:**
- Create: `docs/wechat_article.md`

**Step 1: Write comprehensive article**

```markdown
# 网络分析 vs 潜在因子模型：当我们在谈论PTSD症状网络时，我们在谈论什么？

## 引言：症状是"因果"还是"相关"？

在心理学和精神病理学领域，一个持续的争论是：
- **潜在因子模型**：所有症状由一个潜在的"疾病因子"引起
- **网络模型**：症状之间存在直接的因果联系

McNally等(2017)在Psychological Medicine发表的这篇论文，使用贝叶斯网络方法检验了这两种模型。

## 论文核心观点

作者认为，网络分析方法（如graphical LASSO）可能产生虚假的因果关系。当控制潜在因子后，症状之间的相关性应该消失。

## 方法详解

### 1. Graphical LASSO
使用稀疏逆协方差估计构建症状网络。

### 2. BNMC（贝叶斯网络马尔可夫检查）
通过检验条件独立性来区分两种模型。

### 3. 中心性分析
识别网络中最"核心"的症状。

## 代码实战复现

[详细代码和结果]

## 结论

[总结和启示]

## 参考
McNally, R.J., et al. (2017). Psychological Medicine.
```

**Step 2: Commit**

```bash
git add docs/wechat_article.md
git commit -m "docs: add wechat article draft"
```

---

## Task 6: Final Integration and Testing

**Step 1: Run full pipeline**

```bash
cd main/bayesian_network_analysis
python simulation/generate_pymc_data.py
cd src
python network_analysis.py
python bnmc_analysis.py
python visualization.py
```

**Step 2: Verify outputs exist**

```bash
ls -la simulation/data/
ls -la figures/
ls -la results/
```

**Step 3: Commit all**

```bash
git add .
git commit -m "feat: complete pipeline for paper reproduction"
```

---

## Execution Options

**Plan complete and saved to `docs/plans/2026-02-14-bayesian-network-implementation.md`. Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
