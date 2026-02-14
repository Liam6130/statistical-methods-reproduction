# Bayesian Network Analysis Project Design

## Project Overview
- **Project Name**: Bayesian Network PTSD Analysis Reproduction
- **Purpose**: Reproduce McNally et al. (2017) paper methods for WeChat public account article
- **Target Audience**: Chinese readers interested in psychological statistics methods

## Research Paper Summary
- **Paper**: McNally, Mair, Mugno & Riemann (2017) - Psychological Medicine
- **Core Argument**: Network analysis methods may produce spurious causal relationships; symptoms become conditionally independent when controlling for latent factor
- **Methods**: Graphical LASSO, BNMC (Bayesian Network Markov Check), Centrality Analysis
- **Data**: Two samples - Clinical (n=335), MTurk (n=511), 17 PTSD symptoms (0-3 scale)

## Technical Stack
- **Primary**: Python 3
- **Bayesian Simulation**: PyMC (pymc)
- **Network Analysis**: sklearn-glasso, networkx, graph-tool
- **Visualization**: matplotlib, seaborn, pyvis
- **Statistics**: scipy, numpy, pandas

## Architecture

```
bayesian_network_analysis/
├── src/
│   ├── data_simulation.py     # PyMC-based data generation
│   ├── network_analysis.py   # Graphical LASSO network
│   ├── bnmc_analysis.py      # Conditional independence tests
│   ├── centrality.py         # Centrality metrics
│   └── visualization.py      # Network & centrality plots
├── simulation/
│   └── data/                 # Generated datasets
├── results/                   # Analysis outputs
├── figures/                   # Generated figures
├── docs/
│   ├── bayesian_theory_lecture.md
│   └── wechat_article.md     # WeChat article
└── README.md
```

## Implementation Approach
1. Use PyMC to simulate 0-3 scale symptom data matching paper's Table 2 frequencies
2. Implement Graphical LASSO network construction
3. Implement BNMC conditional independence tests
4. Calculate centrality metrics (strength, betweenness, closeness)
5. Generate figures matching paper's Fig 1 and Fig 2
6. Write comprehensive WeChat article

## Acceptance Criteria
- [ ] Simulated data has similar symptom frequency distributions as Table 2
- [ ] Network structure shows similar patterns to paper
- [ ] BNMC tests produce interpretable results
- [ ] Centrality plot matches paper's Fig 2 style
- [ ] WeChat article is publication-ready

## Timeline
- Phase 1: Data simulation with PyMC
- Phase 2: Network analysis implementation
- Phase 3: BNMC and centrality analysis
- Phase 4: Visualization
- Phase 5: WeChat article writing
