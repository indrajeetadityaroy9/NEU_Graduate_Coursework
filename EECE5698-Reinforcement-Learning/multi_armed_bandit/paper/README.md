# Paper: Adaptive Exploration Under Distribution Shift

arXiv-level LaTeX paper documenting the comprehensive empirical study of non-stationary multi-armed bandit algorithms.

## Paper Structure

```
paper/
├── main.tex          # Complete paper (887 lines)
├── Makefile          # Compilation commands
└── README.md         # This file
```

## Compilation

```bash
# Full compilation (3 passes for cross-references)
make

# Quick single pass
make quick

# Clean auxiliary files
make clean

# View PDF
make view
```

### Requirements

- pdflatex
- Standard LaTeX packages: amsmath, booktabs, algorithm, algorithmic, natbib, hyperref, tikz

### Online Compilation

Upload `main.tex` to [Overleaf](https://overleaf.com) for online compilation without local LaTeX installation.

## Paper Contents

### Sections

1. **Introduction** - Research questions and contributions
2. **Related Work** - Stationary and non-stationary bandit literature
3. **Problem Formulation** - Mathematical framework and metrics
4. **Algorithms** - 13 algorithms with equations and descriptions
5. **Experimental Setup** - Environments, configurations, protocol
6. **Results** - Main findings with statistical analysis
7. **Ablation Studies** - Hyperparameter sensitivity analysis
8. **Theoretical Analysis** - Stability-plasticity dilemma explanation
9. **Benchmark Validation** - OBP integration results
10. **Conclusion** - Summary and future work
11. **Appendix** - Complete results tables and pseudocode

### Key Results Documented

| Finding | Evidence |
|---------|----------|
| D-UCB(0.99) dominates non-stationary | Rank 1 in 3/4 environments |
| Thompson Sampling fails on drift | 88.5% → 23% optimal rate |
| Gradual drift harder than sudden | UCB1: 70.8% sudden vs 40.6% gradual |
| Effective memory ≈ 20% of interval | γ=0.99 optimal for Δ=500 |
| Price of adaptability: ~11% | 0.788 vs 0.885 stationary reward |

### Tables Included

- Table 1: Environment configurations
- Table 2: Algorithm configurations
- Table 3: Main results (regret ± std)
- Table 4: Algorithm rankings
- Table 5: Stability-plasticity tradeoff
- Table 6: Drift comparison (sudden vs gradual)
- Table 7: Window size ablation
- Table 8: Discount factor ablation
- Table 9: Validated theoretical principles
- Table A1: Complete statistics
- Algorithm 1: D-UCB pseudocode
- Algorithm 2: SW-UCB pseudocode

### References

15 citations including:
- Auer et al. (2002) - UCB1, EXP3
- Garivier & Moulines (2011) - D-UCB, SW-UCB
- Thompson (1933), Agrawal & Goyal (2012) - Thompson Sampling
- Besbes et al. (2014) - Non-stationary bandits
- Saito et al. (2020) - Open Bandit Pipeline

## Word Count

Approximately 5,500 words (typical arXiv paper length: 4,000-8,000 words).
