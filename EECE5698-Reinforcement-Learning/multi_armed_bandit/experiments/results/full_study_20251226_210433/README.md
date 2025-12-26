# Multi-Armed Bandit Comparative Study Results

**Generated**: 2025-12-26 21:05:17

**Total Runtime**: 40.2 seconds

## Configuration

- **Algorithms**: 13
- **Environments**: 4
- **Runs per configuration**: 50
- **Horizon**: 10,000 timesteps
- **Arms**: 5
- **Gap**: 1.0

## Environments

- **Stationary**: StationaryBandit
- **Abrupt (100)**: AbruptChangeBandit
- **Abrupt (500)**: AbruptChangeBandit
- **Gradual Drift**: GradualDriftBandit

## Algorithms

- EpsilonGreedy
- DecayingEpsilon
- UCB1
- ThompsonSampling
- GradientBandit
- EpsGreedy-Const
- D-UCB(0.99)
- D-UCB(0.95)
- SW-UCB(100)
- D-TS(0.99)
- EXP3
- Rexp3(100)
- EntropyGradient

## Key Findings

### Stationary

**Best Algorithm**: ThompsonSampling (Mean Regret: 29.3)

Top 3:
1. ThompsonSampling: 29.3
2. UCB1: 52.3
3. GradientBandit: 93.5

### Abrupt (100)

**Best Algorithm**: D-UCB(0.99) (Mean Regret: 2978.7)

Top 3:
1. D-UCB(0.99): 2978.7
2. D-UCB(0.95): 3639.3
3. Rexp3(100): 3702.4

### Abrupt (500)

**Best Algorithm**: D-UCB(0.99) (Mean Regret: 1498.3)

Top 3:
1. D-UCB(0.99): 1498.3
2. EpsGreedy-Const: 1653.9
3. SW-UCB(100): 1761.0

### Gradual Drift

**Best Algorithm**: D-UCB(0.99) (Mean Regret: 1071.7)

Top 3:
1. D-UCB(0.99): 1071.7
2. SW-UCB(100): 1766.0
3. EpsGreedy-Const: 3065.8

## Output Files

```
raw/              - JSON results per environment
plots/            - Visualization PNG files
summary/          - CSV summaries and statistical tests
```
