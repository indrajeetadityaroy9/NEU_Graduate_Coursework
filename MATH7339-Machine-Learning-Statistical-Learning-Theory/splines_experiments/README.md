# Splines Implementations
Spline-based regression methods

## Overview
- Truncated power basis for splines
- Regression splines (manual knot placement)
- Natural cubic splines (reduced boundary variance)
- Smoothing splines (automatic knot selection via regularization)

### 1. Truncated Power Basis

For a $k$-th order spline with knots at $t_1 < \dots < t_m$, the basis consists of:
- Polynomial terms: $g_1(x)=1, g_2(x)=x, \dots, g_{k+1}(x)=x^k$
- Truncated terms: $g_{k+1+j}(x) = (x - t_j)_+^k$ where $(x)_+ = \max(x, 0)$

### 2. Regression Splines

Minimize least squares:
$$\min_{\beta} \|y - G\beta\|_2^2$$

where $G$ is the basis matrix. Solution:
$$\hat{\beta} = (G^T G)^{-1} G^T y$$

### 3. Natural Splines

- Cubic between interior knots
- **Linear beyond boundary knots** (reduces variance)
- Uses only $m$ basis functions (vs $m+k+1$ for regular splines)

### 4. Smoothing Splines

Solve regularized problem:
$$\min_{\beta} \|y - G\beta\|_2^2 + \lambda \beta^T \Omega \beta$$

where $\Omega_{ij} = \int g''_i(t) g''_j(t) dt$ penalizes curvature.
