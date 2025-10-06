# Notes

## Model Setup
- Fixed-design linear model:
```math latex
\mathbf{y} = X \theta + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2 \mathrm{I}_N)
```
- Because the design matrix X is deterministic, all expectations, variances, and covariances condition on X; randomness arises solely from the noise term.

## Ridge Estimator Properties
- Ridge regression estimator:
```math latex
\hat{\theta} = (X^\top X + \lambda \mathrm{I}_d)^{-1} X^\top \mathbf{y}
```
- Bias and mean:
```math latex
\mathrm{Bias}(\hat{\theta}) = -\lambda (X^\top X + \lambda \mathrm{I}_d)^{-1} \theta
```
```math latex
\mathrm{E}[\hat{\theta}] = (\mathrm{I}_d - \lambda (X^\top X + \lambda \mathrm{I}_d)^{-1}) \theta
```
- Covariance:
```math latex
\mathrm{Cov}(\hat{\theta}) = \sigma^2 (X^\top X + \lambda \mathrm{I}_d)^{-1} X^\top X (X^\top X + \lambda \mathrm{I}_d)^{-1}
```

## Prediction Error Decomposition
- Prediction at a fixed test input:
```math latex
\hat{y}^{(0)} = x^{(0)\top} \hat{\theta}
```
- Helper matrix:
```math latex
S = (X^\top X + \lambda \mathrm{I}_d)^{-1}
```
- Bias and variance contributions:
```math latex
\mathrm{Bias}(\hat{y}^{(0)})^2 = \lambda^2 \theta^\top S x^{(0)} x^{(0)\top} S \theta
```
```math latex
\mathrm{Var}(\hat{y}^{(0)}) = \sigma^2 x^{(0)\top} S X^\top X S x^{(0)}
```
- Expected prediction error:
```math latex
\mathrm{EPE}(x^{(0)}) = \sigma^2 + \mathrm{Bias}(\hat{y}^{(0)})^2 + \mathrm{Var}(\hat{y}^{(0)})
```

## Dual / Kernel Ridge Perspective
- Kernel matrix and objective:
```math latex
K = X X^\top
```
```math latex
\mathrm{J}(\beta) = \lVert K \beta - \mathbf{y} \rVert_2^2 + \lambda \beta^\top K \beta
```
- Dual solution and fitted values:
```math latex
\hat{\beta} = (K + \lambda \mathrm{I})^{-1} \mathbf{y}, \qquad \hat{\mathbf{y}} = K (K + \lambda \mathrm{I})^{-1} \mathbf{y}
```

## Degrees of Freedom
- Smoother matrix and degrees of freedom:
```math latex
A = K (K + \lambda \mathrm{I})^{-1}
```
```math latex
\mathrm{df}(\hat{\mathbf{y}}) = \mathrm{tr}\big(A\big) = \mathrm{tr}\big(K (K + \lambda \mathrm{I})^{-1}\big)
```

## Useful Takeaways
- Ridge regression trades bias for variance reduction; understanding the bias, covariance, and prediction error terms helps guide the choice of the regularization parameter lambda.
- Framing ridge in the dual space highlights connections to kernel methods and streamlines the degrees-of-freedom calculation.
