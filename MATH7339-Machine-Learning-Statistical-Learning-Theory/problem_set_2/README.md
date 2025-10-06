# Notes

## Probit Regression
- Specifies the Bernoulli likelihood with success probability given by
```math
\mu_i = \Phi(\mathbf{x}_i^{\top} \boldsymbol{\beta})
```
- Derives the sample log-likelihood
```math
\ell(\boldsymbol{\beta}) = \sum_i \Big[ y_i \log \Phi(\mathbf{x}_i^{\top} \boldsymbol{\beta}) + (1 - y_i) \log \big(1 - \Phi(\mathbf{x}_i^{\top} \boldsymbol{\beta})\big) \Big]
```
  using logarithm rules and a \{0,1\} outcome sanity check.
- Reviews the link between the linear predictor and response probabilities, emphasising the Gaussian CDF transformation and its derivative (the standard normal PDF \(\phi\)).

## Score Vector and Hessian
- Builds the gradient contribution observation-by-observation, applying the chain rule to both the positive and negative outcome terms.
- Simplifies the full gradient to
```math
\nabla_{\boldsymbol{\beta}} \ell(\boldsymbol{\beta}) = \sum_i \frac{\phi(\mathbf{x}_i^{\top} \boldsymbol{\beta})\, (y_i - \Phi(\mathbf{x}_i^{\top} \boldsymbol{\beta}))}{\Phi(\mathbf{x}_i^{\top} \boldsymbol{\beta})\, \big(1 - \Phi(\mathbf{x}_i^{\top} \boldsymbol{\beta})\big)} \, \mathbf{x}_i.
```
- Computes the second derivative matrix by differentiating the gradient, yielding expressions that combine PDF terms, the linear predictor, and observation outer products.

## Newton–Raphson Algorithm
- Implements iterations of the form
```math
\boldsymbol{\beta}_{\mathrm{new}} = \boldsymbol{\beta} - \mathbf{H}(\boldsymbol{\beta})^{-1} \mathbf{G}(\boldsymbol{\beta})
```
  where \(\mathbf{G}\) is the gradient and \(\mathbf{H}\) is the Hessian.
- Applies the routine to a five-observation dataset with design matrix
```math
\mathbf{X} = \begin{bmatrix}
1 & 2 & 4 \\
1 & 1 & 1 \\
1 & 4 & 3 \\
1 & 5 & 6 \\
1 & 3 & 5
\end{bmatrix}, \quad
\mathbf{y} = \begin{bmatrix}
1 \\
0 \\
1 \\
1 \\
0
\end{bmatrix}, \quad
\boldsymbol{\beta}^{(0)} = \begin{bmatrix}
0.1 \\
0.1 \\
0.1
\end{bmatrix}
```
  and converges in five iterations to \(\hat{\boldsymbol{\beta}} \approx (-1.5463,\, 0.7778,\, -0.0971)\).

## Poisson Regression Extension
- Outlines an analogous Newton–Raphson setup for Poisson regression with mean function
```math
\mu_i = \exp(\mathbf{x}_i^{\top} \boldsymbol{\beta})
```
  and log-likelihood
```math
\ell(\boldsymbol{\beta}) = \sum_i \Big[ y_i \log \mu_i - \mu_i - \log \Gamma(y_i + 1) \Big].
```
- Identifies the score and Hessian as
```math
\mathbf{G}(\boldsymbol{\beta}) = \mathbf{X}^{\top}(\mathbf{y} - \boldsymbol{\mu}), \qquad
\mathbf{H}(\boldsymbol{\beta}) = -\mathbf{X}^{\top} \mathrm{diag}(\boldsymbol{\mu}) \, \mathbf{X}.
```
