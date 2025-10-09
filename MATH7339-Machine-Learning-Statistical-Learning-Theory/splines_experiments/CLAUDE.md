# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **graduate-level** mathematical/statistical computing project implementing advanced spline-based regression and classification methods. The codebase combines theoretical rigor with real-world applications, featuring implementations from scratch using NumPy, SciPy, and scikit-learn.

**Educational + Applied**: Demonstrates both theoretical foundations and practical applications to medical/health datasets.

## Core Architecture

### Implementation Files

**splines.py** - Core spline implementations using truncated power basis:
- `RegressionSpline`: Manual knot placement with least squares fitting
- `NaturalCubicSpline`: Reduces boundary variance by constraining spline to be linear beyond boundary knots
- `SmoothingSpline`: Automatic knot selection via regularization (penalizes curvature using penalty matrix Ω)

**utils.py** - Data generation, metrics, and visualization utilities:
- Data generators: `generate_sinusoidal_data()`, `generate_polynomial_data()`, `generate_step_data()`, `generate_discontinuous_data()`
- Metrics: `mean_squared_error()`, `r_squared()`, `root_mean_squared_error()`, `effective_degrees_of_freedom()`
- Plotting: `plot_spline_fit()`, `plot_basis_functions()`, `plot_smoothing_comparison()`, `plot_residuals()`, `plot_cv_curve()`

### Mathematical Foundations

All splines use **truncated power basis**:
- For degree k with knots at t₁, ..., tₘ: basis includes polynomial terms (1, x, x², ..., xᵏ) and truncated terms (x - tⱼ)₊ᵏ
- `truncated_power_basis_matrix()` constructs the G matrix with shape (n, m+k+1)

**Regression splines**: Minimize ||y - Gβ||₂² → solution: β̂ = (GᵀG)⁻¹Gᵀy

**Natural cubic splines**: Use special basis constructed via `natural_cubic_spline_basis_matrix()` that enforces linear extrapolation beyond boundaries

**Smoothing splines**: Solve regularized problem: min ||y - Gβ||₂² + λβᵀΩβ where Ω penalizes second derivative curvature

### Jupyter Notebooks

The notebooks demonstrate spline behavior through experiments:
- `regression_splines.ipynb`: Effect of knot count, polynomial degree, boundary behavior
- `natural_splines.ipynb`: Demonstrates reduced boundary variance
- `smoothing_splines.ipynb`: Cross-validation for λ selection, automatic knot placement
- `basis_functions.ipynb`: Visualizes truncated power basis functions
- `comparison.ipynb`: Compares different spline types
- `splines.ipynb`: Main combined notebook

## Development Commands

### Running Tests
```bash
pytest test_splines.py -v
```

Run specific test class:
```bash
pytest test_splines.py::TestRegressionSpline -v
```

Run with coverage:
```bash
pytest test_splines.py --cov=splines --cov-report=term
```

### Code Quality
```bash
# Format code
black splines.py utils.py test_splines.py

# Lint
flake8 splines.py utils.py test_splines.py
```

### Jupyter Notebooks
```bash
# Launch Jupyter
jupyter notebook

# Convert notebook to Python
jupyter nbconvert --to python <notebook_name>.ipynb
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Important Implementation Notes

### Basis Matrix Construction
- All splines rely on basis matrices (G or N) constructed from input data
- `truncated_power_basis_matrix()` is the foundation for RegressionSpline
- `natural_cubic_spline_basis_matrix()` uses a different basis (d_k functions) that enforces natural boundary conditions

### SmoothingSpline Specifics
- Uses data points themselves as knots: `knots = x_sorted`
- Penalty matrix computation in `_compute_penalty_matrix()` uses numerical differentiation and integration (via scipy.integrate.quad)
- `cross_validate()` method performs k-fold CV to select optimal λ
- Sorts input data internally - predictions must use sorted x_train as knots

### Test Structure
The test suite (test_splines.py) is organized into:
- `TestTruncatedPower`: Tests basis function continuity and correctness
- `TestBasisMatrix`: Validates matrix dimensions and polynomial/truncated terms
- `TestRegressionSpline`: Interpolation, prediction shape, linear regression edge case
- `TestNaturalCubicSpline`: Linear boundary behavior, fit quality
- `TestSmoothingSpline`: Lambda effects (interpolation vs smoothing), CV functionality
- `TestEdgeCases`: Single knot, noisy data, constant data
- `TestMathematicalProperties`: Linear smoother property, spline smoothness

### Knot Selection Guidelines
- **Regression splines**: Manually choose knots based on expected function changes (e.g., `np.linspace()` or quantiles)
- **Natural splines**: Include boundary points in knots array for proper constraint enforcement
- **Smoothing splines**: No manual knot selection - uses all data points

## Data Files

- `Heart.csv`, `fev.csv`: Real-world datasets for demonstrations (not currently imported by main code)
- CSV files can be used in notebooks for real data experiments

## Graduate-Level Enhancements

### **NEW: B-Spline Basis** (`splines.py:12-70`)
- **Function**: `bspline_basis(x, knots, degree, derivatives)`
- Numerically stable alternative to truncated power basis
- Uses scipy's BSpline implementation
- Supports derivative evaluation (0th, 1st, 2nd order)
- **Benefits**: Better conditioning, local support, computational efficiency

### **NEW: Penalized Splines (P-Splines)** (`splines.py:308-535`)
- **Class**: `PenalizedSpline`
- Combines B-splines with difference penalties on coefficients
- More flexible and faster than smoothing splines
- **Methods**:
  - `fit(x, y)`: Fits penalized regression via (B'B + λP)β = B'y
  - `effective_df()`: Computes trace(H) for model complexity
  - `gcv_score()`: Generalized Cross-Validation for model selection
  - `cross_validate()`: K-fold CV for optimal λ
  - `_difference_penalty_matrix()`: Constructs D'D penalty (1st, 2nd, 3rd order)
- **Parameters**:
  - `n_knots`: Number of interior knots (default=20)
  - `degree`: B-spline degree (default=3)
  - `lambda_`: Smoothing parameter
  - `diff_order`: Order of difference penalty (default=2 for curvature)
- **Reference**: Eilers & Marx (1996), Statistical Science

### **NEW: Data Utilities** (`data_utils.py`)
- **Heart.csv loading** (`load_heart_data()`):
  - Classification task: Predict CHD (coronary heart disease)
  - Features: sbp, tobacco, ldl, adiposity, famhist, typea, obesity, alcohol, age
  - 461 observations, binary outcome
  - Automatic encoding of categorical variables
  - Stratified train/test split
  - Optional standardization

- **fev.csv loading** (`load_fev_data()`):
  - Regression task: Predict FEV (forced expiratory volume)
  - Features: age, height, sex, smoke status
  - 654 observations, continuous outcome
  - Lung function prediction from demographics

- **Summary functions**:
  - `get_heart_data_summary()`: Dataset statistics, class balance
  - `get_fev_data_summary()`: Feature distributions, categorical counts

### **NEW: Advanced Metrics & Diagnostics** (`utils.py:231-560`)

**Classification Metrics**:
- `classification_metrics()`: Accuracy, precision, recall, F1, AUC-ROC, confusion matrix
- `plot_roc_curve()`: ROC curve with AUC
- `plot_precision_recall_curve()`: PR curve for imbalanced data
- `plot_calibration_curve()`: Probability calibration assessment
- `plot_confusion_matrix()`: Heatmap visualization

**Model Diagnostics**:
- `plot_partial_dependence()`: Marginal effect plots for interpretability
- `plot_qq_plot()`: Normality assessment for residuals

### Dependencies (Updated)
Added for graduate-level features:
- `pandas>=1.3.0`: Data manipulation
- `scikit-learn>=1.0.0`: Metrics, preprocessing, model utilities
- `seaborn>=0.11.0`: Advanced statistical visualizations
- `patsy>=0.5.2`: Formula parsing for GAMs (optional)

## Real-World Applications

### Heart Disease Classification (Heart.csv)
- **Task**: Binary classification of coronary heart disease
- **Methods**: Will use LogisticGAM with spline-based features
- **Features**: Continuous (blood pressure, cholesterol, age) + categorical (family history)
- **Evaluation**: AUC-ROC, calibration, clinical interpretability

### Lung Function Regression (fev.csv)
- **Task**: Predict FEV from physical/demographic characteristics
- **Methods**: GAM with smooth functions for age and height
- **Features**: Age, height (continuous), sex, smoking status (categorical)
- **Evaluation**: RMSE, R², partial dependence plots

### **NEW: GAM Classes** (`gam.py`)

#### GAM - Generalized Additive Model for Regression
- **Class**: `GAM`
- Model form: E(Y) = β₀ + f₁(X₁) + ... + fₚ(Xₚ)
- Backfitting algorithm for estimation
- **Features**:
  - Mixed smooth and linear terms
  - Multiple smoothing parameters (one per term)
  - Automatic centering of smooth functions
  - Effective DF for each component
  - `get_smooth_function()`: Extract individual smooth effects
  - `summary()`: Model diagnostics
- **Parameters**:
  - `smooth_features`: List of feature indices for smoothing
  - `linear_features`: List for linear inclusion
  - `n_knots`, `lambda_`: Per-term configuration
  - `max_iter=100`, `tol=1e-4`: Convergence control
- **Reference**: Hastie & Tibshirani (1990)

#### LogisticGAM - GAM for Binary Classification
- **Class**: `LogisticGAM`
- Model form: log(p/(1-p)) = β₀ + f₁(X₁) + ... + fₚ(Xₚ)
- IRLS (Iteratively Reweighted Least Squares) with backfitting
- **Features**:
  - Logistic link function with numerical stability
  - Deviance computation for monitoring
  - `predict_proba()`: Probability predictions
  - `predict()`: Class labels with threshold
  - `predict_linear()`: Log-odds (linear predictor)
  - Smooth functions on log-odds scale
- **Performance**:
  - Heart dataset: AUC=0.760, Recall=0.812
  - Effective for medical screening tasks
- **Parameters**: Same structure as GAM
- **Reference**: Wood (2017) - GAM textbook

## Implementation Status

✅ **FULLY COMPLETED** (Core Graduate-Level Infrastructure):
- ✓ B-spline basis function (numerically stable)
- ✓ PenalizedSpline class with GCV and effective DF
- ✓ GAM class with backfitting algorithm
- ✓ LogisticGAM for binary classification with IRLS
- ✓ Data loading utilities for both datasets
- ✓ Advanced classification metrics and diagnostic plots
- ✓ Comprehensive test scripts with validation

📊 **Demonstrated Results**:
- P-spline regression: FEV Test R²=0.631
- GAM regression: FEV Test R²=0.234 (4 features)
- LogisticGAM: Heart Test AUC=0.760, Recall=0.812

🚧 **Optional Enhancements** (Nice-to-have):
- gam_regression.ipynb: Interactive FEV prediction notebook
- gam_classification.ipynb: Interactive CHD prediction notebook
- advanced_splines.ipynb: Theoretical deep-dive
- model_selection.ipynb: Comprehensive method comparison
- Extended unit tests for test_splines.py

## Path Import Pattern

Note: test_splines.py uses `sys.path.append('../src')` which appears to be legacy - the actual structure has all Python files in root, so notebooks use local imports directly.
