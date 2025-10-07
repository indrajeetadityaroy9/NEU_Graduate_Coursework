import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import CubicSpline, UnivariateSpline, splrep, splev
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt, pow
from fev_utils import load_data, plot_model


def plot_association(df):
    """Plot the association between fev and height"""
    X = df['height'].values
    y = df['fev'].values

    plot_model(X, y,
              title='Association between FEV and Height',
              filename='fev_height_association.png',
              show_data_label=False)


def fit_linear_model(df):
    """Fit and plot linear model for fev using height as predictor using scipy"""
    X = df['height'].values
    y = df['fev'].values

    # Fit linear regression using scipy.stats.linregress
    result = stats.linregress(X, y)
    slope = result.slope
    intercept = result.intercept

    # Predictions
    X_plot = np.linspace(X.min(), X.max(), 100)
    y_pred = slope * X_plot + intercept

    # Calculate MSE and R²
    y_train_pred = slope * X + intercept
    mse = mean_squared_error(y, y_train_pred)
    r2 = r2_score(y, y_train_pred)

    # Plot using common function
    plot_model(X, y, X_plot, y_pred,
              title=f'Linear Model: FEV vs Height\nR² = {r2:.4f}, MSE = {mse:.4f}',
              filename='fev_linear_model.png',
              curve_label=f'Linear fit: y = {slope:.4f}x + {intercept:.4f}',
              curve_color='r')

    print(f"\nLinear Model Results (using scipy.stats.linregress):")
    print(f"  Equation: FEV = {slope:.4f} × Height + {intercept:.4f}")
    print(f"  R² = {r2:.4f}")
    print(f"  MSE = {mse:.4f}")


def fit_natural_cubic_spline(df, df_val):
    """
    Fit natural cubic spline with specified degrees of freedom using scipy
    df_val: degrees of freedom (number of knots)
    """
    X = df['height'].values
    y = df['fev'].values

    # Sort data
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]

    # Remove duplicates by averaging
    unique_x, inverse = np.unique(X_sorted, return_inverse=True)
    unique_y = np.array([y_sorted[inverse == i].mean() for i in range(len(unique_x))])

    # For natural cubic spline: df = number of knots
    # Select knots at quantiles
    n_knots = df_val
    knot_positions = np.linspace(0, 1, n_knots)
    knots = np.quantile(unique_x, knot_positions)

    # Find closest data points to knots
    knot_indices = []
    for knot in knots:
        idx = np.argmin(np.abs(unique_x - knot))
        if idx not in knot_indices:
            knot_indices.append(idx)
    knot_indices = sorted(knot_indices)

    X_knots = unique_x[knot_indices]
    y_knots = unique_y[knot_indices]

    # Fit natural cubic spline using scipy
    spline = CubicSpline(X_knots, y_knots, bc_type='natural')

    # Predictions
    X_plot = np.linspace(X.min(), X.max(), 100)
    y_pred = spline(X_plot)

    # Calculate MSE on original data
    y_train_pred = spline(X_sorted)
    mse = mean_squared_error(y_sorted, y_train_pred)

    # Plot using common function
    plot_model(X, y, X_plot, y_pred,
              title=f'Natural Cubic Spline (df={df_val}): FEV vs Height\nMSE = {mse:.4f}',
              filename=f'fev_natural_spline_df{df_val}.png',
              curve_label=f'Natural Cubic Spline (df={df_val})',
              curve_color='b',
              knots_X=X_knots,
              knots_y=y_knots)

    print(f"\nNatural Cubic Spline (df={df_val}) Results:")
    print(f"  Number of knots: {len(X_knots)}")
    print(f"  MSE = {mse:.4f}")


def fit_smoothing_spline_cv(df):
    """
    Fit smoothing cubic spline with cross-validated smoothing parameter lambda
    Uses scipy.interpolate.UnivariateSpline with sklearn KFold cross-validation
    Number of folds is determined dynamically based on dataset size
    """
    X = df['height'].values
    y = df['fev'].values

    # Sort data and remove duplicates
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]

    # Remove duplicate X values by averaging y values
    unique_x, inverse_indices = np.unique(X_sorted, return_inverse=True)
    unique_y = np.array([y_sorted[inverse_indices == i].mean() for i in range(len(unique_x))])

    # Dynamically determine number of folds based on dataset size
    n = len(unique_x)
    if n < 30:
        k_folds = max(3, n // 5)  # At least 3 folds, ~5 samples per fold
    elif n < 100:
        k_folds = 5
    else:
        k_folds = 10

    print(f"  Dataset size (unique): {n}")
    print(f"  Using {k_folds}-fold cross-validation")

    # Test different smoothing parameters (lambda values)
    smoothing_params = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])

    print("  Performing cross-validation to find optimal λ...")

    best_lambda = None
    best_cv_mse = np.inf

    # Use sklearn's KFold for cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Perform k-fold cross-validation for each smoothing parameter
    for s in smoothing_params:
        fold_mses = []

        for train_idx, test_idx in kf.split(unique_x):
            X_train = unique_x[train_idx]
            X_test = unique_x[test_idx]
            y_train = unique_y[train_idx]
            y_test = unique_y[test_idx]

            if len(X_train) < 4 or len(X_test) < 1:
                continue

            try:
                # Fit smoothing spline on training data
                spline_cv = UnivariateSpline(X_train, y_train, s=s, k=3)
                y_pred = spline_cv(X_test)

                # Check for valid predictions
                if not np.any(np.isnan(y_pred)) and not np.any(np.isinf(y_pred)):
                    fold_mse = mean_squared_error(y_test, y_pred)
                    fold_mses.append(fold_mse)
            except:
                continue

        if len(fold_mses) > 0:
            avg_cv_mse = np.mean(fold_mses)
            print(f"    λ = {s:6.1f}: CV-MSE = {avg_cv_mse:.4f}")

            if avg_cv_mse < best_cv_mse:
                best_cv_mse = avg_cv_mse
                best_lambda = s

    if best_lambda is None:
        best_lambda = 5.0
        print(f"  Using default λ = {best_lambda}")
    else:
        print(f"\n  Best λ = {best_lambda:.1f} (CV-MSE = {best_cv_mse:.4f})")

    # Fit final model with best smoothing parameter on all data
    spline = UnivariateSpline(unique_x, unique_y, s=best_lambda, k=3)

    # Predictions
    X_plot = np.linspace(X.min(), X.max(), 100)
    y_pred = spline(X_plot)

    # Calculate training MSE on original data
    y_train_pred = spline(X_sorted)
    train_mse = mean_squared_error(y_sorted, y_train_pred)

    # Plot using common function
    plot_model(X, y, X_plot, y_pred,
              title=f'Smoothing Cubic Spline (λ={best_lambda:.1f}): FEV vs Height\nMSE = {train_mse:.4f}',
              filename='fev_smoothing_spline.png',
              curve_label=f'Smoothing Spline (λ={best_lambda:.1f})',
              curve_color='g')

    print(f"\nSmoothing Cubic Spline Results:")
    print(f"  Optimal λ = {best_lambda:.1f}")
    print(f"  CV-MSE = {best_cv_mse:.4f}")
    print(f"  Training MSE = {train_mse:.4f}")


def fit_bspline(df, df_val):
    """
    Fit cubic B-spline with specified degrees of freedom using scipy
    df_val: degrees of freedom
    For cubic B-splines: df = n_interior_knots + degree + 1
    With degree=3: n_interior_knots = df - 4
    """
    X = df['height'].values
    y = df['fev'].values

    # Sort data
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]

    # Calculate number of interior knots
    degree = 3  # Cubic spline
    n_interior_knots = df_val - degree - 1

    if n_interior_knots < 0:
        n_interior_knots = 0

    # Create interior knots at quantiles
    if n_interior_knots > 0:
        knot_positions = np.linspace(0, 1, n_interior_knots + 2)[1:-1]
        interior_knots = np.quantile(X_sorted, knot_positions)
    else:
        interior_knots = np.array([])

    # Fit B-spline using scipy.interpolate.splrep
    tck = splrep(X_sorted, y_sorted, t=interior_knots, k=degree)

    # Predictions
    X_plot = np.linspace(X.min(), X.max(), 100)
    y_pred = splev(X_plot, tck)

    # Calculate MSE
    y_train_pred = splev(X_sorted, tck)
    mse = mean_squared_error(y_sorted, y_train_pred)

    # Prepare knot positions for plotting
    knots_X_plot = None
    knots_y_plot = None
    if len(interior_knots) > 0:
        knots_X_plot = interior_knots
        knots_y_plot = splev(interior_knots, tck)

    # Plot using common function
    plot_model(X, y, X_plot, y_pred,
              title=f'Cubic B-Spline (df={df_val}): FEV vs Height\nMSE = {mse:.4f}',
              filename=f'fev_bspline_df{df_val}.png',
              curve_label=f'Cubic B-Spline (df={df_val})',
              curve_color='purple',
              knots_X=knots_X_plot,
              knots_y=knots_y_plot)

    print(f"\nCubic B-Spline (df={df_val}) Results:")
    print(f"  Degree: {degree}")
    print(f"  Number of interior knots: {len(interior_knots)}")
    print(f"  MSE = {mse:.4f}")


def plot_all_combined(df):
    """Create a combined plot showing all models overlaid together"""
    X = df['height'].values
    y = df['fev'].values

    # Sort data for splines
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    unique_x, inverse = np.unique(X_sorted, return_inverse=True)
    unique_y = np.array([y_sorted[inverse == i].mean() for i in range(len(unique_x))])

    # Create single figure for overlay comparison
    fig, ax = plt.subplots(figsize=(14, 8))

    X_plot = np.linspace(X.min(), X.max(), 100)

    # Plot data
    ax.scatter(X, y, alpha=0.3, edgecolors='k', facecolors='none', s=30, label='Data', zorder=1)

    # 1. Linear model
    result = stats.linregress(X, y)
    y_linear = result.slope * X_plot + result.intercept
    mse_linear = mean_squared_error(y, result.slope * X + result.intercept)
    ax.plot(X_plot, y_linear, 'r-', linewidth=2.5, label=f'Linear (MSE={mse_linear:.4f})', zorder=2)

    # 2. Natural cubic spline df=5
    knots = np.quantile(unique_x, np.linspace(0, 1, 5))
    knot_indices = sorted(set([np.argmin(np.abs(unique_x - k)) for k in knots]))
    spline_nat5 = CubicSpline(unique_x[knot_indices], unique_y[knot_indices], bc_type='natural')
    mse_nat5 = mean_squared_error(y_sorted, spline_nat5(X_sorted))
    ax.plot(X_plot, spline_nat5(X_plot), 'b-', linewidth=2.5, label=f'Nat. Cubic df=5 (MSE={mse_nat5:.4f})', zorder=3)

    # 3. Natural cubic spline df=10
    knots = np.quantile(unique_x, np.linspace(0, 1, 10))
    knot_indices = sorted(set([np.argmin(np.abs(unique_x - k)) for k in knots]))
    spline_nat10 = CubicSpline(unique_x[knot_indices], unique_y[knot_indices], bc_type='natural')
    mse_nat10 = mean_squared_error(y_sorted, spline_nat10(X_sorted))
    ax.plot(X_plot, spline_nat10(X_plot), 'b--', linewidth=2.5, label=f'Nat. Cubic df=10 (MSE={mse_nat10:.4f})', zorder=4)

    # 4. Smoothing spline
    spline_smooth = UnivariateSpline(unique_x, unique_y, s=5.0, k=3)
    mse_smooth = mean_squared_error(y_sorted, spline_smooth(X_sorted))
    ax.plot(X_plot, spline_smooth(X_plot), 'g-', linewidth=2.5, label=f'Smoothing λ=5.0 (MSE={mse_smooth:.4f})', zorder=5)

    # 5. B-spline df=5
    interior_knots = np.array([np.quantile(X_sorted, 0.5)])
    tck_bs5 = splrep(X_sorted, y_sorted, t=interior_knots, k=3)
    mse_bs5 = mean_squared_error(y_sorted, splev(X_sorted, tck_bs5))
    ax.plot(X_plot, splev(X_plot, tck_bs5), color='purple', linewidth=2.5, label=f'B-spline df=5 (MSE={mse_bs5:.4f})', zorder=6)

    # 6. B-spline df=10
    interior_knots = np.quantile(X_sorted, np.linspace(0, 1, 8)[1:-1])
    tck_bs10 = splrep(X_sorted, y_sorted, t=interior_knots, k=3)
    mse_bs10 = mean_squared_error(y_sorted, splev(X_sorted, tck_bs10))
    ax.plot(X_plot, splev(X_plot, tck_bs10), color='purple', linestyle='--', linewidth=2.5, label=f'B-spline df=10 (MSE={mse_bs10:.4f})', zorder=7)

    ax.set_xlabel('Height', fontsize=14, fontweight='bold')
    ax.set_ylabel('FEV', fontsize=14, fontweight='bold')
    ax.set_title('All Models Comparison: FEV vs Height', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fev_all_models_combined.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nCombined plot saved as 'fev_all_models_combined.png'")


def main():
    """Main analysis pipeline"""
    print("="*70)
    print("FEV SPLINE REGRESSION ANALYSIS (using scipy)")
    print("="*70)
    print()

    # Load data
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Height range: [{df['height'].min():.1f}, {df['height'].max():.1f}]")
    print(f"FEV range: [{df['fev'].min():.3f}, {df['fev'].max():.3f}]")
    print()

    # Plot 1: Association between fev and height
    print("Creating association plot...")
    plot_association(df)
    print()

    # Plot 2: Linear model fit
    print("Fitting linear model...")
    fit_linear_model(df)
    print()

    # Plot 3: Natural cubic spline with df=5
    print("Fitting natural cubic spline with df=5...")
    fit_natural_cubic_spline(df, 5)
    print()

    # Plot 4: Natural cubic spline with df=10
    print("Fitting natural cubic spline with df=10...")
    fit_natural_cubic_spline(df, 10)
    print()

    # Plot 5: Smoothing cubic spline with cross-validated lambda
    print("Fitting smoothing cubic spline with cross-validation...")
    fit_smoothing_spline_cv(df)
    print()

    # Plot 6: Cubic B-spline with df=5
    print("Fitting cubic B-spline with df=5...")
    fit_bspline(df, 5)
    print()

    # Plot 7: Cubic B-spline with df=10
    print("Fitting cubic B-spline with df=10...")
    fit_bspline(df, 10)
    print()

    # Plot 8: Combined plot of all models
    print("="*70)
    print("Creating combined comparison plot...")
    plot_all_combined(df)


if __name__ == "__main__":
    main()
