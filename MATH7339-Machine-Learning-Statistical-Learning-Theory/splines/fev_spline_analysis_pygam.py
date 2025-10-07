"""
FEV Spline Analysis - PyGAM Implementation
Analysis of forced expiratory volume (FEV) using various spline regression techniques
Uses pyGAM instead of scipy for all spline implementations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s, l
from sklearn.metrics import mean_squared_error, r2_score
from fev_utils import load_data, plot_model


def plot_association(df):
    """Plot the association between fev and height"""
    X = df['height'].values
    y = df['fev'].values

    plot_model(X, y,
              title='Association between FEV and Height',
              filename='fev_height_association_pygam.png',
              show_data_label=False)


def fit_linear_model(df):
    """Fit and plot linear model for fev using height as predictor using pyGAM"""
    X = df['height'].values.reshape(-1, 1)
    y = df['fev'].values

    # Fit linear model using pyGAM with l() (linear term)
    gam = LinearGAM(l(0))
    gam.fit(X, y)

    # Get parameters
    # For linear model: y = intercept + slope * x
    intercept = gam.coef_[0]
    slope = gam.coef_[1]

    # Predictions
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = gam.predict(X_plot)

    # Calculate MSE and R²
    y_pred_train = gam.predict(X)
    mse = mean_squared_error(y, y_pred_train)
    r2 = r2_score(y, y_pred_train)

    # Plot using common function
    plot_model(X.flatten(), y, X_plot.flatten(), y_pred,
              title=f'Linear Model: FEV vs Height\nR² = {r2:.4f}, MSE = {mse:.4f}',
              filename='fev_linear_model_pygam.png',
              curve_label=f'Linear fit: y = {slope:.4f}x + {intercept:.4f}',
              curve_color='r')

    print(f"\nLinear Model Results (using pyGAM LinearGAM with l()):")
    print(f"  Equation: FEV = {slope:.4f} × Height + {intercept:.4f}")
    print(f"  R² = {r2:.4f}")
    print(f"  MSE = {mse:.4f}")


def fit_natural_cubic_spline(df, df_val):
    """
    Fit natural cubic spline with specified degrees of freedom using pyGAM
    df_val: degrees of freedom (number of splines/basis functions)
    Uses pyGAM's s() with constraints='natural'
    """
    X = df['height'].values.reshape(-1, 1)
    y = df['fev'].values

    # Fit natural cubic spline using pyGAM
    # n_splines parameter controls degrees of freedom
    gam = LinearGAM(s(0, n_splines=df_val, constraints='natural'))
    gam.fit(X, y)

    # Predictions
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = gam.predict(X_plot)

    # Calculate MSE and R²
    y_train_pred = gam.predict(X)
    mse = mean_squared_error(y, y_train_pred)
    r2 = r2_score(y, y_train_pred)

    # Get knot positions
    # PyGAM places knots evenly across the data range
    knots = np.linspace(X.min(), X.max(), df_val)
    knots_y = gam.predict(knots.reshape(-1, 1))

    # Plot using common function
    plot_model(X.flatten(), y, X_plot.flatten(), y_pred,
              title=f'Natural Cubic Spline (df={df_val}): FEV vs Height\nMSE = {mse:.4f}',
              filename=f'fev_natural_spline_df{df_val}_pygam.png',
              curve_label=f'Natural Cubic Spline (df={df_val})',
              curve_color='b',
              knots_X=knots.flatten(),
              knots_y=knots_y)

    print(f"\nNatural Cubic Spline (df={df_val}) Results:")
    print(f"  Number of splines: {df_val}")
    print(f"  R² = {r2:.4f}")
    print(f"  MSE = {mse:.4f}")


def fit_smoothing_spline_cv(df):
    """
    Fit smoothing cubic spline with optimal smoothing parameter lambda
    Uses pyGAM's gridsearch with GCV (Generalized Cross-Validation) criterion
    """
    X = df['height'].values.reshape(-1, 1)
    y = df['fev'].values

    n = len(X)
    print(f"  Dataset size: {n}")
    print(f"  Using pyGAM's gridsearch with GCV objective...")

    # Fixed number of splines (basis functions) for smoothing spline
    n_splines = min(20, n // 5)

    # Test different lambda (penalty) parameters using log scale
    # Larger lambda = more smoothing
    # np.logspace(-2, 4, 11) creates 11 values from 10^-2 to 10^4
    lambda_params = np.logspace(-2, 4, 11)

    print("  Performing grid search to find optimal λ...")

    # Create initial GAM with spline term
    gam = LinearGAM(s(0, n_splines=n_splines))

    # Use pyGAM's gridsearch to find optimal lambda
    # gridsearch uses GCV (Generalized Cross-Validation) or UBRE by default
    gam.gridsearch(X, y, lam=lambda_params, progress=False)

    # Get the selected optimal lambda
    best_lambda = gam.lam[0][0]  # Extract lambda for term 0, parameter 0

    print(f"\n  Best λ = {best_lambda:.2f} (selected via GCV)")

    # Predictions
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = gam.predict(X_plot)

    # Calculate training MSE and R²
    y_train_pred = gam.predict(X)
    train_mse = mean_squared_error(y, y_train_pred)
    r2 = r2_score(y, y_train_pred)

    # Plot using common function
    plot_model(X.flatten(), y, X_plot.flatten(), y_pred,
              title=f'Smoothing Spline (λ={best_lambda:.2f}): FEV vs Height\nMSE = {train_mse:.4f}',
              filename='fev_smoothing_spline_pygam.png',
              curve_label=f'Penalized Spline (λ={best_lambda:.2f})',
              curve_color='g')

    print(f"\nSmoothing Spline (Penalized Spline) Results:")
    print(f"  Number of splines: {n_splines}")
    print(f"  Optimal λ = {best_lambda:.2f}")
    print(f"  Training MSE = {train_mse:.4f}")
    print(f"  R² = {r2:.4f}")


def fit_bspline(df, df_val):
    """
    Fit cubic B-spline with specified degrees of freedom using pyGAM
    df_val: degrees of freedom (number of basis functions)
    Uses pyGAM's s() with spline_order=3 (cubic) and no penalty
    """
    X = df['height'].values.reshape(-1, 1)
    y = df['fev'].values

    # Fit B-spline using pyGAM
    # n_splines controls the number of basis functions (df)
    # spline_order=3 for cubic splines
    # lam=0.001 for minimal penalty (unpenalized)
    gam = LinearGAM(s(0, n_splines=df_val, spline_order=3, lam=0.001))
    gam.fit(X, y)

    # Predictions
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = gam.predict(X_plot)

    # Calculate MSE and R²
    y_train_pred = gam.predict(X)
    mse = mean_squared_error(y, y_train_pred)
    r2 = r2_score(y, y_train_pred)

    # Get knot positions (interior knots)
    # For B-splines: n_interior_knots = df - spline_order - 1
    spline_order = 3
    n_interior_knots = df_val - spline_order - 1

    knots_X_plot = None
    knots_y_plot = None
    if n_interior_knots > 0:
        knot_positions = np.linspace(0, 1, n_interior_knots + 2)[1:-1]
        knots_X_plot = np.quantile(X, knot_positions)
        knots_y_plot = gam.predict(knots_X_plot.reshape(-1, 1))

    # Plot using common function
    plot_model(X.flatten(), y, X_plot.flatten(), y_pred,
              title=f'Cubic B-Spline (df={df_val}): FEV vs Height\nMSE = {mse:.4f}',
              filename=f'fev_bspline_df{df_val}_pygam.png',
              curve_label=f'Cubic B-Spline (df={df_val})',
              curve_color='purple',
              knots_X=knots_X_plot.flatten() if knots_X_plot is not None else None,
              knots_y=knots_y_plot if knots_y_plot is not None else None)

    print(f"\nCubic B-Spline (df={df_val}) Results:")
    print(f"  Spline order: {spline_order}")
    print(f"  Number of splines: {df_val}")
    print(f"  Number of interior knots: {n_interior_knots}")
    print(f"  R² = {r2:.4f}")
    print(f"  MSE = {mse:.4f}")


def plot_all_combined(df):
    """Create a combined plot showing all models overlaid together"""
    X = df['height'].values.reshape(-1, 1)
    y = df['fev'].values

    # Create single figure for overlay comparison
    fig, ax = plt.subplots(figsize=(14, 8))

    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

    # Plot data
    ax.scatter(X, y, alpha=0.3, edgecolors='k', facecolors='none', s=30, label='Data', zorder=1)

    # 1. Linear model
    gam_linear = LinearGAM(l(0))
    gam_linear.fit(X, y)
    y_linear = gam_linear.predict(X_plot)
    mse_linear = mean_squared_error(y, gam_linear.predict(X))
    ax.plot(X_plot, y_linear, 'r-', linewidth=2.5, label=f'Linear (MSE={mse_linear:.4f})', zorder=2)

    # 2. Natural cubic spline df=5
    gam_nat5 = LinearGAM(s(0, n_splines=5, constraints='natural'))
    gam_nat5.fit(X, y)
    y_nat5 = gam_nat5.predict(X_plot)
    mse_nat5 = mean_squared_error(y, gam_nat5.predict(X))
    ax.plot(X_plot, y_nat5, 'b-', linewidth=2.5, label=f'Nat. Cubic df=5 (MSE={mse_nat5:.4f})', zorder=3)

    # 3. Natural cubic spline df=10
    gam_nat10 = LinearGAM(s(0, n_splines=10, constraints='natural'))
    gam_nat10.fit(X, y)
    y_nat10 = gam_nat10.predict(X_plot)
    mse_nat10 = mean_squared_error(y, gam_nat10.predict(X))
    ax.plot(X_plot, y_nat10, 'b--', linewidth=2.5, label=f'Nat. Cubic df=10 (MSE={mse_nat10:.4f})', zorder=4)

    # 4. Smoothing spline
    n_splines = min(20, len(X) // 5)
    gam_smooth = LinearGAM(s(0, n_splines=n_splines, lam=1.0))
    gam_smooth.fit(X, y)
    y_smooth = gam_smooth.predict(X_plot)
    mse_smooth = mean_squared_error(y, gam_smooth.predict(X))
    ax.plot(X_plot, y_smooth, 'g-', linewidth=2.5, label=f'Smoothing λ=1.0 (MSE={mse_smooth:.4f})', zorder=5)

    # 5. B-spline df=5
    gam_bs5 = LinearGAM(s(0, n_splines=5, spline_order=3, lam=0.001))
    gam_bs5.fit(X, y)
    y_bs5 = gam_bs5.predict(X_plot)
    mse_bs5 = mean_squared_error(y, gam_bs5.predict(X))
    ax.plot(X_plot, y_bs5, color='purple', linewidth=2.5, label=f'B-spline df=5 (MSE={mse_bs5:.4f})', zorder=6)

    # 6. B-spline df=10
    gam_bs10 = LinearGAM(s(0, n_splines=10, spline_order=3, lam=0.001))
    gam_bs10.fit(X, y)
    y_bs10 = gam_bs10.predict(X_plot)
    mse_bs10 = mean_squared_error(y, gam_bs10.predict(X))
    ax.plot(X_plot, y_bs10, color='purple', linestyle='--', linewidth=2.5, label=f'B-spline df=10 (MSE={mse_bs10:.4f})', zorder=7)

    ax.set_xlabel('Height', fontsize=14, fontweight='bold')
    ax.set_ylabel('FEV', fontsize=14, fontweight='bold')
    ax.set_title('All Models Comparison: FEV vs Height (PyGAM)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fev_all_models_combined_pygam.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nCombined plot saved as 'fev_all_models_combined_pygam.png'")


def main():
    """Main analysis pipeline"""
    print("="*70)
    print("FEV SPLINE REGRESSION ANALYSIS (using pyGAM)")
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
    print("Fitting smoothing spline (penalized spline) with cross-validation...")
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
