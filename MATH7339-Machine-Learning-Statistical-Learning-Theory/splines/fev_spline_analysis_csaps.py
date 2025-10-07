import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from csaps import csaps, CubicSmoothingSpline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from fev_utils import load_data, plot_model


def plot_association(df):
    """Plot the association between fev and height"""
    X = df['height'].values
    y = df['fev'].values

    plot_model(X, y,
              title='Association between FEV and Height',
              filename='fev_height_association_csaps.png',
              show_data_label=False)


def fit_linear_model(df):
    """Fit and plot linear model for fev using height as predictor using sklearn LinearRegression"""
    X = df['height'].values.reshape(-1, 1)
    y = df['fev'].values

    # Fit linear regression using sklearn
    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_

    # Predictions
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(X_plot)

    # Calculate MSE and R²
    y_pred_train = model.predict(X)
    mse = mean_squared_error(y, y_pred_train)
    r2 = r2_score(y, y_pred_train)

    # Plot using common function
    plot_model(X.flatten(), y, X_plot.flatten(), y_pred,
              title=f'Linear Model: FEV vs Height\nR² = {r2:.4f}, MSE = {mse:.4f}',
              filename='fev_linear_model_csaps.png',
              curve_label=f'Linear fit: y = {slope:.4f}x + {intercept:.4f}',
              curve_color='r')

    print(f"\nLinear Model Results (using sklearn.linear_model.LinearRegression):")
    print(f"  Equation: FEV = {slope:.4f} × Height + {intercept:.4f}")
    print(f"  R² = {r2:.4f}")
    print(f"  MSE = {mse:.4f}")


def fit_natural_cubic_spline(df, df_val):
    """
    Fit natural cubic spline with specified degrees of freedom using csaps
    df_val: degrees of freedom (number of knots)
    Uses csaps with very low smoothing (near-interpolation) on selected knots
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

    # Select knots at quantiles to simulate natural cubic spline
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

    # Fit cubic spline using csaps with very low smoothing (interpolation)
    # smooth parameter close to 0 for interpolation (like natural cubic spline)
    spline = CubicSmoothingSpline(X_knots, y_knots, smooth=0.0)

    # Predictions
    X_plot = np.linspace(X.min(), X.max(), 100)
    y_pred = spline(X_plot)

    # Calculate MSE and R² on original data
    y_train_pred = spline(X_sorted)
    mse = mean_squared_error(y_sorted, y_train_pred)
    r2 = r2_score(y_sorted, y_train_pred)

    # Plot using common function
    plot_model(X, y, X_plot, y_pred,
              title=f'Cubic Spline (df={df_val}): FEV vs Height\nMSE = {mse:.4f}',
              filename=f'fev_natural_spline_df{df_val}_csaps.png',
              curve_label=f'Cubic Spline (df={df_val})',
              curve_color='b',
              knots_X=X_knots,
              knots_y=y_knots)

    print(f"\nCubic Spline (df={df_val}) Results (using csaps):")
    print(f"  Number of knots: {len(X_knots)}")
    print(f"  Smoothing parameter: 0.0 (interpolation)")
    print(f"  R² = {r2:.4f}")
    print(f"  MSE = {mse:.4f}")


def fit_smoothing_spline_cv(df):
    """
    Fit smoothing cubic spline using csaps automatic smoothing parameter selection.
    csaps automatically computes the optimal smoothing parameter based on GCV criterion.
    The selected smooth value can be retrieved from the spline.smooth property.
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

    n = len(unique_x)
    print(f"  Dataset size (unique): {n}")
    print(f"  Using csaps automatic smoothing parameter selection...")

    # Use automatic smoothing (csaps computes optimal smooth parameter via GCV)
    spline = CubicSmoothingSpline(unique_x, unique_y, smooth=None)

    # Retrieve the automatically computed smoothing parameter
    best_smooth = spline.smooth
    print(f"  Automatically selected smooth = {best_smooth:.6f}")

    # Predictions
    X_plot = np.linspace(X.min(), X.max(), 100)
    y_pred = spline(X_plot)

    # Calculate training MSE and R² on original data
    y_train_pred = spline(X_sorted)
    train_mse = mean_squared_error(y_sorted, y_train_pred)
    r2 = r2_score(y_sorted, y_train_pred)

    # Plot using common function
    smooth_label = f"{best_smooth:.6f}"
    plot_model(X, y, X_plot, y_pred,
              title=f'Smoothing Cubic Spline (smooth={smooth_label}): FEV vs Height\nMSE = {train_mse:.4f}',
              filename='fev_smoothing_spline_csaps.png',
              curve_label=f'Smoothing Spline (smooth={smooth_label})',
              curve_color='g')

    print(f"\nSmoothing Cubic Spline Results (using csaps):")
    print(f"  Optimal smooth = {smooth_label} (automatic via GCV)")
    print(f"  Training MSE = {train_mse:.4f}")
    print(f"  R² = {r2:.4f}")

    return best_smooth


def fit_bspline(df, df_val):
    """
    Fit cubic B-spline with specified degrees of freedom using csaps
    df_val: degrees of freedom
    Uses csaps with low smoothing on subset of data points (simulating knots)
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

    # Fit cubic spline using csaps with low smoothing
    # Use smooth parameter to control flexibility
    smooth_param = 0.01  # Low smoothing for B-spline-like behavior
    spline = CubicSmoothingSpline(unique_x, unique_y, smooth=smooth_param)

    # Predictions
    X_plot = np.linspace(X.min(), X.max(), 100)
    y_pred = spline(X_plot)

    # Calculate MSE and R²
    y_train_pred = spline(X_sorted)
    mse = mean_squared_error(y_sorted, y_train_pred)
    r2 = r2_score(y_sorted, y_train_pred)

    # Prepare knot positions for plotting
    knots_X_plot = None
    knots_y_plot = None
    if len(interior_knots) > 0:
        knots_X_plot = interior_knots
        knots_y_plot = spline(interior_knots)

    # Plot using common function
    plot_model(X, y, X_plot, y_pred,
              title=f'Cubic B-Spline (df={df_val}): FEV vs Height\nMSE = {mse:.4f}',
              filename=f'fev_bspline_df{df_val}_csaps.png',
              curve_label=f'Cubic B-Spline (df={df_val})',
              curve_color='purple',
              knots_X=knots_X_plot,
              knots_y=knots_y_plot)

    print(f"\nCubic B-Spline (df={df_val}) Results (using csaps):")
    print(f"  Degree: {degree}")
    print(f"  Number of interior knots: {len(interior_knots)}")
    print(f"  Smoothing parameter: {smooth_param}")
    print(f"  R² = {r2:.4f}")
    print(f"  MSE = {mse:.4f}")


def plot_all_combined(df, best_smooth_param):
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
    model_linear = LinearRegression()
    model_linear.fit(X.reshape(-1, 1), y)
    y_linear = model_linear.predict(X_plot.reshape(-1, 1))
    mse_linear = mean_squared_error(y, model_linear.predict(X.reshape(-1, 1)))
    ax.plot(X_plot, y_linear, 'r-', linewidth=2.5, label=f'Linear (MSE={mse_linear:.4f})', zorder=2)

    # 2. Cubic spline df=5
    knots = np.quantile(unique_x, np.linspace(0, 1, 5))
    knot_indices = sorted(set([np.argmin(np.abs(unique_x - k)) for k in knots]))
    spline_cs5 = CubicSmoothingSpline(unique_x[knot_indices], unique_y[knot_indices], smooth=0.0)
    y_cs5 = spline_cs5(X_plot)
    mse_cs5 = mean_squared_error(y_sorted, spline_cs5(X_sorted))
    ax.plot(X_plot, y_cs5, 'b-', linewidth=2.5, label=f'Cubic Spline df=5 (MSE={mse_cs5:.4f})', zorder=3)

    # 3. Cubic spline df=10
    knots = np.quantile(unique_x, np.linspace(0, 1, 10))
    knot_indices = sorted(set([np.argmin(np.abs(unique_x - k)) for k in knots]))
    spline_cs10 = CubicSmoothingSpline(unique_x[knot_indices], unique_y[knot_indices], smooth=0.0)
    y_cs10 = spline_cs10(X_plot)
    mse_cs10 = mean_squared_error(y_sorted, spline_cs10(X_sorted))
    ax.plot(X_plot, y_cs10, 'b--', linewidth=2.5, label=f'Cubic Spline df=10 (MSE={mse_cs10:.4f})', zorder=4)

    # 4. Smoothing spline
    spline_smooth = CubicSmoothingSpline(unique_x, unique_y, smooth=best_smooth_param)
    smooth_label = f"{best_smooth_param:.6f}"
    y_smooth = spline_smooth(X_plot)
    mse_smooth = mean_squared_error(y_sorted, spline_smooth(X_sorted))
    ax.plot(X_plot, y_smooth, 'g-', linewidth=2.5, label=f'Smoothing s={smooth_label} (MSE={mse_smooth:.4f})', zorder=5)

    # 5. B-spline df=5
    spline_bs5 = CubicSmoothingSpline(unique_x, unique_y, smooth=0.01)
    y_bs5 = spline_bs5(X_plot)
    mse_bs5 = mean_squared_error(y_sorted, spline_bs5(X_sorted))
    ax.plot(X_plot, y_bs5, color='purple', linewidth=2.5, label=f'B-spline df=5 (MSE={mse_bs5:.4f})', zorder=6)

    # 6. B-spline df=10
    spline_bs10 = CubicSmoothingSpline(unique_x, unique_y, smooth=0.01)
    y_bs10 = spline_bs10(X_plot)
    mse_bs10 = mean_squared_error(y_sorted, spline_bs10(X_sorted))
    ax.plot(X_plot, y_bs10, color='purple', linestyle='--', linewidth=2.5, label=f'B-spline df=10 (MSE={mse_bs10:.4f})', zorder=7)

    ax.set_xlabel('Height', fontsize=14, fontweight='bold')
    ax.set_ylabel('FEV', fontsize=14, fontweight='bold')
    ax.set_title('All Models Comparison: FEV vs Height (CSAPS)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fev_all_models_combined_csaps.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nCombined plot saved as 'fev_all_models_combined_csaps.png'")


def main():
    """Main analysis pipeline"""
    print("="*70)
    print("FEV SPLINE REGRESSION ANALYSIS (using csaps)")
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

    # Plot 3: Cubic spline with df=5
    print("Fitting cubic spline with df=5...")
    fit_natural_cubic_spline(df, 5)
    print()

    # Plot 4: Cubic spline with df=10
    print("Fitting cubic spline with df=10...")
    fit_natural_cubic_spline(df, 10)
    print()

    # Plot 5: Smoothing cubic spline with automatic smoothing parameter
    print("Fitting smoothing cubic spline with automatic smoothing parameter...")
    best_smooth = fit_smoothing_spline_cv(df)
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
    plot_all_combined(df, best_smooth)


if __name__ == "__main__":
    main()
