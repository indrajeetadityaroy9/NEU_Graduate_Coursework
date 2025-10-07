import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.gam.api import GLMGam, BSplines
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from fev_utils import load_data, plot_model


def plot_association(df):
    """Plot the association between fev and height"""
    X = df['height'].values
    y = df['fev'].values

    plot_model(X, y,
              title='Association between FEV and Height',
              filename='fev_height_association_sm.png',
              show_data_label=False)


def fit_linear_model(df):
    """Fit and plot linear model for fev using height as predictor using statsmodels"""
    X = df['height'].values
    y = df['fev'].values

    # Fit linear regression using statsmodels formula API
    result = smf.ols('fev ~ height', data=df).fit()

    intercept = result.params['Intercept']
    slope = result.params['height']

    # Predictions
    X_plot = np.linspace(X.min(), X.max(), 100)
    df_plot = pd.DataFrame({'height': X_plot})
    y_pred = result.predict(df_plot)

    # Calculate MSE and R²
    y_train_pred = result.predict(df)
    mse = mean_squared_error(y, y_train_pred)
    r2 = r2_score(y, y_train_pred)

    # Plot using common function
    plot_model(X, y, X_plot, y_pred,
              title=f'Linear Model: FEV vs Height\nR² = {r2:.4f}, MSE = {mse:.4f}',
              filename='fev_linear_model_sm.png',
              curve_label=f'Linear fit: y = {slope:.4f}x + {intercept:.4f}',
              curve_color='r')

    print(f"\nLinear Model Results (using statsmodels.formula.api.ols):")
    print(f"  Equation: FEV = {slope:.4f} × Height + {intercept:.4f}")
    print(f"  R² = {r2:.4f}")
    print(f"  MSE = {mse:.4f}")


def fit_natural_cubic_spline(df, df_val):
    """
    Fit natural cubic spline with specified degrees of freedom using statsmodels/patsy
    df_val: degrees of freedom
    Uses patsy's cr() for natural cubic regression splines
    """
    X = df['height'].values
    y = df['fev'].values

    # Fit natural cubic spline using formula API
    # cr() creates natural cubic regression splines
    formula = f'fev ~ cr(height, df={df_val})'
    result = smf.ols(formula, data=df).fit()

    # Predictions for plotting
    X_plot = np.linspace(X.min(), X.max(), 100)
    df_plot = pd.DataFrame({'height': X_plot})
    y_pred = result.predict(df_plot)

    # Calculate MSE and R²
    y_train_pred = result.predict(df)
    mse = mean_squared_error(y, y_train_pred)
    r2 = r2_score(y, y_train_pred)

    # Get knot positions (approximate from quantiles)
    knot_positions = np.linspace(0, 1, df_val)
    X_knots = np.quantile(X, knot_positions)
    df_knots = pd.DataFrame({'height': X_knots})
    y_knots = result.predict(df_knots)

    # Plot using common function
    plot_model(X, y, X_plot, y_pred,
              title=f'Natural Cubic Spline (df={df_val}): FEV vs Height\nMSE = {mse:.4f}',
              filename=f'fev_natural_spline_df{df_val}_sm.png',
              curve_label=f'Natural Cubic Spline (df={df_val})',
              curve_color='b',
              knots_X=X_knots,
              knots_y=y_knots)

    print(f"\nNatural Cubic Spline (df={df_val}) Results:")
    print(f"  Number of basis functions: {df_val}")
    print(f"  R² = {r2:.4f}")
    print(f"  MSE = {mse:.4f}")


def fit_smoothing_spline_cv(df):
    """
    Fit smoothing cubic spline with cross-validated smoothing parameter
    Uses statsmodels GLMGam with penalized B-splines
    """
    X = df['height'].values
    y = df['fev'].values

    # Sort data
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]

    # Dynamically determine number of folds
    n = len(X_sorted)
    if n < 30:
        k_folds = max(3, n // 5)
    elif n < 100:
        k_folds = 5
    else:
        k_folds = 10

    print(f"  Dataset size: {n}")
    print(f"  Using {k_folds}-fold cross-validation")

    # Test different alpha (penalty/smoothing) parameters
    # Larger alpha = more smoothing (equivalent to larger lambda in scipy)
    alpha_params = np.array([0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0])

    print("  Performing cross-validation to find optimal α...")

    best_alpha = None
    best_cv_mse = np.inf

    # Use sklearn's KFold for cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Create B-spline basis with moderate number of knots
    n_splines = min(20, n // 10)  # Adaptive number of basis functions
    x_spline = np.linspace(X_sorted.min(), X_sorted.max(), n_splines)
    bs = BSplines(x_spline, degree=[3], include_intercept=True)

    for alpha in alpha_params:
        fold_mses = []

        for train_idx, test_idx in kf.split(X_sorted):
            X_train = X_sorted[train_idx]
            X_test = X_sorted[test_idx]
            y_train = y_sorted[train_idx]
            y_test = y_sorted[test_idx]

            if len(X_train) < 4 or len(X_test) < 1:
                continue

            try:
                # Fit penalized GAM on training data
                gam = GLMGam(y_train, smoother=bs, exog=X_train.reshape(-1, 1), alpha=alpha)
                result_cv = gam.fit()

                # Predict on test data
                y_pred = result_cv.predict(exog=X_test.reshape(-1, 1))

                # Check for valid predictions
                if not np.any(np.isnan(y_pred)) and not np.any(np.isinf(y_pred)):
                    fold_mse = mean_squared_error(y_test, y_pred)
                    fold_mses.append(fold_mse)
            except:
                continue

        if len(fold_mses) > 0:
            avg_cv_mse = np.mean(fold_mses)
            print(f"    α = {alpha:8.1f}: CV-MSE = {avg_cv_mse:.4f}")

            if avg_cv_mse < best_cv_mse:
                best_cv_mse = avg_cv_mse
                best_alpha = alpha

    if best_alpha is None:
        best_alpha = 100.0
        print(f"  Using default α = {best_alpha}")
    else:
        print(f"\n  Best α = {best_alpha:.1f} (CV-MSE = {best_cv_mse:.4f})")

    # Fit final model with best smoothing parameter on all data
    gam_final = GLMGam(y_sorted, smoother=bs, exog=X_sorted.reshape(-1, 1), alpha=best_alpha)
    result = gam_final.fit()

    # Predictions
    X_plot = np.linspace(X.min(), X.max(), 100)
    y_pred = result.predict(exog=X_plot.reshape(-1, 1))

    # Calculate training MSE and R²
    y_train_pred = result.predict(exog=X_sorted.reshape(-1, 1))
    train_mse = mean_squared_error(y_sorted, y_train_pred)
    r2 = r2_score(y_sorted, y_train_pred)

    # Plot using common function
    plot_model(X, y, X_plot, y_pred,
              title=f'Smoothing Spline (α={best_alpha:.1f}): FEV vs Height\nMSE = {train_mse:.4f}',
              filename='fev_smoothing_spline_sm.png',
              curve_label=f'Penalized B-spline (α={best_alpha:.1f})',
              curve_color='g')

    print(f"\nSmoothing Spline (Penalized B-spline) Results:")
    print(f"  Optimal α = {best_alpha:.1f}")
    print(f"  CV-MSE = {best_cv_mse:.4f}")
    print(f"  Training MSE = {train_mse:.4f}")
    print(f"  R² = {r2:.4f}")


def fit_bspline(df, df_val):
    """
    Fit cubic B-spline with specified degrees of freedom using statsmodels/patsy
    df_val: degrees of freedom
    Uses patsy's bs() for B-splines
    """
    X = df['height'].values
    y = df['fev'].values

    # Fit B-spline using formula API
    # bs() creates B-spline basis functions
    formula = f'fev ~ bs(height, df={df_val}, degree=3)'
    result = smf.ols(formula, data=df).fit()

    # Predictions for plotting
    X_plot = np.linspace(X.min(), X.max(), 100)
    df_plot = pd.DataFrame({'height': X_plot})
    y_pred = result.predict(df_plot)

    # Calculate MSE and R²
    y_train_pred = result.predict(df)
    mse = mean_squared_error(y, y_train_pred)
    r2 = r2_score(y, y_train_pred)

    # Get knot positions (interior knots)
    # For B-splines: n_interior_knots = df - degree - 1
    degree = 3
    n_interior_knots = df_val - degree - 1

    knots_X_plot = None
    knots_y_plot = None
    if n_interior_knots > 0:
        knot_positions = np.linspace(0, 1, n_interior_knots + 2)[1:-1]
        knots_X_plot = np.quantile(X, knot_positions)
        df_knots = pd.DataFrame({'height': knots_X_plot})
        knots_y_plot = result.predict(df_knots)

    # Plot using common function
    plot_model(X, y, X_plot, y_pred,
              title=f'Cubic B-Spline (df={df_val}): FEV vs Height\nMSE = {mse:.4f}',
              filename=f'fev_bspline_df{df_val}_sm.png',
              curve_label=f'Cubic B-Spline (df={df_val})',
              curve_color='purple',
              knots_X=knots_X_plot,
              knots_y=knots_y_plot)

    print(f"\nCubic B-Spline (df={df_val}) Results:")
    print(f"  Degree: {degree}")
    print(f"  Number of basis functions: {df_val}")
    print(f"  Number of interior knots: {n_interior_knots}")
    print(f"  R² = {r2:.4f}")
    print(f"  MSE = {mse:.4f}")


def plot_all_combined(df):
    """Create a combined plot showing all models overlaid together"""
    X = df['height'].values
    y = df['fev'].values

    # Create single figure for overlay comparison
    fig, ax = plt.subplots(figsize=(14, 8))

    X_plot = np.linspace(X.min(), X.max(), 100)
    df_plot = pd.DataFrame({'height': X_plot})

    # Plot data
    ax.scatter(X, y, alpha=0.3, edgecolors='k', facecolors='none', s=30, label='Data', zorder=1)

    # 1. Linear model using formula API
    model_linear = smf.ols('fev ~ height', data=df).fit()
    y_linear = model_linear.predict(df_plot)
    mse_linear = mean_squared_error(y, model_linear.predict(df))
    ax.plot(X_plot, y_linear, 'r-', linewidth=2.5, label=f'Linear (MSE={mse_linear:.4f})', zorder=2)

    # 2. Natural cubic spline df=5
    model_nat5 = smf.ols('fev ~ cr(height, df=5)', data=df).fit()
    y_nat5 = model_nat5.predict(df_plot)
    mse_nat5 = mean_squared_error(y, model_nat5.predict(df))
    ax.plot(X_plot, y_nat5, 'b-', linewidth=2.5, label=f'Nat. Cubic df=5 (MSE={mse_nat5:.4f})', zorder=3)

    # 3. Natural cubic spline df=10
    model_nat10 = smf.ols('fev ~ cr(height, df=10)', data=df).fit()
    y_nat10 = model_nat10.predict(df_plot)
    mse_nat10 = mean_squared_error(y, model_nat10.predict(df))
    ax.plot(X_plot, y_nat10, 'b--', linewidth=2.5, label=f'Nat. Cubic df=10 (MSE={mse_nat10:.4f})', zorder=4)

    # 4. Smoothing spline (Penalized B-spline)
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    n_splines = min(20, len(X) // 10)
    x_spline = np.linspace(X_sorted.min(), X_sorted.max(), n_splines)
    bs = BSplines(x_spline, degree=[3], include_intercept=True)
    gam = GLMGam(y_sorted, smoother=bs, exog=X_sorted.reshape(-1, 1), alpha=100.0)
    result_smooth = gam.fit()
    y_smooth = result_smooth.predict(exog=X_plot.reshape(-1, 1))
    mse_smooth = mean_squared_error(y_sorted, result_smooth.predict(exog=X_sorted.reshape(-1, 1)))
    ax.plot(X_plot, y_smooth, 'g-', linewidth=2.5, label=f'Smoothing α=100.0 (MSE={mse_smooth:.4f})', zorder=5)

    # 5. B-spline df=5
    model_bs5 = smf.ols('fev ~ bs(height, df=5, degree=3)', data=df).fit()
    y_bs5 = model_bs5.predict(df_plot)
    mse_bs5 = mean_squared_error(y, model_bs5.predict(df))
    ax.plot(X_plot, y_bs5, color='purple', linewidth=2.5, label=f'B-spline df=5 (MSE={mse_bs5:.4f})', zorder=6)

    # 6. B-spline df=10
    model_bs10 = smf.ols('fev ~ bs(height, df=10, degree=3)', data=df).fit()
    y_bs10 = model_bs10.predict(df_plot)
    mse_bs10 = mean_squared_error(y, model_bs10.predict(df))
    ax.plot(X_plot, y_bs10, color='purple', linestyle='--', linewidth=2.5, label=f'B-spline df=10 (MSE={mse_bs10:.4f})', zorder=7)

    ax.set_xlabel('Height', fontsize=14, fontweight='bold')
    ax.set_ylabel('FEV', fontsize=14, fontweight='bold')
    ax.set_title('All Models Comparison: FEV vs Height (Statsmodels)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fev_all_models_combined_sm.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nCombined plot saved as 'fev_all_models_combined_sm.png'")


def main():
    """Main analysis pipeline"""
    print("="*70)
    print("FEV SPLINE REGRESSION ANALYSIS (using statsmodels)")
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

    # Plot 5: Smoothing cubic spline with cross-validated alpha
    print("Fitting smoothing spline (penalized B-spline) with cross-validation...")
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
