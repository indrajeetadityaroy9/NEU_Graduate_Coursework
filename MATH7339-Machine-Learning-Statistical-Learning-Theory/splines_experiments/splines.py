import numpy as np
from scipy.integrate import quad
from scipy.linalg import solve
from scipy.interpolate import BSpline as ScipyBSpline


def truncated_power(x: np.ndarray, knot: float, degree: int) -> np.ndarray:
    return np.maximum(x - knot, 0) ** degree


def bspline_basis(x: np.ndarray, knots: np.ndarray, degree: int = 3,
                  derivatives: int = 0) -> np.ndarray:
    x = np.asarray(x).ravel()
    knots = np.asarray(knots).ravel()

    x_min, x_max = np.min(x), np.max(x)

    full_knots = np.concatenate([
        np.repeat(x_min, degree + 1),
        knots,
        np.repeat(x_max, degree + 1)
    ])

    n_basis = len(knots) + degree + 1
    n = len(x)

    B = np.zeros((n, n_basis))

    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1.0

        try:
            bspl = ScipyBSpline(full_knots, c, degree)
            B[:, i] = bspl(x, nu=derivatives)
        except:
            B[:, i] = 0.0

    return B


def truncated_power_basis_matrix(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    x = np.asarray(x).ravel()
    knots = np.asarray(knots).ravel()
    n = len(x)
    m = len(knots)
    n_basis = m + degree + 1

    G = np.zeros((n, n_basis))

    for j in range(degree + 1):
        G[:, j] = x ** j

    for j, knot in enumerate(knots):
        G[:, degree + 1 + j] = truncated_power(x, knot, degree)

    return G


class RegressionSpline:

    def __init__(self, degree: int = 3):
        self.degree = degree
        self.knots = None
        self.coefficients = None
        self.x_train = None
        self.y_train = None

    def fit(self, x: np.ndarray, y: np.ndarray, knots: np.ndarray) -> 'RegressionSpline':
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        self.knots = np.asarray(knots).ravel()

        self.x_train = x
        self.y_train = y

        G = truncated_power_basis_matrix(x, self.knots, self.degree)

        self.coefficients, _, _, _ = np.linalg.lstsq(G, y, rcond=None)

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise ValueError("Model not fitted. Call fit() first.")

        x = np.asarray(x).ravel()
        G = truncated_power_basis_matrix(x, self.knots, self.degree)
        return G @ self.coefficients

    def get_basis_matrix(self, x: np.ndarray) -> np.ndarray:
        if self.knots is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return truncated_power_basis_matrix(x, self.knots, self.degree)


def natural_cubic_spline_basis_matrix(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    x = np.asarray(x).ravel()
    knots = np.asarray(knots).ravel()
    knots = np.sort(knots)

    n = len(x)
    m = len(knots)

    if m < 2:
        raise ValueError("Need at least 2 knots for natural cubic splines")

    x_min, x_max = knots[0], knots[-1]

    N = np.zeros((n, m))

    def d_k(x_val, knot_idx):
        if knot_idx < 0 or knot_idx >= m:
            return np.zeros_like(x_val)

        t_k = knots[knot_idx]
        numerator = (truncated_power(x_val, t_k, 3) -
                     truncated_power(x_val, x_max, 3))
        denominator = x_max - t_k

        if denominator == 0:
            return np.zeros_like(x_val)

        return numerator / denominator

    N[:, 0] = 1.0
    N[:, 1] = x

    if m > 2:
        for j in range(2, m):
            N[:, j] = d_k(x, j-2) - d_k(x, m-2)

    return N


class NaturalCubicSpline:

    def __init__(self):
        self.knots = None
        self.coefficients = None
        self.x_train = None
        self.y_train = None

    def fit(self, x: np.ndarray, y: np.ndarray, knots: np.ndarray) -> 'NaturalCubicSpline':
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        self.knots = np.asarray(knots).ravel()

        self.x_train = x
        self.y_train = y

        N = natural_cubic_spline_basis_matrix(x, self.knots)

        self.coefficients, _, _, _ = np.linalg.lstsq(N, y, rcond=None)

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise ValueError("Model not fitted. Call fit() first.")

        x = np.asarray(x).ravel()
        N = natural_cubic_spline_basis_matrix(x, self.knots)
        return N @ self.coefficients


class SmoothingSpline:

    def __init__(self, lambda_: float = 1.0):
        self.lambda_ = lambda_
        self.x_train = None
        self.y_train = None
        self.coefficients = None
        self.basis_matrix = None
        self.penalty_matrix = None

    def _compute_penalty_matrix(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        x = np.sort(x)

        Omega = np.zeros((n, n))

        def second_derivative_basis(t, x_data, basis_idx):
            h = 1e-5
            N_plus = natural_cubic_spline_basis_matrix(np.array([t + h]), x_data)[0, basis_idx]
            N = natural_cubic_spline_basis_matrix(np.array([t]), x_data)[0, basis_idx]
            N_minus = natural_cubic_spline_basis_matrix(np.array([t - h]), x_data)[0, basis_idx]
            return (N_plus - 2*N + N_minus) / (h**2)

        x_min, x_max = x[0], x[-1]

        for i in range(n):
            for j in range(i, n):
                integrand = lambda t: (second_derivative_basis(t, x, i) *
                                      second_derivative_basis(t, x, j))
                result, _ = quad(integrand, x_min, x_max, limit=50, epsabs=1e-6, epsrel=1e-6)
                Omega[i, j] = result
                Omega[j, i] = result

        return Omega

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'SmoothingSpline':
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()

        self.x_train = x
        self.y_train = y

        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]

        knots = x_sorted

        self.basis_matrix = natural_cubic_spline_basis_matrix(x_sorted, knots)

        self.penalty_matrix = self._compute_penalty_matrix(x_sorted)

        G = self.basis_matrix
        Omega = self.penalty_matrix

        A = G.T @ G + self.lambda_ * Omega
        b = G.T @ y_sorted

        self.coefficients = solve(A, b, assume_a='pos')

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise ValueError("Model not fitted. Call fit() first.")

        x = np.asarray(x).ravel()

        N = natural_cubic_spline_basis_matrix(x, self.x_train)

        return N @ self.coefficients

    def cross_validate(self, x: np.ndarray, y: np.ndarray,
                      lambdas: np.ndarray, cv_folds: int = 5):
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        n = len(x)

        cv_errors = []

        for lam in lambdas:
            fold_errors = []
            fold_size = n // cv_folds

            for fold in range(cv_folds):
                test_idx = np.arange(fold * fold_size, (fold + 1) * fold_size)
                train_idx = np.setdiff1d(np.arange(n), test_idx)

                x_train_fold = x[train_idx]
                y_train_fold = y[train_idx]
                x_test_fold = x[test_idx]
                y_test_fold = y[test_idx]

                model = SmoothingSpline(lambda_=lam)
                model.fit(x_train_fold, y_train_fold)
                y_pred = model.predict(x_test_fold)

                mse = np.mean((y_test_fold - y_pred) ** 2)
                fold_errors.append(mse)

            cv_errors.append(np.mean(fold_errors))

        cv_errors = np.array(cv_errors)
        best_idx = np.argmin(cv_errors)
        best_lambda = lambdas[best_idx]

        return best_lambda, cv_errors


class PenalizedSpline:

    def __init__(self, n_knots: int = 20, degree: int = 3, lambda_: float = 1.0,
                 diff_order: int = 2):
        self.n_knots = n_knots
        self.degree = degree
        self.lambda_ = lambda_
        self.diff_order = diff_order
        self.knots = None
        self.coefficients = None
        self.x_train = None
        self.y_train = None

    def _difference_penalty_matrix(self, n_basis: int) -> np.ndarray:
        D = np.diff(np.eye(n_basis), n=self.diff_order, axis=0)
        P = D.T @ D
        return P

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'PenalizedSpline':
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()

        self.x_train = x
        self.y_train = y

        x_min, x_max = np.min(x), np.max(x)
        self.knots = np.linspace(x_min, x_max, self.n_knots + 2)[1:-1]

        B = bspline_basis(x, self.knots, self.degree)

        n_basis = B.shape[1]
        P = self._difference_penalty_matrix(n_basis)

        A = B.T @ B + self.lambda_ * P
        b = B.T @ y

        self.coefficients = solve(A, b, assume_a='pos')

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise ValueError("Model not fitted. Call fit() first.")

        x = np.asarray(x).ravel()
        B = bspline_basis(x, self.knots, self.degree)
        return B @ self.coefficients

    def effective_df(self) -> float:
        if self.x_train is None:
            raise ValueError("Model not fitted. Call fit() first.")

        B = bspline_basis(self.x_train, self.knots, self.degree)
        n_basis = B.shape[1]
        P = self._difference_penalty_matrix(n_basis)

        A = B.T @ B + self.lambda_ * P
        H = B @ solve(A, B.T, assume_a='pos')

        return np.trace(H)

    def gcv_score(self) -> float:
        if self.y_train is None:
            raise ValueError("Model not fitted. Call fit() first.")

        y_pred = self.predict(self.x_train)
        rss = np.sum((self.y_train - y_pred) ** 2)
        df = self.effective_df()
        n = len(self.y_train)

        gcv = (n * rss) / (n - df) ** 2
        return gcv

    def cross_validate(self, x: np.ndarray, y: np.ndarray,
                      lambdas: np.ndarray, cv_folds: int = 5):
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        n = len(x)

        cv_errors = []

        for lam in lambdas:
            fold_errors = []
            fold_size = n // cv_folds

            for fold in range(cv_folds):
                test_idx = np.arange(fold * fold_size, (fold + 1) * fold_size)
                train_idx = np.setdiff1d(np.arange(n), test_idx)

                x_train_fold = x[train_idx]
                y_train_fold = y[train_idx]
                x_test_fold = x[test_idx]
                y_test_fold = y[test_idx]

                model = PenalizedSpline(n_knots=self.n_knots, degree=self.degree,
                                       lambda_=lam, diff_order=self.diff_order)
                model.fit(x_train_fold, y_train_fold)
                y_pred = model.predict(x_test_fold)

                mse = np.mean((y_test_fold - y_pred) ** 2)
                fold_errors.append(mse)

            cv_errors.append(np.mean(fold_errors))

        cv_errors = np.array(cv_errors)
        best_idx = np.argmin(cv_errors)
        best_lambda = lambdas[best_idx]

        return best_lambda, cv_errors
