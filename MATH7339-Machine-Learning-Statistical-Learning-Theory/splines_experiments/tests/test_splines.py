import numpy as np
import pytest
import sys
sys.path.append('../src')

from splines import (
    truncated_power,
    truncated_power_basis_matrix,
    RegressionSpline,
    NaturalCubicSpline,
    SmoothingSpline
)


class TestTruncatedPower:

    def test_positive_part(self):
        x = np.array([-1, 0, 1, 2, 3])
        t = 1.0

        result = truncated_power(x, t, 1)
        expected = np.array([0, 0, 0, 1, 2])

        np.testing.assert_array_almost_equal(result, expected)

    def test_degree_zero(self):
        x = np.array([0, 0.5, 1.0, 1.5, 2.0])
        t = 1.0

        result = truncated_power(x, t, 0)
        expected = np.array([0, 0, 0, 1, 1])

        np.testing.assert_array_almost_equal(result, expected)

    def test_degree_three_cubic(self):
        x = np.array([0, 1, 2, 3, 4])
        t = 2.0

        result = truncated_power(x, t, 3)
        expected = np.array([0, 0, 0, 1, 8])

        np.testing.assert_array_almost_equal(result, expected)

    def test_continuity_at_knot(self):
        t = 1.0
        k = 3
        eps = 1e-6

        x_left = np.array([t - eps])
        x_right = np.array([t + eps])

        y_left = truncated_power(x_left, t, k)
        y_right = truncated_power(x_right, t, k)

        assert np.abs(y_left - y_right) < 1e-5


class TestBasisMatrix:

    def test_shape(self):
        x = np.linspace(0, 10, 50)
        knots = np.array([3, 5, 7])
        degree = 3

        G = truncated_power_basis_matrix(x, knots, degree)

        n = len(x)
        m = len(knots)
        expected_cols = m + degree + 1

        assert G.shape == (n, expected_cols)

    def test_polynomial_terms(self):
        x = np.array([0, 1, 2, 3])
        knots = np.array([5])
        degree = 2

        G = truncated_power_basis_matrix(x, knots, degree)

        np.testing.assert_array_almost_equal(G[:, 0], np.ones(len(x)))

        np.testing.assert_array_almost_equal(G[:, 1], x)

        np.testing.assert_array_almost_equal(G[:, 2], x**2)

    def test_truncated_terms(self):
        x = np.array([0, 1, 2, 3, 4])
        knots = np.array([2.0])
        degree = 1

        G = truncated_power_basis_matrix(x, knots, degree)

        expected_truncated = np.array([0, 0, 0, 1, 2])
        np.testing.assert_array_almost_equal(G[:, -1], expected_truncated)


class TestRegressionSpline:

    def test_perfect_fit_interpolation(self):
        x = np.linspace(0, 10, 20)
        y = 2 + 3*x - 0.5*x**2 + 0.01*x**3

        knots = np.array([])
        model = RegressionSpline(degree=3)
        model.fit(x, y, knots)

        y_pred = model.predict(x)

        np.testing.assert_array_almost_equal(y_pred, y, decimal=10)

    def test_interpolation_at_training_points_with_many_knots(self):
        np.random.seed(42)
        x = np.linspace(0, 10, 10)
        y = np.sin(x) + np.random.normal(0, 0.01, len(x))

        knots = x[1:-1]
        model = RegressionSpline(degree=3)
        model.fit(x, y, knots)

        y_pred = model.predict(x)

        mse = np.mean((y - y_pred)**2)
        assert mse < 0.01

    def test_predict_shape(self):
        x_train = np.linspace(0, 10, 20)
        y_train = np.sin(x_train)
        knots = np.array([3, 5, 7])

        model = RegressionSpline(degree=3)
        model.fit(x_train, y_train, knots)

        x_test = np.linspace(0, 10, 100)
        y_pred = model.predict(x_test)

        assert y_pred.shape == (len(x_test),)

    def test_linear_regression_special_case(self):
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y_true = 2 + 3*x
        y = y_true + np.random.normal(0, 0.5, len(x))

        model = RegressionSpline(degree=1)
        model.fit(x, y, np.array([]))

        assert np.abs(model.coefficients[0] - 2) < 0.5
        assert np.abs(model.coefficients[1] - 3) < 0.2


class TestNaturalCubicSpline:

    def test_linear_beyond_boundaries(self):
        np.random.seed(42)
        x_train = np.linspace(2, 8, 20)
        y_train = np.sin(x_train)

        knots = np.concatenate([[2, 8], np.linspace(3, 7, 4)])

        model = NaturalCubicSpline()
        model.fit(x_train, y_train, knots)

        x_left = np.array([0, 1, 2])
        y_left = model.predict(x_left)

        second_diff = np.diff(y_left, n=2)
        assert np.all(np.abs(second_diff) < 0.1)

    def test_fit_and_predict(self):
        x = np.linspace(0, 10, 30)
        y = np.sin(x)
        knots = np.linspace(0, 10, 6)

        model = NaturalCubicSpline()
        model.fit(x, y, knots)

        y_pred = model.predict(x)

        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
        assert r2 > 0.9


class TestSmoothingSpline:

    def test_lambda_zero_interpolates(self):
        np.random.seed(42)
        x = np.linspace(0, 10, 15)
        y = np.sin(x)

        model = SmoothingSpline(lambda_=1e-6)
        model.fit(x, y)
        y_pred = model.predict(x)

        mse = np.mean((y - y_pred)**2)
        assert mse < 0.01

    def test_large_lambda_smooths(self):
        np.random.seed(42)
        x = np.linspace(0, 10, 20)
        y = np.sin(x) + np.random.normal(0, 0.5, len(x))

        model_smooth = SmoothingSpline(lambda_=100.0)
        model_smooth.fit(x, y)
        y_smooth = model_smooth.predict(x)

        model_wiggly = SmoothingSpline(lambda_=0.001)
        model_wiggly.fit(x, y)
        y_wiggly = model_wiggly.predict(x)

        smooth_curv = np.sum(np.abs(np.diff(y_smooth, n=2)))
        wiggly_curv = np.sum(np.abs(np.diff(y_wiggly, n=2)))

        assert smooth_curv < wiggly_curv

    def test_cross_validation(self):
        np.random.seed(42)
        x = np.linspace(0, 10, 40)
        y = np.sin(x) + np.random.normal(0, 0.2, len(x))

        lambdas = np.array([0.01, 0.1, 1.0, 10.0])

        model = SmoothingSpline()
        best_lambda, cv_errors = model.cross_validate(x, y, lambdas, cv_folds=5)

        assert best_lambda in lambdas

        assert len(cv_errors) == len(lambdas)

        assert np.all(cv_errors >= 0)

    def test_automatic_knot_placement(self):
        x = np.array([0, 1, 2, 3, 4, 5])
        y = np.array([1, 2, 1.5, 3, 2.5, 4])

        model = SmoothingSpline(lambda_=0.1)
        model.fit(x, y)

        assert len(model.x_train) == len(x)
        np.testing.assert_array_equal(np.sort(model.x_train), np.sort(x))


class TestEdgeCases:

    def test_single_knot(self):
        x = np.linspace(0, 10, 20)
        y = np.sin(x)
        knots = np.array([5.0])

        model = RegressionSpline(degree=3)
        model.fit(x, y, knots)
        y_pred = model.predict(x)

        assert y_pred.shape == y.shape

    def test_very_noisy_data(self):
        np.random.seed(42)
        x = np.linspace(0, 10, 30)
        y = np.random.normal(0, 10, len(x))

        model1 = RegressionSpline(degree=3)
        model1.fit(x, y, np.array([3, 5, 7]))
        _ = model1.predict(x)

        model2 = NaturalCubicSpline()
        model2.fit(x, y, np.linspace(0, 10, 5))
        _ = model2.predict(x)

    def test_constant_data(self):
        x = np.linspace(0, 10, 20)
        y = np.ones(len(x)) * 5.0

        model = RegressionSpline(degree=3)
        model.fit(x, y, np.array([3, 7]))
        y_pred = model.predict(x)

        np.testing.assert_array_almost_equal(y_pred, y, decimal=5)


class TestMathematicalProperties:

    def test_linear_smoother_property(self):
        x = np.linspace(0, 10, 20)
        y1 = np.sin(x)
        y2 = np.cos(x)
        alpha = 0.3
        y_combined = alpha * y1 + (1 - alpha) * y2

        knots = np.array([3, 5, 7])

        model1 = RegressionSpline(degree=3)
        model1.fit(x, y1, knots)
        pred1 = model1.predict(x)

        model2 = RegressionSpline(degree=3)
        model2.fit(x, y2, knots)
        pred2 = model2.predict(x)

        model_combined = RegressionSpline(degree=3)
        model_combined.fit(x, y_combined, knots)
        pred_combined = model_combined.predict(x)

        expected = alpha * pred1 + (1 - alpha) * pred2
        np.testing.assert_array_almost_equal(pred_combined, expected, decimal=10)

    def test_spline_smoothness(self):
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        knots = np.array([3, 5, 7])

        model = RegressionSpline(degree=3)
        model.fit(x, y, knots)

        x_fine = np.linspace(0, 10, 1000)
        y_pred = model.predict(x_fine)

        h = x_fine[1] - x_fine[0]
        second_deriv = np.diff(y_pred, n=2) / h**2

        third_deriv = np.diff(second_deriv)

        assert np.std(third_deriv) < 100

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
