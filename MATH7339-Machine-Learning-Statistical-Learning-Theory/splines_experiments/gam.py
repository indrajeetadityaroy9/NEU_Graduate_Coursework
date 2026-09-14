import numpy as np
from splines import PenalizedSpline


class GAM:
    def __init__(self, smooth_features,
                 linear_features=None,
                 n_knots=10,
                 lambda_=1.0,
                 degree=3,
                 max_iter=100,
                 tol=1e-4):

        self.smooth_features = smooth_features
        self.linear_features = linear_features if linear_features is not None else []
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol

        if isinstance(n_knots, int):
            self.n_knots = [n_knots] * len(smooth_features)
        else:
            self.n_knots = n_knots

        if isinstance(lambda_, (int, float)):
            self.lambda_ = [float(lambda_)] * len(smooth_features)
        else:
            self.lambda_ = lambda_

        self.smooth_models_ = {}
        self.linear_coef_ = None
        self.intercept_ = 0.0
        self.fitted_ = False

    def fit(self, X, y, verbose=False):
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        n, p = X.shape

        self.intercept_ = np.mean(y)
        fitted = np.full(n, self.intercept_)

        if len(self.linear_features) > 0:
            self.linear_coef_ = np.zeros(len(self.linear_features))

        for idx, feat_idx in enumerate(self.smooth_features):
            self.smooth_models_[feat_idx] = PenalizedSpline(
                n_knots=self.n_knots[idx],
                degree=self.degree,
                lambda_=self.lambda_[idx],
                diff_order=2
            )

        for iteration in range(self.max_iter):
            fitted_old = fitted.copy()

            for idx, feat_idx in enumerate(self.smooth_features):
                partial_residual = y - fitted + self._get_smooth_contribution(X, feat_idx)

                x_feat = X[:, feat_idx]
                self.smooth_models_[feat_idx].fit(x_feat, partial_residual)

                fitted = fitted - self._get_smooth_contribution(X, feat_idx)
                smooth_pred = self.smooth_models_[feat_idx].predict(x_feat)
                smooth_pred = smooth_pred - np.mean(smooth_pred)
                fitted = fitted + smooth_pred

            if len(self.linear_features) > 0:
                partial_residual = y - fitted
                for i, linear_idx in enumerate(self.linear_features):
                    partial_residual += self.linear_coef_[i] * X[:, linear_idx]

                X_linear = X[:, self.linear_features]
                self.linear_coef_ = np.linalg.lstsq(X_linear, partial_residual, rcond=None)[0]

                fitted = fitted - X_linear @ self.linear_coef_
                fitted = fitted + X_linear @ self.linear_coef_

            self.intercept_ = np.mean(y - fitted + self.intercept_)
            fitted = fitted - self.intercept_ + self.intercept_

            change = np.mean((fitted - fitted_old) ** 2)
            if verbose:
                print(f"Iteration {iteration + 1}: Change = {change:.6f}")

            if change < self.tol:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

        self.fitted_ = True
        return self

    def _get_smooth_contribution(self, X, feat_idx):
        if feat_idx not in self.smooth_models_:
            return np.zeros(X.shape[0])

        model = self.smooth_models_[feat_idx]
        if model.x_train is None:
            return np.zeros(X.shape[0])

        x_feat = X[:, feat_idx]
        pred = model.predict(x_feat)
        pred = pred - np.mean(pred)
        return pred

    def predict(self, X):
        if not self.fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.asarray(X)
        n = X.shape[0]

        y_pred = np.full(n, self.intercept_)

        for feat_idx in self.smooth_features:
            x_feat = X[:, feat_idx]
            smooth_pred = self.smooth_models_[feat_idx].predict(x_feat)
            smooth_pred = smooth_pred - np.mean(smooth_pred)
            y_pred += smooth_pred

        if len(self.linear_features) > 0:
            X_linear = X[:, self.linear_features]
            y_pred += X_linear @ self.linear_coef_

        return y_pred

    def get_smooth_function(self, feat_idx, n_points=100):
        if feat_idx not in self.smooth_models_:
            raise ValueError(f"Feature {feat_idx} is not a smooth term")

        model = self.smooth_models_[feat_idx]
        if model.x_train is None:
            raise ValueError("Model not fitted")

        x_min, x_max = model.x_train.min(), model.x_train.max()
        x_vals = np.linspace(x_min, x_max, n_points)
        y_vals = model.predict(x_vals)
        y_vals = y_vals - np.mean(y_vals)

        return x_vals, y_vals

    def summary(self):
        if not self.fitted_:
            raise ValueError("Model not fitted")

        summary = {
            'intercept': self.intercept_,
            'smooth_terms': {},
            'linear_terms': {}
        }

        for idx, feat_idx in enumerate(self.smooth_features):
            model = self.smooth_models_[feat_idx]
            summary['smooth_terms'][feat_idx] = {
                'n_knots': self.n_knots[idx],
                'lambda': self.lambda_[idx],
                'edf': model.effective_df()
            }

        if len(self.linear_features) > 0:
            for i, feat_idx in enumerate(self.linear_features):
                summary['linear_terms'][feat_idx] = {
                    'coefficient': self.linear_coef_[i],
                    'edf': 1.0
                }

        total_edf = 1.0
        total_edf += sum(info['edf'] for info in summary['smooth_terms'].values())
        total_edf += len(self.linear_features)
        summary['total_edf'] = total_edf

        return summary


class LogisticGAM:

    def __init__(self, smooth_features,
                 linear_features=None,
                 n_knots=10,
                 lambda_=1.0,
                 degree=3,
                 max_iter=25,
                 tol=1e-4):

        self.smooth_features = smooth_features
        self.linear_features = linear_features if linear_features is not None else []
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol

        if isinstance(n_knots, int):
            self.n_knots = [n_knots] * len(smooth_features)
        else:
            self.n_knots = n_knots

        if isinstance(lambda_, (int, float)):
            self.lambda_ = [float(lambda_)] * len(smooth_features)
        else:
            self.lambda_ = lambda_

        self.smooth_models_ = {}
        self.linear_coef_ = None
        self.intercept_ = 0.0
        self.fitted_ = False

    def _logistic(self, eta):
        eta = np.clip(eta, -500, 500)
        return 1.0 / (1.0 + np.exp(-eta))

    def _compute_deviance(self, y, p):
        eps = 1e-10
        p = np.clip(p, eps, 1 - eps)

        deviance = -2 * np.sum(
            y * np.log(p) + (1 - y) * np.log(1 - p)
        )
        return deviance

    def fit(self, X, y, verbose=False):
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        n, p = X.shape

        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must be binary (0/1)")

        p_init = np.mean(y)
        p_init = np.clip(p_init, 0.01, 0.99)
        self.intercept_ = np.log(p_init / (1 - p_init))

        if len(self.linear_features) > 0:
            self.linear_coef_ = np.zeros(len(self.linear_features))

        for idx, feat_idx in enumerate(self.smooth_features):
            self.smooth_models_[feat_idx] = PenalizedSpline(
                n_knots=self.n_knots[idx],
                degree=self.degree,
                lambda_=self.lambda_[idx],
                diff_order=2
            )

        eta = np.full(n, self.intercept_)
        deviance_old = np.inf

        for irls_iter in range(self.max_iter):
            p = self._logistic(eta)

            w = p * (1 - p)
            w = np.maximum(w, 1e-6)

            z = eta + (y - p) / w

            eta = np.full(n, self.intercept_)

            for _ in range(10):
                eta_old = eta.copy()

                for idx, feat_idx in enumerate(self.smooth_features):
                    partial_residual = z - eta + self._get_smooth_contribution(X, feat_idx)

                    x_feat = X[:, feat_idx]
                    self._fit_weighted_pspline(feat_idx, x_feat, partial_residual, w)

                    eta = eta - self._get_smooth_contribution(X, feat_idx)
                    smooth_pred = self.smooth_models_[feat_idx].predict(x_feat)
                    smooth_pred = smooth_pred - np.mean(smooth_pred)
                    eta = eta + smooth_pred

                if len(self.linear_features) > 0:
                    partial_residual = z - eta
                    for i, linear_idx in enumerate(self.linear_features):
                        partial_residual += self.linear_coef_[i] * X[:, linear_idx]

                    X_linear = X[:, self.linear_features]
                    W_sqrt = np.sqrt(w)
                    X_weighted = X_linear * W_sqrt[:, np.newaxis]
                    z_weighted = partial_residual * W_sqrt

                    self.linear_coef_ = np.linalg.lstsq(X_weighted, z_weighted, rcond=None)[0]

                    eta = eta - X_linear @ self.linear_coef_
                    eta = eta + X_linear @ self.linear_coef_

                self.intercept_ = np.average(z - eta, weights=w)
                eta = eta - self.intercept_
                eta = eta + self.intercept_

                if np.mean((eta - eta_old) ** 2) < self.tol * 0.1:
                    break

            p = self._logistic(eta)
            deviance = self._compute_deviance(y, p)

            if verbose:
                print(f"IRLS Iteration {irls_iter + 1}: Deviance = {deviance:.4f}")

            if abs(deviance - deviance_old) < self.tol:
                if verbose:
                    print(f"Converged after {irls_iter + 1} iterations")
                break

            deviance_old = deviance

        self.fitted_ = True
        return self

    def _fit_weighted_pspline(self, feat_idx, x,
                              z, w):
        model = self.smooth_models_[feat_idx]
        model.fit(x, z)

    def _get_smooth_contribution(self, X, feat_idx):
        if feat_idx not in self.smooth_models_:
            return np.zeros(X.shape[0])

        model = self.smooth_models_[feat_idx]
        if model.x_train is None:
            return np.zeros(X.shape[0])

        x_feat = X[:, feat_idx]
        pred = model.predict(x_feat)
        pred = pred - np.mean(pred)
        return pred

    def predict_proba(self, X):
        if not self.fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        eta = self.predict_linear(X)
        return self._logistic(eta)

    def predict_linear(self, X):
        if not self.fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.asarray(X)
        n = X.shape[0]

        eta = np.full(n, self.intercept_)

        for feat_idx in self.smooth_features:
            x_feat = X[:, feat_idx]
            smooth_pred = self.smooth_models_[feat_idx].predict(x_feat)
            smooth_pred = smooth_pred - np.mean(smooth_pred)
            eta += smooth_pred

        if len(self.linear_features) > 0:
            X_linear = X[:, self.linear_features]
            eta += X_linear @ self.linear_coef_

        return eta

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_smooth_function(self, feat_idx, n_points=100):
        if feat_idx not in self.smooth_models_:
            raise ValueError(f"Feature {feat_idx} is not a smooth term")

        model = self.smooth_models_[feat_idx]
        if model.x_train is None:
            raise ValueError("Model not fitted")

        x_min, x_max = model.x_train.min(), model.x_train.max()
        x_vals = np.linspace(x_min, x_max, n_points)
        y_vals = model.predict(x_vals)
        y_vals = y_vals - np.mean(y_vals)

        return x_vals, y_vals

    def summary(self):
        if not self.fitted_:
            raise ValueError("Model not fitted")

        summary = {
            'intercept': self.intercept_,
            'smooth_terms': {},
            'linear_terms': {}
        }

        for idx, feat_idx in enumerate(self.smooth_features):
            model = self.smooth_models_[feat_idx]
            summary['smooth_terms'][feat_idx] = {
                'n_knots': self.n_knots[idx],
                'lambda': self.lambda_[idx],
                'edf': model.effective_df()
            }

        if len(self.linear_features) > 0:
            for i, feat_idx in enumerate(self.linear_features):
                summary['linear_terms'][feat_idx] = {
                    'coefficient': self.linear_coef_[i],
                    'edf': 1.0
                }

        total_edf = 1.0
        total_edf += sum(info['edf'] for info in summary['smooth_terms'].values())
        total_edf += len(self.linear_features)
        summary['total_edf'] = total_edf

        return summary
