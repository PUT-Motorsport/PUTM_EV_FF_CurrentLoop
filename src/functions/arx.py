import numpy as np
from sklearn.linear_model import Ridge, QuantileRegressor


class ARXModel:
    """
    ARX (AutoRegressive with eXogenous inputs) via Ridge or quantile regression.

    I(t+1) = a1·I(t) + ... + ap·I(t-p+1)
            + b1·T(t) + ... + bq·T(t-q+1)
            + c·U(t) + bias

    quantile=None  → Ridge (mean prediction, multi-output native)
    quantile=0.90  → QuantileRegressor (conservative, 4 × single-output internally)
    """

    def __init__(self, alpha: float = 1.0, quantile: float | None = None):
        self.alpha = alpha
        self.quantile = quantile
        self._models: list | None = None
        if quantile is None:
            self._ridge = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ARXModel":
        if self.quantile is not None:
            y2d = y if y.ndim == 2 else y[:, None]
            self._models = []
            for j in range(y2d.shape[1]):
                m = QuantileRegressor(quantile=self.quantile, alpha=0, solver="highs")
                m.fit(X, y2d[:, j])
                self._models.append(m)
        else:
            self._ridge.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.quantile is not None:
            preds = [m.predict(X) for m in self._models]
            return np.column_stack(preds) if len(preds) > 1 else preds[0]
        return self._ridge.predict(X)

    @property
    def coef_(self) -> np.ndarray:
        if self.quantile is not None:
            return np.array([m.coef_ for m in self._models])
        return self._ridge.coef_

    @property
    def intercept_(self):
        if self.quantile is not None:
            return np.array([m.intercept_ for m in self._models])
        return float(self._ridge.intercept_)
