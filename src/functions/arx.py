import numpy as np
from sklearn.linear_model import Ridge


class ARXModel:
    """
    ARX (AutoRegressive with eXogenous inputs) via Ridge regression.

    I(t+1) = a1·I(t) + ... + ap·I(t-p+1)
            + b1·T(t) + ... + bq·T(t-q+1)
            + c·U(t) + d·dI/dt(t) + ε

    Closed-form fit — trivial C port (single dot product + scalar bias).
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._ridge = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ARXModel":
        self._ridge.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._ridge.predict(X)

    @property
    def coef_(self) -> np.ndarray:
        return self._ridge.coef_

    @property
    def intercept_(self) -> float:
        return float(self._ridge.intercept_)
