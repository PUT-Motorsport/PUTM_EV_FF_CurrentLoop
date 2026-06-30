import numpy as np
from sklearn.linear_model import Ridge


class ARMAXModel:
    """
    ARMAX (ARX + Moving-Average residuals) via two-pass Ridge regression.

    I(t+1) = ARX(t) + c1·ε(t) + ... + cr·ε(t-r+1) + ε(t+1)

    Two-pass fit:
      1. Fit ARX, compute training residuals ε̂
      2. Append lagged ε̂ as features, refit

    Online inference: maintain a rolling residual buffer updated each tick.
    Residuals are clipped to ±3σ of training residuals to prevent runaway.
    """

    def __init__(self, n_ma: int = 5, alpha: float = 1.0):
        self.n_ma = n_ma
        self.alpha = alpha
        self._model: Ridge | None = None
        self.res_clip_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ARMAXModel":
        n = len(y)

        # Pass 1: ARX residuals
        arx_pass1 = Ridge(alpha=self.alpha)
        arx_pass1.fit(X, y)
        res_train = y - arx_pass1.predict(X)
        self.res_clip_ = 3.0 * float(np.std(res_train))

        # Lagged residuals — pure numpy, no pandas needed
        res_lags = np.zeros((n, self.n_ma))
        for i, lag in enumerate(range(1, self.n_ma + 1)):
            if lag < n:
                res_lags[lag:, i] = res_train[:-lag]

        X_armax = np.hstack([X, res_lags])

        # Pass 2: ARMAX
        self._model = Ridge(alpha=self.alpha)
        self._model.fit(X_armax, y)
        return self

    def predict(self, X: np.ndarray, y_true: np.ndarray | None = None) -> np.ndarray:
        """
        Online inference with a rolling residual buffer.

        y_true : ground-truth targets for residual feedback.
                 Pass None to run open-loop (residuals stay zero).
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")

        y_pred = np.zeros(len(X))
        res_buf = np.zeros(self.n_ma)

        for i in range(len(X)):
            x_i = np.concatenate([X[i], res_buf])
            y_hat = float(self._model.predict(x_i.reshape(1, -1))[0])
            y_pred[i] = y_hat
            res_buf = np.roll(res_buf, 1)
            if y_true is not None:
                res_buf[0] = np.clip(y_true[i] - y_hat, -self.res_clip_, self.res_clip_)
        return y_pred

    @property
    def coef_(self) -> np.ndarray:
        return self._model.coef_

    @property
    def n_params(self) -> int:
        return int(self._model.coef_.shape[0])
