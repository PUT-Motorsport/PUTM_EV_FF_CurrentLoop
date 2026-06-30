import numpy as np
from sklearn.linear_model import Ridge


class EchoStateNetwork:
    """
    Echo State Network (Reservoir Computing) for time-series regression.

    Architecture:
      u(t) → W_in → reservoir x(t) → W_out → ŷ(t)

    State update (leaky integrator):
      x(t+1) = (1-α)·x(t) + α·tanh(W_in·u(t+1) + W_res·x(t))

    Only W_out is trained (Ridge). W_in and W_res are fixed random.
    Inference cost: one sparse matrix-vector multiply → comparable to ARX,
    straightforward C port (static arrays, no dynamic allocation).
    """

    def __init__(self, n_inputs: int, n_reservoir: int = 300,
                 spectral_radius: float = 0.95, sparsity: float = 0.9,
                 leaking_rate: float = 0.3, input_scaling: float = 1.0,
                 ridge_alpha: float = 1e-4, seed: int = 42):
        self.n_inputs      = n_inputs
        self.N             = n_reservoir
        self.rho           = spectral_radius
        self.sparsity      = sparsity
        self.alpha         = leaking_rate
        self.input_scaling = input_scaling
        self.ridge_alpha   = ridge_alpha

        rng = np.random.default_rng(seed)

        # Input weights: (N, n_inputs), uniform [-input_scaling, +input_scaling]
        self.W_in = rng.uniform(-1.0, 1.0, (self.N, n_inputs)) * input_scaling

        # Reservoir: sparse random, rescaled to desired spectral radius
        W = rng.uniform(-1.0, 1.0, (self.N, self.N))
        W[rng.random((self.N, self.N)) < sparsity] = 0.0
        sr = np.max(np.abs(np.linalg.eigvals(W)))
        if sr > 0:
            W *= spectral_radius / sr
        self.W_res = W

        self._ridge: Ridge | None = None
        self.x_ = np.zeros(self.N)   # current reservoir state (updated by step/fit)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        pre = self.W_in @ u + self.W_res @ x
        return (1.0 - self.alpha) * x + self.alpha * np.tanh(pre)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, U: np.ndarray, y: np.ndarray,
            washout: int = 100) -> "EchoStateNetwork":
        """
        U       : (n, n_inputs) normalized input sequence
        y       : (n,) target values
        washout : initial steps discarded while reservoir settles from x=0
        """
        n = len(U)
        states = np.zeros((n, self.N))
        x = np.zeros(self.N)
        for t in range(n):
            x = self._step(x, U[t])
            states[t] = x

        self._ridge = Ridge(alpha=self.ridge_alpha)
        self._ridge.fit(states[washout:], y[washout:])
        self.x_ = x.copy()   # save final state for warm-start predict
        return self

    def predict(self, U: np.ndarray, warm_start: bool = True) -> np.ndarray:
        """
        Batch prediction over U.

        warm_start : True  → continue from reservoir state after fit() — correct
                             for sequential test data immediately following training.
                     False → reset reservoir to zeros (independent evaluation).
        """
        if self._ridge is None:
            raise RuntimeError("Call fit() before predict().")
        x = self.x_.copy() if warm_start else np.zeros(self.N)
        states = np.zeros((len(U), self.N))
        for t in range(len(U)):
            x = self._step(x, U[t])
            states[t] = x
        return self._ridge.predict(states)

    def step(self, u: np.ndarray) -> float:
        """
        Single-step online inference — updates internal state, returns scalar.
        Mimics VCU tick-by-tick execution.
        """
        if self._ridge is None:
            raise RuntimeError("Call fit() before step().")
        self.x_ = self._step(self.x_, u)
        return float(self._ridge.predict(self.x_.reshape(1, -1))[0])

    def reset_state(self, x0: np.ndarray | None = None) -> None:
        """Reset reservoir state to zeros or a provided vector."""
        self.x_ = np.zeros(self.N) if x0 is None else np.asarray(x0, dtype=float).copy()

    @property
    def n_params(self) -> int:
        """Trainable parameters: readout weights + bias."""
        return self.N + 1
