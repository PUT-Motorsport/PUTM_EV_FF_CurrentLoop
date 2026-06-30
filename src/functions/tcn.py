import numpy as np


class MinimalTCN:
    """Pure-NumPy TCN: 2 dilated causal conv blocks + linear output.

    Architecture:
      Input (B, n_ch, seq_len)
        └─ CausalConv1d(n_ch→n_filters, k=kernel_size, d=1) + ReLU
        └─ CausalConv1d(n_filters→n_filters, k=kernel_size, d=2) + ReLU
        └─ Take last timestep → (B, n_filters)
        └─ Linear(n_filters→1)

    Causal dilated conv at dilation d, kernel size k:
      out[t] = Σ_{j=0}^{k-1} W[:,j] · x[t - d·j]   (only past, never future)

    Implemented with im2col → single matrix multiply per layer.
    Trained with mini-batch Adam on MSE loss.
    """

    def __init__(self, n_ch: int = 3, n_filters: int = 16,
                 kernel_size: int = 3, lr: float = 5e-4, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.ks = kernel_size

        def _he(shape):
            return rng.normal(0, np.sqrt(2.0 / shape[1]), shape)

        self.W1 = _he((n_filters, n_ch * kernel_size))
        self.b1 = np.zeros(n_filters)
        self.W2 = _he((n_filters, n_filters * kernel_size))
        self.b2 = np.zeros(n_filters)
        self.Wo = rng.normal(0, 0.01, (1, n_filters))
        self.bo = np.zeros(1)
        self.lr = lr

        self._pnames = ['W1', 'b1', 'W2', 'b2', 'Wo', 'bo']
        self._m = {k: np.zeros_like(getattr(self, k)) for k in self._pnames}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in self._pnames}
        self._t = 0

    # ------------------------------------------------------------------
    # im2col / col2im — causal dilated conv as matrix multiply
    # ------------------------------------------------------------------

    def _im2col(self, x: np.ndarray, dilation: int) -> np.ndarray:
        """(B, C, T) -> (B, C*ks, T)  causal, zero-pads past boundaries."""
        B, C, T = x.shape
        out = np.zeros((B, C * self.ks, T))
        for k in range(self.ks):
            shift = dilation * k
            src = x[:, :, :T - shift] if shift > 0 else x
            out[:, k * C:(k + 1) * C, shift:] = src
        return out

    def _col2im_grad(self, dx_col: np.ndarray, dilation: int,
                     C: int, T: int) -> np.ndarray:
        """(B, C*ks, T) -> (B, C, T)  accumulate gradients."""
        B = dx_col.shape[0]
        dx = np.zeros((B, C, T))
        for k in range(self.ks):
            shift = dilation * k
            if shift == 0:
                dx += dx_col[:, k * C:(k + 1) * C, :]
            else:
                dx[:, :, :T - shift] += dx_col[:, k * C:(k + 1) * C, shift:]
        return dx

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _conv_fwd(self, x, W, b, dilation):
        xcol = self._im2col(x, dilation)
        out = np.tensordot(W, xcol, axes=([1], [1])).transpose(1, 0, 2)
        out += b[np.newaxis, :, np.newaxis]
        return out, xcol

    def _conv_bwd(self, xcol, W, d_out, dilation, C_in, T):
        dW = np.tensordot(d_out, xcol, axes=([0, 2], [0, 2]))
        db = d_out.sum(axis=(0, 2))
        dx_col = np.tensordot(W, d_out, axes=([0], [1])).transpose(1, 0, 2)
        dx = self._col2im_grad(dx_col, dilation, C_in, T)
        return dW, db, dx

    def _forward(self, x):
        z1, x1col = self._conv_fwd(x,  self.W1, self.b1, dilation=1)
        h1 = np.maximum(0, z1)
        z2, x2col = self._conv_fwd(h1, self.W2, self.b2, dilation=2)
        h2 = np.maximum(0, z2)
        hl = h2[:, :, -1]                                   # (B, F)
        y = (hl @ self.Wo.T + self.bo).squeeze(-1)          # (B,)
        return y, (x, x1col, z1, h1, x2col, z2, h2, hl)

    def _backward(self, cache, y_pred, y_true):
        x, x1col, z1, h1, x2col, z2, h2, hl = cache
        B, _, T = x.shape

        dy  = 2.0 * (y_pred - y_true) / B
        dWo = np.einsum('b,bf->f', dy, hl).reshape(1, -1)
        dbo = dy.sum(keepdims=True)
        dhl = np.outer(dy, self.Wo[0])

        dh2 = np.zeros_like(h2)
        dh2[:, :, -1] = dhl
        dz2 = dh2 * (z2 > 0)

        dW2, db2, dh1 = self._conv_bwd(x2col, self.W2, dz2, 2, h1.shape[1], T)
        dz1 = dh1 * (z1 > 0)
        dW1, db1, _   = self._conv_bwd(x1col, self.W1, dz1, 1, x.shape[1],  T)

        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'Wo': dWo, 'bo': dbo}

    def _adam_step(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        for k in self._pnames:
            g          = grads[k]
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * g
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * g ** 2
            mh = self._m[k] / (1 - beta1 ** self._t)
            vh = self._v[k] / (1 - beta2 ** self._t)
            getattr(self, k)[:] -= self.lr * mh / (np.sqrt(vh) + eps)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 50, batch_size: int = 512) -> list[float]:
        """
        X : (n, n_ch, seq_len)
        y : (n,) normalized targets
        Returns per-epoch RMSE history.
        """
        n = len(X)
        history = []
        for ep in range(1, epochs + 1):
            idx = np.random.permutation(n)
            losses = []
            for s in range(0, n, batch_size):
                bi = idx[s:s + batch_size]
                y_hat, cache = self._forward(X[bi])
                losses.append(np.mean((y_hat - y[bi]) ** 2))
                self._adam_step(self._backward(cache, y_hat, y[bi]))
            rmse = np.sqrt(np.mean(losses))
            history.append(rmse)
            if ep % 10 == 0:
                print(f'  Epoch {ep:3d}/{epochs}  train RMSE (norm) = {rmse:.4f}')
        return history

    def predict(self, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        """X : (n, n_ch, seq_len) -> y_pred : (n,)"""
        parts = []
        for s in range(0, len(X), batch_size):
            y_hat, _ = self._forward(X[s:s + batch_size])
            parts.append(y_hat)
        return np.concatenate(parts)
