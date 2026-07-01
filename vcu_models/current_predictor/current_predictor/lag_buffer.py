import collections
import numpy as np


class LagBuffer:
    """
    Shift-register maintaining the last N_LAGS+1 samples of each input signal.

    Feature vector order (31 elements, must match training):
      I_FL, I_FL_lag1..5,
      I_FR, I_FR_lag1..5,
      I_RL, I_RL_lag1..5,
      I_RR, I_RR_lag1..5,
      T_sum, T_sum_lag1..5,
      U_dc
    """

    MOTOR_NAMES = ('FL', 'FR', 'RL', 'RR')
    N_FEATURES  = 31  # 4*6 + 6 + 1

    def __init__(self, n_lags: int = 5) -> None:
        self.n_lags = n_lags
        maxlen = n_lags + 1
        self._I: dict[str, collections.deque] = {
            m: collections.deque([0.0] * maxlen, maxlen=maxlen)
            for m in self.MOTOR_NAMES
        }
        self._T_sum: collections.deque = collections.deque([0.0] * maxlen, maxlen=maxlen)
        self._u_dc: float = 0.0
        self._tick: int = 0

    def update(self, currents: np.ndarray, t_sum: float, u_dc: float) -> None:
        """Push one new observation.

        currents: array-like [I_FL, I_FR, I_RL, I_RR]
        t_sum:    T_FL + T_FR + T_RL + T_RR [Nm]
        u_dc:     DC bus voltage [V]
        """
        for i, m in enumerate(self.MOTOR_NAMES):
            self._I[m].appendleft(float(currents[i]))
        self._T_sum.appendleft(float(t_sum))
        self._u_dc = float(u_dc)
        self._tick += 1

    @property
    def ready(self) -> bool:
        """True once n_lags+1 real samples have been pushed."""
        return self._tick >= self.n_lags + 1

    def get_feature_vector(self) -> np.ndarray:
        """Return (31,) feature vector ready for model inference."""
        feats: list[float] = []
        for m in self.MOTOR_NAMES:
            buf = list(self._I[m])   # [t, t-1, t-2, t-3, t-4, t-5]
            feats.append(buf[0])     # I_m at t
            feats.extend(buf[1:])    # I_m_lag1 .. I_m_lag5
        T_buf = list(self._T_sum)
        feats.append(T_buf[0])       # T_sum at t
        feats.extend(T_buf[1:])      # T_sum_lag1 .. T_sum_lag5
        feats.append(self._u_dc)     # U_dc
        return np.array(feats, dtype=np.float64)

    def reset(self) -> None:
        """Clear all history (call at the start of each new run)."""
        maxlen = self.n_lags + 1
        for m in self.MOTOR_NAMES:
            self._I[m] = collections.deque([0.0] * maxlen, maxlen=maxlen)
        self._T_sum = collections.deque([0.0] * maxlen, maxlen=maxlen)
        self._u_dc = 0.0
        self._tick = 0
