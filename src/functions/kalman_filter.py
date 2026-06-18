# src/functions/kalman_filter.py
# Linear Kalman Filter for motor current prediction — Formula Student TV / Power Limiter
#
# Current implementation: 1 DC-bus current (fsp_endu_current).
# Extensible to 4 per-motor currents via block-diagonal structure (n_motors=4).

import numpy as np


class LinearKalmanFilter:
    """
    Discrete Linear Kalman Filter.

    State model:  x(t+1) = F·x(t) + B·u(t) + w,    w ~ N(0, Q)
    Observation:  z(t)   = H·x(t) + v,               v ~ N(0, R)

    For n_motors currents with Constant Velocity model:
        state per motor = [I, dI/dt]  →  total state dim = 2 * n_motors
    """

    def __init__(self, F, H, Q, R, B=None, x0=None, P0=None):
        """
        F  : (n, n)  state transition matrix
        H  : (m, n)  observation matrix
        Q  : (n, n)  process noise covariance
        R  : (m, m)  measurement noise covariance
        B  : (n, k)  control input matrix (optional — for setpoint feed-forward)
        x0 : (n,)    initial state  (default: zeros)
        P0 : (n, n)  initial covariance (default: large diagonal)
        """
        self.F = np.asarray(F, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.B = np.asarray(B, dtype=float) if B is not None else None

        n = self.F.shape[0]
        self.x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float)
        self.P = np.eye(n) * 1e4 if P0 is None else np.asarray(P0, dtype=float)

    # ------------------------------------------------------------------
    # Core filter steps
    # ------------------------------------------------------------------

    def predict(self, u=None, F_override=None, Q_override=None):
        """
        Prediction step — propagate state one timestep forward.
        Returns (x_pred, P_pred) and modifies internal state.

        Pass F_override / Q_override when dt varies between steps.
        """
        F = self.F if F_override is None else np.asarray(F_override, dtype=float)
        Q = self.Q if Q_override is None else np.asarray(Q_override, dtype=float)

        self.x = F @ self.x
        if self.B is not None and u is not None:
            self.x = self.x + self.B @ np.asarray(u, dtype=float)
        self.P = F @ self.P @ F.T + Q
        return self.x.copy(), self.P.copy()

    def update(self, z):
        """
        Update step — incorporate a new measurement z.
        Returns (x_est, P_est) and modifies internal state.
        """
        z = np.asarray(z, dtype=float).ravel()
        y = z - self.H @ self.x                      # innovation
        S = self.H @ self.P @ self.H.T + self.R      # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain
        self.x = self.x + K @ y
        I_KH = np.eye(len(self.x)) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T  # Joseph form (numerically stable)
        return self.x.copy(), self.P.copy()

    # ------------------------------------------------------------------
    # Look-ahead prediction (does NOT modify internal state)
    # ------------------------------------------------------------------

    def predict_k_ahead(self, k, u_sequence=None, F_override=None):
        """
        Return k-step-ahead state prediction WITHOUT advancing the filter.

        u_sequence : list/array of k control inputs shape (k, ku) — optional
        F_override : use instead of self.F (e.g. a pre-computed F for fixed dt)
        """
        F = self.F if F_override is None else np.asarray(F_override, dtype=float)
        x = self.x.copy()
        for i in range(k):
            x = F @ x
            if self.B is not None and u_sequence is not None and i < len(u_sequence):
                x = x + self.B @ np.asarray(u_sequence[i], dtype=float)
        return x

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self):
        return self.x.copy()

    @property
    def covariance(self):
        return self.P.copy()

    def estimate_std(self):
        """Standard deviation of each state element (sqrt of P diagonal)."""
        return np.sqrt(np.maximum(np.diag(self.P), 0.0))

    def current_indices(self, n_motors=None):
        """
        Indices in x that correspond to currents (not derivatives).
        For CV model: [0, 2, 4, ...] — every other element.
        """
        if n_motors is None:
            n_motors = self.F.shape[0] // 2
        return list(range(0, 2 * n_motors, 2))


# ------------------------------------------------------------------
# Factory helpers
# ------------------------------------------------------------------

def make_cv_matrices(dt, q_accel, n_motors=1):
    """
    Return (F, Q, H) for the Constant-Velocity (CV) model.

    State per motor: [I, dI/dt]
    Total state:     2 * n_motors elements

    Uses the Continuous White Noise Acceleration (CWNA) discretization for Q.

    Parameters
    ----------
    dt       : time step [s]
    q_accel  : acceleration noise standard deviation [A/s²]
    n_motors : number of motors
    """
    F1 = np.array([[1.0, dt],
                   [0.0, 1.0]])
    Q1 = (q_accel ** 2) * np.array([[dt**3 / 3.0, dt**2 / 2.0],
                                     [dt**2 / 2.0, dt]])
    H1 = np.array([[1.0, 0.0]])

    if n_motors == 1:
        return F1, Q1, H1

    F = np.kron(np.eye(n_motors), F1)
    Q = np.kron(np.eye(n_motors), Q1)
    H = np.kron(np.eye(n_motors), H1)   # shape: (n_motors, 2*n_motors)
    return F, Q, H


def make_constant_velocity_kf(dt, r_noise, q_accel, n_motors=1,
                               B=None, x0=None, P0=None):
    """
    Build a LinearKalmanFilter using the CV model for n_motors currents.

    Parameters
    ----------
    dt       : sampling interval [s]  (VCU loop period = 0.01 s at 100 Hz)
    r_noise  : measurement noise std [A] — scalar or (n_motors,) array
    q_accel  : process noise (acceleration std) [A/s²]
    n_motors : 1 (now) → 4 (when per-motor CAN data is available)
    B        : (2*n_motors, k) control matrix (optional, for setpoint feed-forward)
    x0       : initial state vector (2*n_motors,) — default zeros
    P0       : initial covariance (2*n_motors, 2*n_motors) — default large diagonal
    """
    F, Q, H = make_cv_matrices(dt, q_accel, n_motors)

    r = np.atleast_1d(np.asarray(r_noise, dtype=float))
    if r.size == 1 and n_motors > 1:
        r = np.repeat(r, n_motors)
    R = np.diag(r ** 2)

    return LinearKalmanFilter(F=F, H=H, Q=Q, R=R, B=B, x0=x0, P0=P0)
