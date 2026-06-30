# src/functions/ekf_nonlinear.py
# Extended Kalman Filter for DC current prediction — Formula Student Power Limiter.
#
# State  x = [I_dc [A],  dI/dt [A/s],  P_mech [W]]
#   I_dc   — DC bus current
#   dI/dt  — rate of change of current
#   P_mech — estimated mechanical power tracked as internal state
#
# Nonlinear transition  f(x, P_input, U_dc):
#   I_dc(t+1)   = I_dc(t) + dt * dI/dt(t)
#   dI/dt(t+1)  = (1 - γ·dt)·dI/dt(t) + γ·dt·P_mech(t)/U_dc(t) - γ·dt·I_dc(t)
#   P_mech(t+1) = (1 - α)·P_mech(t) + α·P_input(t)
#
# Jacobian  F_j = ∂f/∂x  evaluated at (x̂, U_dc(t)):
#   [ 1,       dt,            0              ]
#   [ -γ·dt,   1 − γ·dt,     γ·dt / U_dc(t) ]   ← time-varying → true EKF
#   [ 0,       0,             1 − α          ]
#
# Observation (linear):  z = H·x = I_dc,   H = [1, 0, 0]

import numpy as np


class ExtendedKalmanFilter:
    """
    EKF with state [I_dc, dI/dt, P_mech].

    The term γ·dt/U_dc(t) in F_jacobian varies each tick because U_dc changes
    with battery state-of-charge — this is what makes it a genuine EKF rather
    than a time-invariant linear KF.
    """

    def __init__(self, dt, gamma, alpha, Q, R, x0=None, P0=None):
        """
        dt    : VCU time step [s]
        gamma : current error bandwidth [1/s]  (≈ 1 / τ_motor)
        alpha : mechanical power IIR update rate ∈ (0, 1]  (= dt / τ_mech)
        Q     : (3, 3) process noise covariance
        R     : measurement noise variance [A²]  (scalar or (1,1))
        x0    : (3,) initial state  [I0, 0, P_mech0] — default zeros
        P0    : (3, 3) initial covariance — default 1e4·I
        """
        self.dt    = float(dt)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.Q     = np.asarray(Q, dtype=float)
        self.R     = float(np.asarray(R).ravel()[0])
        self.H     = np.array([[1.0, 0.0, 0.0]])

        n = 3
        self.x = np.zeros(n)     if x0 is None else np.asarray(x0, dtype=float).copy()
        self.P = np.eye(n) * 1e4 if P0 is None else np.asarray(P0, dtype=float).copy()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _f(self, x, P_input, U_dc):
        I, dI, Pm = x
        dt, g, a = self.dt, self.gamma, self.alpha
        I_next  = I + dt * dI
        dI_next = (1.0 - g * dt) * dI + g * dt * (Pm / U_dc - I)
        Pm_next = (1.0 - a) * Pm + a * P_input
        return np.array([I_next, dI_next, Pm_next])

    def _F_jacobian(self, U_dc):
        dt, g, a = self.dt, self.gamma, self.alpha
        return np.array([
            [1.0,      dt,            0.0           ],
            [-g * dt,  1.0 - g * dt,  g * dt / U_dc ],
            [0.0,      0.0,           1.0 - a       ],
        ])

    # ------------------------------------------------------------------
    # EKF steps
    # ------------------------------------------------------------------

    def predict(self, P_input, U_dc):
        """
        EKF prediction step.

        P_input : mechanical power proxy at current tick [W]  (k_p · Σ|τ_i|·|ω_i|)
        U_dc    : battery voltage at current tick [V]  — enters F_jacobian
        Returns (x_pred, P_pred), modifies internal state.
        """
        F_j    = self._F_jacobian(U_dc)
        self.x = self._f(self.x, P_input, U_dc)
        self.P = F_j @ self.P @ F_j.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, z):
        """
        Linear update step (H = [1, 0, 0] — we measure I_dc directly).

        z : DC current measurement [A]
        Returns (x_est, P_est), modifies internal state.
        """
        z    = float(z)
        y    = z - float(self.H @ self.x)
        S    = float(self.H @ self.P @ self.H.T) + self.R
        K    = (self.P @ self.H.T / S).ravel()                    # (3,)
        self.x = self.x + K * y
        I_KH   = np.eye(3) - np.outer(K, self.H.ravel())
        self.P = I_KH @ self.P @ I_KH.T + self.R * np.outer(K, K)  # Joseph form
        return self.x.copy(), self.P.copy()

    def predict_k_ahead(self, k, P_input_seq, U_dc_seq):
        """
        k-step-ahead prediction WITHOUT modifying internal state.

        P_input_seq : (k,) mechanical power inputs
        U_dc_seq    : (k,) battery voltages [V]
        Returns predicted state x̂(t+k|t).
        """
        x = self.x.copy()
        for i in range(k):
            x = self._f(x, float(P_input_seq[i]), float(U_dc_seq[i]))
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
        """Per-state standard deviation: sqrt of P diagonal."""
        return np.sqrt(np.maximum(np.diag(self.P), 0.0))


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def make_power_ekf(dt, gamma, alpha, q_I, q_dI, q_Pm, r_noise,
                   I0=0.0, P_mech0=0.0):
    """
    Build an ExtendedKalmanFilter for [I_dc, dI/dt, P_mech].

    Parameters
    ----------
    dt      : VCU time step [s]
    gamma   : current error bandwidth [1/s]  (≈ 1/τ_motor)
    alpha   : P_mech IIR update rate (= dt / τ_mech)
    q_I     : process noise std for I_dc  [A]
    q_dI    : process noise std for dI/dt [A/s]
    q_Pm    : process noise std for P_mech [W]
    r_noise : measurement noise std [A]
    I0      : initial I_dc estimate [A]
    P_mech0 : initial P_mech estimate [W]
    """
    Q  = np.diag([q_I ** 2, q_dI ** 2, q_Pm ** 2])
    R  = r_noise ** 2
    x0 = np.array([I0, 0.0, P_mech0])
    P0 = np.diag([r_noise ** 2, (q_dI * dt) ** 2, q_Pm ** 2]) * 100.0
    return ExtendedKalmanFilter(dt=dt, gamma=gamma, alpha=alpha,
                                Q=Q, R=R, x0=x0, P0=P0)
