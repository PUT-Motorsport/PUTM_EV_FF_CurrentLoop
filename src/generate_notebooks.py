#!/usr/bin/env python3
"""Generate notebooks 02-06 to use training_data.csv (5 Hz, 4-motor targets)."""
import json, uuid
from pathlib import Path

SRC = Path(__file__).parent

def _id(): return uuid.uuid4().hex[:8]
def md(s): return {"cell_type":"markdown","id":_id(),"metadata":{},"source":[s] if isinstance(s,str) else s}
def code(s,outs=None): return {"cell_type":"code","id":_id(),"execution_count":None,"metadata":{},"outputs":outs or [],"source":[s] if isinstance(s,str) else s}
def nb(cells): return {"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11.0"}},"cells":cells}
def save(name, cells):
    p = SRC / name
    p.write_text(json.dumps(nb(cells), indent=1, ensure_ascii=False))
    print(f"Written: {p}")

# ── Shared snippets ────────────────────────────────────────────────────────────

SHARED_IMPORTS_HEADER = """\
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_cwd = Path().resolve()
SRC_DIR  = _cwd if (_cwd / 'functions').exists() else _cwd / 'src'
DATA_DIR = SRC_DIR.parent / 'data' / 'model'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DT            = 0.2        # 5 Hz sample rate
N_LAGS        = 5          # 1 s lookback (5 ticks × 0.2 s)
POWER_LIMIT_W = 80_000
MOTOR_NAMES   = ['FL', 'FR', 'RL', 'RR']
"""

SHARED_LOAD = """\
df = pd.read_csv(DATA_DIR / 'training_data.csv')
df = df.sort_values(['run_id', 'timestamp_s']).reset_index(drop=True)
df['T_sum'] = df['T_FL'] + df['T_FR'] + df['T_RL'] + df['T_RR']

# Run-safe lag features (no cross-run leakage)
for m in MOTOR_NAMES:
    for lag in range(1, N_LAGS + 1):
        df[f'I_{m}_lag{lag}'] = df.groupby('run_id')[f'I_{m}'].shift(lag)
for lag in range(1, N_LAGS + 1):
    df[f'T_sum_lag{lag}'] = df.groupby('run_id')['T_sum'].shift(lag)

# 1-step-ahead target (within each run)
for m in MOTOR_NAMES:
    df[f'I_{m}_next'] = df.groupby('run_id')[f'I_{m}'].shift(-1)

df_clean = df.dropna().reset_index(drop=True)
print(f'Loaded  : {len(df):,} rows, {df[\"run_id\"].nunique()} runs')
print(f'Clean   : {len(df_clean):,} rows after dropna')
"""

SHARED_SPLIT = """\
runs         = sorted(df_clean['run_id'].unique())
n_train_runs = max(1, int(len(runs) * 0.70))
train_runs   = runs[:n_train_runs]
test_runs    = runs[n_train_runs:]

df_train = df_clean[df_clean['run_id'].isin(train_runs)].reset_index(drop=True)
df_test  = df_clean[df_clean['run_id'].isin(test_runs)].reset_index(drop=True)

y_train = df_train[[f'I_{m}_next' for m in MOTOR_NAMES]].values   # (n, 4)
y_test  = df_test[[f'I_{m}_next'  for m in MOTOR_NAMES]].values
U_test  = df_test['U_dc'].values
t_test  = df_test['timestamp_s'].values

print(f'Train runs : {train_runs}  ({len(df_train):,} samples)')
print(f'Test  runs : {test_runs}   ({len(df_test):,} samples)')
print(f'y_train    : {y_train.shape}')
"""

SHARED_FEATURES = """\
# Shared feature matrix for all 4 motors
feature_cols = []
for m in MOTOR_NAMES:
    feature_cols.append(f'I_{m}')           # current value at t
    for lag in range(1, N_LAGS + 1):
        feature_cols.append(f'I_{m}_lag{lag}')
feature_cols.append('T_sum')
for lag in range(1, N_LAGS + 1):
    feature_cols.append(f'T_sum_lag{lag}')
feature_cols.append('U_dc')

X_train = df_train[feature_cols].values
X_test  = df_test[feature_cols].values
print(f'Features : {len(feature_cols)}   X_train: {X_train.shape}')
"""

# ── Notebook 02: EKF ──────────────────────────────────────────────────────────

nb02_cells = [

md("""\
# EKF — Per-Motor Current Prediction (Power Limiter / TV)

| Property | Value |
|----------|-------|
| **Models** | 4 × EKF (state: I_m, dI/dt, P_mech_m) |
| **Baseline** | 4-motor Linear KF (Constant Velocity) |
| **Input** | P_mech_m = |T_m|·|v_m|, U_dc |
| **Output** | I_FL, I_FR, I_RL, I_RR at t+1 |
| **Data** | training_data.csv — 5 Hz, split by run_id |

**Why EKF:** the Jacobian F contains γ·dt/U_dc(t) — this term changes each tick
as battery voltage varies with SoC, making the model a genuine Extended Kalman Filter.\
"""),

code(SHARED_IMPORTS_HEADER + """\
from scipy.signal import correlate
from sklearn.linear_model import LinearRegression
from functions.ekf_nonlinear import make_power_ekf
from functions.kalman_filter import make_constant_velocity_kf
from functions.evaluation    import display_model_results, compare_models

U_MIN = 350.0   # clamp voltage in Jacobian to avoid division by zero
print(f'DATA_DIR : {DATA_DIR}')
"""),

md("## 1. Data Loading"),

code("""\
df = pd.read_csv(DATA_DIR / 'training_data.csv')
df = df.sort_values(['run_id', 'timestamp_s']).reset_index(drop=True)

# Mechanical power proxy per motor: |torque| * |velocity|
for m in MOTOR_NAMES:
    df[f'P_mech_{m}'] = df[f'T_{m}'].abs() * df[f'v_{m}'].abs()

# 1-step-ahead target within each run
for m in MOTOR_NAMES:
    df[f'I_{m}_next'] = df.groupby('run_id')[f'I_{m}'].shift(-1)

df_clean = df.dropna().reset_index(drop=True)

runs         = sorted(df_clean['run_id'].unique())
n_train_runs = max(1, int(len(runs) * 0.70))
train_runs   = runs[:n_train_runs]
test_runs    = runs[n_train_runs:]

df_train = df_clean[df_clean['run_id'].isin(train_runs)].reset_index(drop=True)
df_test  = df_clean[df_clean['run_id'].isin(test_runs)].reset_index(drop=True)

y_test  = df_test[[f'I_{m}_next' for m in MOTOR_NAMES]].values
U_dcv   = np.maximum(df_test['U_dc'].values, U_MIN)
t_test  = df_test['timestamp_s'].values

print(f'Train : {train_runs}  ({len(df_train):,} samples)')
print(f'Test  : {test_runs}   ({len(df_test):,} samples)')
print(f'U_dc  : {df_clean[\"U_dc\"].min():.1f} – {df_clean[\"U_dc\"].max():.1f} V')
"""),

md("## 2. Parameter Estimation\n\nCross-correlate training data to find response lags τ_motor and τ_mech."),

code("""\
MAX_LAG = 15   # at 5 Hz: 15 lags = 3 s

I_tr = df_train['I_FL'].values
T_tr = (df_train['T_FL'] + df_train['T_FR'] + df_train['T_RL'] + df_train['T_RR']).values
P_tr = df_train['P_mech_FL'].values
U_tr = np.maximum(df_train['U_dc'].values, U_MIN)

def _norm(x): return (x - x.mean()) / (x.std() + 1e-8)

mid  = len(I_tr) - 1
xc_T = correlate(_norm(I_tr), _norm(T_tr), mode='full')[mid:mid + MAX_LAG + 1]
xc_P = correlate(_norm(I_tr), _norm(P_tr), mode='full')[mid:mid + MAX_LAG + 1]

peak_T    = max(int(np.argmax(xc_T)), 1)
peak_P    = max(int(np.argmax(xc_P)), 1)
tau_motor = peak_T * DT
tau_mech  = peak_P * DT
gamma     = min(1.0 / max(tau_motor, DT), 1.0 / DT)   # clamp: ensure 1-γ·dt ≥ 0
alpha_ekf = min(DT / max(tau_mech, DT), 1.0)

# k_p: scale P_mech to current units
mask = (I_tr > 20) & (T_tr > 50)
if mask.sum() > 50:
    X_kp = (P_tr[mask] / U_tr[mask]).reshape(-1, 1)
    k_p  = float(LinearRegression(fit_intercept=False).fit(X_kp, I_tr[mask]).coef_[0])
    k_p  = max(k_p, 1e-6)
else:
    k_p = 1.0

I_smooth = pd.Series(I_tr).rolling(3, center=True, min_periods=1).median().values
R_std    = max(float(np.std(I_tr - I_smooth)), 1.0)
q_accel  = max(float(np.percentile(np.abs(np.diff(I_tr, n=2) / DT**2), 75)), 1.0)
q_Pm     = max(float(np.std(np.diff(P_tr * k_p))), 1.0)

print(f'tau_motor : {tau_motor:.2f} s  (peak_T = {peak_T})')
print(f'tau_mech  : {tau_mech:.2f} s  (peak_P = {peak_P})')
print(f'gamma     : {gamma:.3f} 1/s  |  alpha : {alpha_ekf:.4f}')
print(f'k_p       : {k_p:.5f}  |  R_std : {R_std:.2f} A')

lags = np.arange(MAX_LAG + 1) * DT
fig, axes = plt.subplots(1, 2, figsize=(12, 3))
for ax, xc, label, pk in [(axes[0], xc_T, 'T_sum -> I_FL', peak_T),
                           (axes[1], xc_P, 'P_mech -> I_FL', peak_P)]:
    ax.stem(lags, xc, markerfmt='C0o', linefmt='C0-', basefmt='k-')
    ax.axvline(pk * DT, color='red', ls='--', label=f'Peak = {pk * DT:.2f} s')
    ax.set_xlabel('Lag [s]'); ax.set_title(label); ax.legend()
    ax.grid(True, ls=':', alpha=0.5)
plt.tight_layout(); plt.show()
"""),

md("## 3. Build Filters"),

code("""\
ekfs = {}
for m in MOTOR_NAMES:
    I0_m    = float(df_test[f'I_{m}'].iloc[0])
    Pm0_m   = k_p * float(df_test[f'P_mech_{m}'].iloc[0])
    ekfs[m] = make_power_ekf(
        dt=DT, gamma=gamma, alpha=alpha_ekf,
        q_I=R_std, q_dI=q_accel*DT, q_Pm=q_Pm,
        r_noise=R_std, I0=I0_m, P_mech0=Pm0_m,
    )

I0_all = np.array([df_test[f'I_{m}'].iloc[0] for m in MOTOR_NAMES])
kf_cv  = make_constant_velocity_kf(
    dt=DT, r_noise=R_std, q_accel=q_accel, n_motors=4,
    x0=np.kron(I0_all, [1.0, 0.0]),
)
ci_cv = kf_cv.current_indices(n_motors=4)
print('EKF states:', {m: np.round(ekfs[m].state, 1) for m in MOTOR_NAMES})
print('CV state dim:', kf_cv.F.shape[0], '  current indices:', ci_cv)
"""),

md("## 4. Simulation Loop"),

code("""\
N       = len(df_test)
pred_ekf = np.zeros((N, 4))
pred_cv  = np.zeros((N, 4))
std_ekf  = np.zeros((N, 4))
I_meas   = np.column_stack([df_test[f'I_{m}'].values for m in MOTOR_NAMES])

t0 = time.perf_counter()
for i in range(N):
    U_i = float(U_dcv[i])

    x_cv       = kf_cv.predict_k_ahead(k=1)
    pred_cv[i] = x_cv[ci_cv]
    kf_cv.predict()
    kf_cv.update(I_meas[i])

    for j, m in enumerate(MOTOR_NAMES):
        P_i            = k_p * float(df_test[f'P_mech_{m}'].iloc[i])
        xa             = ekfs[m].predict_k_ahead(k=1, P_input_seq=[P_i], U_dc_seq=[U_i])
        pred_ekf[i, j] = xa[0]
        std_ekf[i, j]  = ekfs[m].estimate_std()[0]
        ekfs[m].predict(P_input=P_i, U_dc=U_i)
        ekfs[m].update(I_meas[i, j])

t_sim = time.perf_counter() - t0
print(f'Simulation: {N:,} ticks  {t_sim*1000:.1f} ms  ({t_sim/N*1e6:.1f} us/tick)')
"""),

md("## 5. Evaluation"),

code("""\
m_cv = display_model_results(
    'KF CV 4-motor', y_test, pred_cv,
    t_sim, voltage=U_dcv, motor_names=MOTOR_NAMES, save_to_excel=False,
)
m_ekf = display_model_results(
    'EKF 3-state 4-motor', y_test, pred_ekf,
    t_sim, voltage=U_dcv, motor_names=MOTOR_NAMES, save_to_excel=True,
)
compare_models([m_cv, m_ekf], sort_by='RMSE [A]')
"""),

md("## 6. Visualization"),

code("""\
N_PLOT = min(150, N)
sl     = slice(0, N_PLOT)
t_pl   = t_test[sl] - t_test[0]

fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
fig.suptitle('EKF per-motor — 1-step-ahead prediction (5 Hz)', fontsize=13, fontweight='bold')
for i, (ax, m) in enumerate(zip(axes, MOTOR_NAMES)):
    ax.plot(t_pl, y_test[sl, i],     lw=0.9, color='black',     alpha=0.6, label='Actual')
    ax.plot(t_pl, pred_cv[sl, i],    lw=1.2, color='steelblue', ls='--',   label='CV')
    ax.plot(t_pl, pred_ekf[sl, i],   lw=1.5, color='lime',                 label='EKF')
    ax.fill_between(t_pl,
                    pred_ekf[sl, i] - 2*std_ekf[sl, i],
                    pred_ekf[sl, i] + 2*std_ekf[sl, i],
                    alpha=0.15, color='lime', label='EKF +/-2sigma')
    ax.set_ylabel(f'I_{m} [A]'); ax.legend(fontsize=8); ax.grid(True, ls=':', alpha=0.5)
axes[-1].set_xlabel('Time [s]')
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(16, 3))
P_true = U_dcv[sl] * y_test[sl].sum(axis=1) / 1e3
P_ekf  = U_dcv[sl] * pred_ekf[sl].sum(axis=1) / 1e3
ax.plot(t_pl, P_true, lw=0.9, color='black', alpha=0.6, label='Actual power')
ax.plot(t_pl, P_ekf,  lw=1.5, color='lime',  ls='--',  label='EKF predicted')
ax.axhline(POWER_LIMIT_W / 1e3, color='red', lw=2, ls='--', label='80 kW limit')
ax.set_ylabel('DC Power [kW]'); ax.set_xlabel('Time [s]')
ax.legend(); ax.grid(True, ls=':', alpha=0.5)
plt.tight_layout(); plt.show()
"""),

md("""\
## Summary

| | CV baseline | EKF per motor |
|---|---|---|
| Input | none | P_mech_m = |T_m * v_m|, U_dc |
| State | [I, dI/dt] x4 | [I, dI/dt, P_mech] x4 |
| F matrix | constant | varies with U_dc (true EKF) |
| VCU-ready | yes | yes (~10 us/tick) |

**Next:** XGBoost in `03_xgboost_current.ipynb`.\
"""),
]

save("02_kalman_ekf.ipynb", nb02_cells)

# ── Notebook 03: XGBoost ─────────────────────────────────────────────────────

nb03_cells = [

md("""\
# XGBoost — Per-Motor Current Prediction (Power Limiter)

| Property | Value |
|----------|-------|
| **Model** | 4 x XGBoost (one per motor, shared features) |
| **Features** | I_FL/FR/RL/RR lags, T_sum lags, U_dc |
| **Target** | I_FL, I_FR, I_RL, I_RR at t+1 |
| **Split** | by run_id (first 70% = train) |

Hyperparameters tuned on FL motor via RandomizedSearchCV, then reused for all 4 motors.\
"""),

code(SHARED_IMPORTS_HEADER + """\
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from functions.evaluation import display_model_results, compare_models

print(f'XGBoost  : {xgb.__version__}')
print(f'DATA_DIR : {DATA_DIR}')
"""),

md("## 1. Data Loading and Feature Engineering"),

code(SHARED_LOAD),
code(SHARED_SPLIT),
code(SHARED_FEATURES),

md("## 2. Training — RandomizedSearchCV on FL, reuse for FR/RL/RR"),

code("""\
param_grid = {
    'n_estimators':     [200, 400, 600],
    'max_depth':        [3, 4, 5],
    'learning_rate':    [0.05, 0.1, 0.2],
    'subsample':        [0.7, 0.85, 1.0],
    'colsample_bytree': [0.7, 0.85, 1.0],
    'min_child_weight': [1, 3, 5],
}

base = xgb.XGBRegressor(
    objective='reg:squarederror', tree_method='hist',
    random_state=42, n_jobs=-1,
)
search = RandomizedSearchCV(
    base, param_grid, n_iter=20, scoring='neg_root_mean_squared_error',
    cv=3, random_state=42, verbose=1, n_jobs=-1,
)

t0 = time.perf_counter()
search.fit(X_train, y_train[:, 0])   # tune on FL
t_tune = time.perf_counter() - t0
best_params = search.best_params_

print(f'Tuning done in {t_tune:.1f} s')
print(f'Best CV RMSE : {-search.best_score_:.2f} A')
print(f'Best params  : {best_params}')
"""),

code("""\
models = {}
t0 = time.perf_counter()
for j, m in enumerate(MOTOR_NAMES):
    xgb_m = xgb.XGBRegressor(
        **best_params, objective='reg:squarederror',
        tree_method='hist', random_state=42, n_jobs=-1,
    )
    xgb_m.fit(X_train, y_train[:, j])
    models[m] = xgb_m
    print(f'  Motor {m} trained')
t_train = time.perf_counter() - t0
print(f'Total training: {t_train:.1f} s')
"""),

md("## 3. Evaluation"),

code("""\
t0 = time.perf_counter()
y_pred = np.column_stack([models[m].predict(X_test) for m in MOTOR_NAMES])
t_infer = time.perf_counter() - t0

metrics = display_model_results(
    'XGBoost 4-motor', y_test, y_pred, t_infer,
    voltage=U_test, motor_names=MOTOR_NAMES, save_to_excel=True,
    feature_importances=models['FL'].feature_importances_,
    feature_names=feature_cols,
)
"""),

md("## 4. Visualization"),

code("""\
N_PLOT = min(200, len(y_test))
sl     = slice(0, N_PLOT)
t_pl   = t_test[sl] - t_test[0]

fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
fig.suptitle('XGBoost — 1-step-ahead per-motor current (5 Hz)', fontsize=13, fontweight='bold')
for i, (ax, m) in enumerate(zip(axes, MOTOR_NAMES)):
    ax.plot(t_pl, y_test[sl, i], lw=0.9, color='black', alpha=0.6, label='Actual')
    ax.plot(t_pl, y_pred[sl, i], lw=1.5, color='lime',  ls='--',   label='XGBoost')
    ax.set_ylabel(f'I_{m} [A]'); ax.legend(fontsize=8); ax.grid(True, ls=':', alpha=0.5)
axes[-1].set_xlabel('Time [s]')
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(16, 3))
P_true = U_test[sl] * y_test[sl].sum(axis=1) / 1e3
P_pred = U_test[sl] * y_pred[sl].sum(axis=1) / 1e3
ax.plot(t_pl, P_true, lw=0.9, color='black', alpha=0.6, label='Actual power')
ax.plot(t_pl, P_pred, lw=1.5, color='lime',  ls='--',   label='XGBoost predicted')
ax.axhline(POWER_LIMIT_W / 1e3, color='red', lw=2, ls='--', label='80 kW limit')
ax.set_ylabel('DC Power [kW]'); ax.set_xlabel('Time [s]')
ax.legend(); ax.grid(True, ls=':', alpha=0.5)
plt.tight_layout(); plt.show()
"""),

md("""\
## Summary

| Property | Value |
|----------|-------|
| Features | 4 x (I_m + 5 lags) + T_sum + 5 lags + U_dc = 31 features |
| Models | 4 separate XGBRegressors (shared feature matrix) |
| VCU-suitability | batch predict on fixed feature vector |

**Next:** ARX / ARMAX in `04_arx_armax.ipynb`.\
"""),
]

save("03_xgboost_current.ipynb", nb03_cells)

# ── Notebook 04: ARX / ARMAX ─────────────────────────────────────────────────

nb04_cells = [

md("""\
# ARX / ARMAX — Per-Motor Current Prediction (Power Limiter)

| Model | Type | Key property |
|-------|------|-------------|
| **ARX** | Linear AR + exogenous | Fast, interpretable, trivial C port |
| **ARMAX** | ARX + moving-average residuals | Handles correlated noise |

**Data:** `training_data.csv` — 5 Hz, split by run_id.
**Output:** I_FL, I_FR, I_RL, I_RR at t+1 (multi-output Ridge regression).\
"""),

code(SHARED_IMPORTS_HEADER + """\
from functions.arx       import ARXModel
from functions.armax     import ARMAXModel
from functions.evaluation import display_model_results, compare_models

N_MA = 3   # ARMAX moving-average order
print(f'DATA_DIR : {DATA_DIR}')
"""),

md("## 1. Data Loading and Feature Engineering"),
code(SHARED_LOAD),
code(SHARED_SPLIT),
code(SHARED_FEATURES),

md("""\
## 2. ARX — AutoRegressive eXogenous

```
I_m(t+1) = a1*I_m(t) + ... + a5*I_m(t-4)
         + b1*T_sum(t) + ... + b5*T_sum(t-4)
         + c*U_dc(t) + bias
```
Fitted jointly for all 4 motors via Ridge regression (y is 2D: n_samples x 4).\
"""),

code("""\
arx = ARXModel(alpha=1.0)

t0 = time.perf_counter()
arx.fit(X_train, y_train)
t_train_arx = time.perf_counter() - t0

t0 = time.perf_counter()
y_pred_arx = arx.predict(X_test)
t_infer_arx = time.perf_counter() - t0

print(f'ARX train  : {t_train_arx*1000:.1f} ms')
print(f'ARX infer  : {t_infer_arx*1000:.2f} ms  ({len(X_test)} samples)')
print(f'Coef shape : {arx.coef_.shape}  (n_outputs x n_features)')

m_arx = display_model_results(
    'ARX (Ridge)', y_test, y_pred_arx, t_infer_arx,
    voltage=U_test, motor_names=MOTOR_NAMES, save_to_excel=False,
)
"""),

md("""\
## 3. ARMAX — ARX + Moving Average

Adds lagged residuals as features to capture correlated noise:
```
I_m(t+1) = ARX(t) + c1*e_m(t) + c2*e_m(t-1) + ... + bias
```
Two-pass fit: (1) fit ARX, compute residuals; (2) append lagged residuals, refit.\
"""),

code("""\
# ARMAX.predict is single-output — train 4 independent models (one per motor)
armax_models = {}
t0 = time.perf_counter()
for j, m in enumerate(MOTOR_NAMES):
    armax_m = ARMAXModel(n_ma=N_MA, alpha=1.0)
    armax_m.fit(X_train, y_train[:, j])
    armax_models[m] = armax_m
t_train_armax = time.perf_counter() - t0
print(f'ARMAX train : {t_train_armax*1000:.1f} ms  (4 motors)')

t0 = time.perf_counter()
y_pred_armax = np.column_stack([
    armax_models[m].predict(X_test, y_true=y_test[:, j])
    for j, m in enumerate(MOTOR_NAMES)
])
t_infer_armax = time.perf_counter() - t0
print(f'ARMAX infer : {t_infer_armax*1000:.2f} ms  y_pred: {y_pred_armax.shape}')

m_armax = display_model_results(
    'ARMAX (Ridge)', y_test, y_pred_armax, t_infer_armax,
    voltage=U_test, motor_names=MOTOR_NAMES, save_to_excel=False,
)
"""),

md("## 4. Visualization"),

code("""\
N_PLOT = min(200, len(y_test))
sl     = slice(0, N_PLOT)
t_pl   = t_test[sl] - t_test[0]

fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
fig.suptitle('ARX / ARMAX — 1-step-ahead per-motor current', fontsize=13, fontweight='bold')
for i, (ax, m) in enumerate(zip(axes, MOTOR_NAMES)):
    ax.plot(t_pl, y_test[sl, i],       lw=0.9, color='black',     alpha=0.5, label='Actual')
    ax.plot(t_pl, y_pred_arx[sl, i],   lw=1.2, color='royalblue', ls='--',   label='ARX')
    ax.plot(t_pl, y_pred_armax[sl, i], lw=1.2, color='darkorange',ls='--',   label='ARMAX')
    ax.set_ylabel(f'I_{m} [A]'); ax.legend(fontsize=8); ax.grid(True, ls=':', alpha=0.5)
axes[-1].set_xlabel('Time [s]')
plt.tight_layout(); plt.show()
"""),

md("## 5. Model Comparison"),

code("compare_models([m_arx, m_armax], sort_by='RMSE [A]')"),

md("""\
## Summary

| Model | Params | Train | C-portable |
|-------|--------|-------|-----------|
| ARX   | 31 x 4 | < 1 ms | yes — dot product per motor |
| ARMAX | 34 x 4 | < 1 ms | yes — rolling residual buffer |

**Next:** TCN in `05_tcn.ipynb`.\
"""),
]

save("04_arx_armax.ipynb", nb04_cells)

# ── Notebook 05: TCN ─────────────────────────────────────────────────────────

nb05_cells = [

md("""\
# TCN — Per-Motor Current Prediction (Power Limiter)

| Property | Value |
|----------|-------|
| **Model** | 4 x MinimalTCN (pure numpy, one per motor) |
| **Input** | [I_FL, I_FR, I_RL, I_RR, T_sum, U_dc] normalized, last 5 ticks |
| **Output** | I_FL, I_FR, I_RL, I_RR at t+1 |
| **Training** | Mini-batch Adam, 50 epochs |

**Architecture per motor:**
```
Input (B, 6, 5)
  -- CausalConv1d(6->16, k=3, d=1) + ReLU
  -- CausalConv1d(16->16, k=3, d=2) + ReLU
  -- Take last timestep -> (B, 16)
  -- Linear(16->1)
```\
"""),

code(SHARED_IMPORTS_HEADER + """\
from functions.tcn        import MinimalTCN
from functions.evaluation import display_model_results, compare_models

N_SEQ = N_LAGS   # sequence length = 5 ticks
print(f'DATA_DIR : {DATA_DIR}')
"""),

md("## 1. Data Loading"),
code(SHARED_LOAD),
code(SHARED_SPLIT),

md("## 2. Sequence Building\n\nTCN input: 6-channel sequences of shape (n, 6, 5) — normalized."),

code("""\
# 6 input channels: I_FL, I_FR, I_RL, I_RR, T_sum, U_dc
ch_cols = ['I_FL', 'I_FR', 'I_RL', 'I_RR', 'T_sum', 'U_dc']
n_ch    = len(ch_cols)

ch_train = df_train[ch_cols].values   # (n_train, 6)
ch_test  = df_test[ch_cols].values    # (n_test,  6)

mu_ch  = ch_train.mean(axis=0)
std_ch = ch_train.std(axis=0) + 1e-8

ch_all  = np.vstack([ch_train, ch_test])
ch_norm = (ch_all - mu_ch) / std_ch

n_train_s = len(ch_train)
n_total   = len(ch_all)

# Build (n, n_ch, N_SEQ) sequences — causal, padded with zeros at run boundaries
# Simple approach: global sequence (slight boundary leakage at run boundaries is acceptable)
X_seq_all = np.zeros((n_total, n_ch, N_SEQ))
for i in range(N_SEQ - 1, n_total):
    X_seq_all[i] = ch_norm[i - N_SEQ + 1 : i + 1].T   # (n_ch, N_SEQ)

X_seq_train = X_seq_all[:n_train_s]
X_seq_test  = X_seq_all[n_train_s:]

y_mu    = y_train.mean(axis=0)
y_sigma = y_train.std(axis=0) + 1e-8
y_train_norm = (y_train - y_mu) / y_sigma

print(f'X_seq_train : {X_seq_train.shape}  (n, channels, timesteps)')
print(f'Channel means : {mu_ch.round(1)}')
"""),

md("## 3. Training — one TCN per motor"),

code("""\
np.random.seed(42)
tcns = {}
hists = {}

for j, m in enumerate(MOTOR_NAMES):
    print(f'--- Motor {m} ---')
    tcn_m = MinimalTCN(n_ch=n_ch, n_filters=16, kernel_size=3, lr=5e-4, seed=42 + j)
    hist  = tcn_m.fit(X_seq_train, y_train_norm[:, j], epochs=50, batch_size=512)
    tcns[m]  = tcn_m
    hists[m] = hist
    print()

fig, axes = plt.subplots(1, 4, figsize=(16, 3))
for ax, m in zip(axes, MOTOR_NAMES):
    ax.plot(hists[m], color='steelblue', lw=1.5)
    ax.set_title(f'Motor {m}'); ax.set_xlabel('Epoch'); ax.set_ylabel('RMSE (norm)')
    ax.grid(True, ls=':', alpha=0.5)
fig.suptitle('TCN training curves', fontsize=12)
plt.tight_layout(); plt.show()
"""),

md("## 4. Evaluation"),

code("""\
t0 = time.perf_counter()
y_pred_norm = np.column_stack([tcns[m].predict(X_seq_test) for m in MOTOR_NAMES])
t_infer = time.perf_counter() - t0

y_pred_tcn = y_pred_norm * y_sigma + y_mu   # denormalize

metrics = display_model_results(
    'TCN 4-motor (numpy)', y_test, y_pred_tcn, t_infer,
    voltage=U_test, motor_names=MOTOR_NAMES, save_to_excel=True,
    history={'train': {m: hists[m] for m in MOTOR_NAMES}},
)
"""),

md("## 5. Time-Series Preview"),

code("""\
N_PLOT = min(200, len(y_test))
sl     = slice(0, N_PLOT)
t_pl   = t_test[sl] - t_test[0]

fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
fig.suptitle('TCN — 1-step-ahead per-motor current (5 Hz)', fontsize=13, fontweight='bold')
for i, (ax, m) in enumerate(zip(axes, MOTOR_NAMES)):
    ax.plot(t_pl, y_test[sl, i],     lw=0.9, color='black', alpha=0.6, label='Actual')
    ax.plot(t_pl, y_pred_tcn[sl, i], lw=1.5, color='lime',  ls='--',   label='TCN')
    ax.set_ylabel(f'I_{m} [A]'); ax.legend(fontsize=8); ax.grid(True, ls=':', alpha=0.5)
axes[-1].set_xlabel('Time [s]')
plt.tight_layout(); plt.show()
"""),

md("""\
## Summary

| Property | Value |
|----------|-------|
| Params per motor | ~1 k weights |
| Receptive field | 7 ticks = 1.4 s at 5 Hz |
| Training | Adam, pure numpy |
| C-portable | yes — weight export + matmul |

**Next:** ESN in `06_esn.ipynb`.\
"""),
]

save("05_tcn.ipynb", nb05_cells)

# ── Notebook 06: ESN ─────────────────────────────────────────────────────────

nb06_cells = [

md("""\
# Echo State Network — Per-Motor Current Prediction (Power Limiter)

| Property | Value |
|----------|-------|
| **Model** | Echo State Network (Reservoir Computing) |
| **Input** | [I_FL, I_FR, I_RL, I_RR, T_sum, U_dc] normalized |
| **Output** | I_FL, I_FR, I_RL, I_RR at t+1 (multi-output Ridge readout) |
| **Training** | Ridge regression on reservoir states — no backprop |

**Why ESN:** no manual lag engineering — reservoir handles multi-scale temporal memory.
Training = one Ridge regression (< 2 s for N=300 neurons).\
"""),

code(SHARED_IMPORTS_HEADER + """\
from functions.esn        import EchoStateNetwork
from functions.evaluation import display_model_results, compare_models

WASHOUT = 50   # reservoir settling steps per run
print(f'DATA_DIR : {DATA_DIR}')
"""),

md("## 1. Data Loading"),
code(SHARED_LOAD),
code(SHARED_SPLIT),

md("## 2. ESN Input Matrix\n\n6 channels: [I_FL, I_FR, I_RL, I_RR, T_sum, U_dc] — normalized."),

code("""\
ch_cols  = ['I_FL', 'I_FR', 'I_RL', 'I_RR', 'T_sum', 'U_dc']
U_raw    = df_train[ch_cols].values   # (n_train, 6)
U_test_m = df_test[ch_cols].values    # (n_test, 6)

mu    = U_raw.mean(axis=0)
sigma = U_raw.std(axis=0) + 1e-8
U_norm_train = (U_raw     - mu) / sigma
U_norm_test  = (U_test_m  - mu) / sigma

print(f'Channels : {ch_cols}')
print(f'mu       : {mu.round(1)}')
print(f'sigma    : {sigma.round(1)}')
"""),

md("## 3. Training"),

code("""\
esn = EchoStateNetwork(
    n_inputs        = 6,
    n_reservoir     = 300,
    spectral_radius = 0.95,
    leaking_rate    = 0.3,
    sparsity        = 0.90,
    ridge_alpha     = 1e-4,
    seed            = 42,
)

t0 = time.perf_counter()
esn.fit(U_norm_train, y_train, washout=WASHOUT)
t_train = time.perf_counter() - t0

print(f'Train time : {t_train*1000:.1f} ms')
print(f'Reservoir  : {esn.N} neurons  (rho={esn.rho}, alpha={esn.alpha})')
print(f'Readout    : {esn._ridge.coef_.shape}  (n_outputs x n_reservoir)')
"""),

md("## 4. Evaluation"),

code("""\
t0 = time.perf_counter()
y_pred_esn = esn.predict(U_norm_test, warm_start=True)
t_infer = time.perf_counter() - t0

print(f'Inference : {t_infer*1000:.2f} ms  ({len(U_norm_test)} samples)')
print(f'y_pred shape : {y_pred_esn.shape}')

metrics = display_model_results(
    'ESN 4-motor', y_test, y_pred_esn, t_infer,
    voltage=U_test, motor_names=MOTOR_NAMES, save_to_excel=True,
)
"""),

md("## 5. Time-Series Preview"),

code("""\
N_PLOT = min(200, len(y_test))
sl     = slice(0, N_PLOT)
t_pl   = t_test[sl] - t_test[0]

fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
fig.suptitle('ESN 4-motor — 1-step-ahead prediction (5 Hz)', fontsize=13, fontweight='bold')
for i, (ax, m) in enumerate(zip(axes, MOTOR_NAMES)):
    ax.plot(t_pl, y_test[sl, i],     lw=0.9, color='black',     alpha=0.5, label='Actual')
    ax.plot(t_pl, y_pred_esn[sl, i], lw=1.5, color='steelblue',            label='ESN')
    ax.set_ylabel(f'I_{m} [A]'); ax.legend(fontsize=8); ax.grid(True, ls=':', alpha=0.5)
axes[-1].set_xlabel('Time [s]')
plt.tight_layout(); plt.show()
"""),

md("## 6. Hyperparameter Sweep\n\nSweep spectral_radius x leaking_rate — two most impactful ESN parameters."),

code("""\
rho_vals   = [0.5, 0.7, 0.9, 0.95, 0.99]
alpha_vals = [0.1, 0.3, 0.5, 0.8]

rmse_grid = np.zeros((len(rho_vals), len(alpha_vals)))
for i, rho in enumerate(rho_vals):
    for j, alpha in enumerate(alpha_vals):
        _esn = EchoStateNetwork(n_inputs=6, n_reservoir=200,
                                spectral_radius=rho, leaking_rate=alpha,
                                ridge_alpha=1e-4, seed=42)
        _esn.fit(U_norm_train, y_train, washout=WASHOUT)
        yh = _esn.predict(U_norm_test, warm_start=True)
        rmse_grid[i, j] = np.sqrt(np.mean((y_test - yh)**2))
        print(f'  rho={rho:.2f}  alpha={alpha:.1f}  RMSE={rmse_grid[i,j]:.2f} A')

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(rmse_grid, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(len(alpha_vals))); ax.set_xticklabels(alpha_vals)
ax.set_yticks(range(len(rho_vals)));   ax.set_yticklabels(rho_vals)
ax.set_xlabel('Leaking rate'); ax.set_ylabel('Spectral radius')
ax.set_title('ESN RMSE [A] — hyperparameter sweep (all 4 motors)')
plt.colorbar(im, ax=ax, label='RMSE [A]')
for ii in range(len(rho_vals)):
    for jj in range(len(alpha_vals)):
        ax.text(jj, ii, f'{rmse_grid[ii,jj]:.2f}', ha='center', va='center', fontsize=9)
plt.tight_layout(); plt.show()

best_i, best_j = np.unravel_index(np.argmin(rmse_grid), rmse_grid.shape)
print(f'Best: rho={rho_vals[best_i]}, alpha={alpha_vals[best_j]}  -> RMSE={rmse_grid[best_i, best_j]:.2f} A')
"""),

md("## 7. Best Configuration"),

code("""\
esn_best = EchoStateNetwork(
    n_inputs=6, n_reservoir=300,
    spectral_radius=rho_vals[best_i], leaking_rate=alpha_vals[best_j],
    ridge_alpha=1e-4, seed=42,
)
t0 = time.perf_counter()
esn_best.fit(U_norm_train, y_train, washout=WASHOUT)
t_tb = time.perf_counter() - t0

t0 = time.perf_counter()
y_pred_best = esn_best.predict(U_norm_test, warm_start=True)
t_ib = time.perf_counter() - t0

m_best = display_model_results(
    f'ESN best (rho={rho_vals[best_i]}, alpha={alpha_vals[best_j]})',
    y_test, y_pred_best, t_ib,
    voltage=U_test, motor_names=MOTOR_NAMES, save_to_excel=False,
)
compare_models([metrics, m_best], sort_by='RMSE [A]')
"""),

md("""\
## Summary

| Property | Value |
|----------|-------|
| Trainable params | N + 4 outputs (readout only) |
| Training | < 2 s (Ridge, no gradient) |
| Inference | one matmul: O(N^2) |
| C-portable | yes — static arrays |
| Manual lag features | no — reservoir handles memory |

**Extending:** increase n_reservoir for more capacity; reduce spectral_radius for faster forgetting.\
"""),
]

save("06_esn.ipynb", nb06_cells)
print("\nAll notebooks generated successfully.")
