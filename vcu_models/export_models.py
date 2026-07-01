"""
Export trained ARX Q=0.90 and XGBoost Q=0.90 models to files for ROS2 deployment.

Outputs written to vcu_models/model_weights/:
  arx_q90_weights.json    -- ARX coefficients (4 x 31) + intercepts
  xgb_q90_FL.ubj          -- XGBoost model for motor FL
  xgb_q90_FR.ubj          -- XGBoost model for motor FR
  xgb_q90_RL.ubj          -- XGBoost model for motor RL
  xgb_q90_RR.ubj          -- XGBoost model for motor RR
  xgb_meta.json           -- feature names and motor order (for XGBPredictor)

Usage:
  python vcu_models/export_models.py

After running, copy model_weights/ to the ROS2 package:
  cp -r vcu_models/model_weights/* vcu_models/current_predictor/model_weights/
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

VCU_DIR     = Path(__file__).parent
ROOT_DIR    = VCU_DIR.parent
SRC_DIR     = ROOT_DIR / 'src'
DATA_DIR    = ROOT_DIR / 'data' / 'model'
WEIGHTS_DIR = VCU_DIR / 'model_weights'
WEIGHTS_DIR.mkdir(exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from functions.arx import ARXModel

N_LAGS      = 5
MOTOR_NAMES = ['FL', 'FR', 'RL', 'RR']


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['T_sum'] = df['T_FL'] + df['T_FR'] + df['T_RL'] + df['T_RR']
    for m in MOTOR_NAMES:
        for lag in range(1, N_LAGS + 1):
            df[f'I_{m}_lag{lag}'] = df.groupby('run_id')[f'I_{m}'].shift(lag)
    for lag in range(1, N_LAGS + 1):
        df[f'T_sum_lag{lag}'] = df.groupby('run_id')['T_sum'].shift(lag)
    for m in MOTOR_NAMES:
        df[f'I_{m}_next'] = df.groupby('run_id')[f'I_{m}'].shift(-1)
    return df.dropna().reset_index(drop=True)


def _feature_cols() -> list[str]:
    cols = []
    for m in MOTOR_NAMES:
        cols.append(f'I_{m}')
        for lag in range(1, N_LAGS + 1):
            cols.append(f'I_{m}_lag{lag}')
    cols.append('T_sum')
    for lag in range(1, N_LAGS + 1):
        cols.append(f'T_sum_lag{lag}')
    cols.append('U_dc')
    return cols  # 31 features


def main() -> None:
    print('=' * 60)
    print('Model export: ARX Q=0.90 + XGBoost Q=0.90')
    print('=' * 60)

    print('\nLoading training data...')
    df_raw = pd.read_csv(DATA_DIR / 'training_data.csv')
    df_raw = df_raw.sort_values(['run_id', 'timestamp_s']).reset_index(drop=True)
    df = _build_features(df_raw)

    feature_cols = _feature_cols()
    runs = sorted(df['run_id'].unique())
    n_train = max(1, int(len(runs) * 0.70))
    train_runs = runs[:n_train]

    df_train = df[df['run_id'].isin(train_runs)].reset_index(drop=True)
    X_train  = df_train[feature_cols].values
    y_train  = df_train[[f'I_{m}_next' for m in MOTOR_NAMES]].values
    print(f'  Train runs {train_runs}: {len(df_train)} samples, {len(feature_cols)} features')

    # ----------------------------------------------------------------
    # ARX Q=0.90
    # ----------------------------------------------------------------
    print('\n[1/2] Training ARX Q=0.90 ...')
    t0 = time.perf_counter()
    arx = ARXModel(alpha=1.0, quantile=0.90)
    arx.fit(X_train, y_train)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f'  Done in {elapsed:.0f} ms')

    arx_data = {
        'feature_names': feature_cols,
        'motors':        MOTOR_NAMES,
        'coef':          arx.coef_.tolist(),       # (4, 31)
        'intercept':     arx.intercept_.tolist(),  # (4,)
        'quantile':      0.90,
        'n_lags':        N_LAGS,
    }
    arx_path = WEIGHTS_DIR / 'arx_q90_weights.json'
    with open(arx_path, 'w') as f:
        json.dump(arx_data, f, indent=2)
    print(f'  Saved -> {arx_path}')

    # Sanity check: one sample prediction
    x_sample = X_train[0]
    pred = arx.coef_ @ x_sample + arx.intercept_
    print(f'  Sample prediction (train[0]): {pred.round(1)}')

    # ----------------------------------------------------------------
    # XGBoost Q=0.90
    # ----------------------------------------------------------------
    xgb_params = {
        'n_estimators':     200,
        'max_depth':        3,
        'learning_rate':    0.1,
        'subsample':        0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 1,
    }
    print('\n[2/2] Training XGBoost Q=0.90 (4 motors) ...')
    t0 = time.perf_counter()
    for j, m in enumerate(MOTOR_NAMES):
        model = xgb.XGBRegressor(
            **xgb_params,
            objective='reg:quantileerror',
            quantile_alpha=0.90,
            tree_method='hist',
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train[:, j])
        path = WEIGHTS_DIR / f'xgb_q90_{m}.ubj'
        model.save_model(str(path))
        print(f'  Motor {m} -> {path}')
    print(f'  Total: {time.perf_counter() - t0:.1f} s')

    meta = {
        'feature_names': feature_cols,
        'motors':        MOTOR_NAMES,
        'n_lags':        N_LAGS,
        'quantile':      0.90,
        'params':        xgb_params,
    }
    meta_path = WEIGHTS_DIR / 'xgb_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  Meta  -> {meta_path}')

    print('\n' + '=' * 60)
    print(f'All weights saved to: {WEIGHTS_DIR}')
    print('\nNext steps:')
    print('  cp -r model_weights/* ros2_pkg/current_predictor/model_weights/')
    print('  cd <your_ros2_ws> && colcon build --packages-select current_predictor')
    print('  ros2 launch current_predictor current_predictor.launch.py')
    print('=' * 60)


if __name__ == '__main__':
    main()
