# src/functions/evaluation.py
# Evaluation module for per-motor current prediction (multi-output)
# and DC power derived from it (P = U_batt * sum(I_motors)).

import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error


RESULTS_XLSX_NAME = "model_results.xlsx"
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../src
RESULTS_XLSX_PATH = os.path.join(_PROJECT_DIR, RESULTS_XLSX_NAME)

POWER_LIMIT_W = 80_000  # Formula Student power limit [W]


def _as_2d(a):
    """Return 2-D array (n_samples, n_motors). Accepts 1-D (single motor) or 2-D."""
    a = np.asarray(a)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def _compute_metrics(model_name, y_true, y_pred, inference_time,
                     voltage=None, power_limit_w=POWER_LIMIT_W, motor_names=None):
    """Per-motor current prediction metrics + optional DC power metrics.

    y_true, y_pred : (n, n_motors) — motor currents [A]
    voltage        : (n,) — battery voltage [V]; if provided → power metrics
    """
    yt = _as_2d(y_true)
    yp = _as_2d(y_pred)
    n_motors = yt.shape[1]
    if motor_names is None:
        motor_names = [f"M{i+1}" for i in range(n_motors)]

    yt_f, yp_f = yt.flatten(), yp.flatten()
    residuals = yt_f - yp_f  # >0 → under-prediction (dangerous for limiter)

    r2    = r2_score(yt_f, yp_f)
    rmse  = np.sqrt(mean_squared_error(yt_f, yp_f))
    mae   = mean_absolute_error(yt_f, yp_f)
    medae = median_absolute_error(yt_f, yp_f)
    mask  = yt_f != 0
    mape  = np.mean(np.abs(residuals[mask] / yt_f[mask])) * 100 if mask.any() else float('nan')

    ms_per_sample = (inference_time / yt.shape[0]) * 1000
    freq_hz       = 1000 / ms_per_sample if ms_per_sample > 0 else float('inf')

    under = residuals > 0
    underpred_rate = float(np.mean(under)) * 100
    mean_under_err = float(np.mean(residuals[under])) if under.any() else 0.0

    metrics = {
        'model':               model_name,
        'R2':                  r2,
        'RMSE [A]':            rmse,
        'MAE [A]':             mae,
        'MedAE [A]':           medae,
        'MAPE [%]':            mape,
        'Underpred. [%]':      underpred_rate,
        'Mean underpred. [A]': mean_under_err,
        'ms/sample':           ms_per_sample,
        'Freq. [Hz]':          freq_hz,
    }

    for i, nm in enumerate(motor_names):
        metrics[f'R2_{nm}'] = r2_score(yt[:, i], yp[:, i])

    if voltage is not None:
        U = np.asarray(voltage).flatten()
        i_sum_true = yt.sum(axis=1)
        i_sum_pred = yp.sum(axis=1)
        p_true = U * i_sum_true
        p_pred = U * i_sum_pred

        over_true = p_true > power_limit_w
        over_pred = p_pred > power_limit_w
        n_over = int(over_true.sum())
        fn = int((over_true & ~over_pred).sum())
        fp = int((~over_true & over_pred).sum())
        recall_over = (1 - fn / n_over) * 100 if n_over > 0 else float('nan')
        p_rmse_kw = np.sqrt(mean_squared_error(p_true, p_pred)) / 1000

        metrics.update({
            'Power RMSE [kW]':    p_rmse_kw,
            'Violations [n]':     n_over,
            'FN (missed)':        fn,
            'FP (unnecessary)':   fp,
            'Violation recall [%]': recall_over,
        })

    return metrics


def save_results_to_excel(metrics, xlsx_path=None):
    """Append or update a row in the shared results Excel file."""
    if xlsx_path is None:
        xlsx_path = RESULTS_XLSX_PATH
    new_row = dict(metrics)
    new_row['Updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if os.path.exists(xlsx_path):
        try:
            df = pd.read_excel(xlsx_path)
        except Exception as e:
            print(f"  [Excel] Warning: could not read file ({e}). Creating new.")
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    columns = list(new_row.keys())
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA

    name = new_row['model']
    if 'model' in df.columns and (df['model'] == name).any():
        idx = df.index[df['model'] == name][0]
        for col in columns:
            df.at[idx, col] = new_row.get(col, pd.NA)
        action = "updated"
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        action = "added"

    ordered = columns + [c for c in df.columns if c not in columns]
    df = df[ordered]
    try:
        df.to_excel(xlsx_path, index=False)
        print(f"  [Excel] Result {action} → {xlsx_path}")
    except PermissionError:
        print(f"  [Excel] ERROR: '{xlsx_path}' is open in another program. Close it and retry.")
    except Exception as e:
        print(f"  [Excel] Write error: {e}")
    return xlsx_path


def display_model_results(
    model_name, y_true, y_pred, inference_time,
    voltage=None, power_limit_w=POWER_LIMIT_W, motor_names=None,
    history=None, history_metric='rmse', history_val_key=None,
    feature_importances=None, feature_names=None,
    subset_size=300, save_to_excel=True, excel_path=None,
):
    """Display evaluation results for a current prediction model.

    Pass `voltage` to compute DC power metrics and 80 kW limit violations.
    `y_true`/`y_pred` can be 1-D (single motor) or 2-D (n_samples, n_motors).
    """
    yt = _as_2d(y_true)
    yp = _as_2d(y_pred)
    n_motors = yt.shape[1]
    if motor_names is None:
        motor_names = [f"M{i+1}" for i in range(n_motors)]

    metrics = _compute_metrics(model_name, yt, yp, inference_time,
                               voltage=voltage, power_limit_w=power_limit_w,
                               motor_names=motor_names)

    yt_f, yp_f = yt.flatten(), yp.flatten()
    residuals = yt_f - yp_f

    sep = "=" * 56
    print(sep); print(f"  MODEL: {model_name}"); print(sep)
    print(f"  R2              : {metrics['R2']:.4f}")
    print(f"  RMSE            : {metrics['RMSE [A]']:.2f} A")
    print(f"  MAE             : {metrics['MAE [A]']:.2f} A")
    print(f"  MedAE           : {metrics['MedAE [A]']:.2f} A")
    print(f"  MAPE            : {metrics['MAPE [%]']:.2f} %")
    print("  R2 per motor    : " + "  ".join(f"{nm}={metrics[f'R2_{nm}']:.3f}" for nm in motor_names))
    print("-" * 56)
    print(f"  Under-pred.     : {metrics['Underpred. [%]']:.1f} % of samples")
    print(f"  Mean under-pred.: {metrics['Mean underpred. [A]']:.2f} A   (dangerous direction!)")
    if voltage is not None:
        print("-" * 56)
        print(f"  Power RMSE      : {metrics['Power RMSE [kW]']:.2f} kW")
        print(f"  Real violations : {metrics['Violations [n]']} events > {power_limit_w/1000:.0f} kW")
        print(f"  Missed (FN)     : {metrics['FN (missed)']}   <- CRITICAL")
        print(f"  Unnecessary (FP): {metrics['FP (unnecessary)']}")
        print(f"  Violation recall: {metrics['Violation recall [%]']:.1f} %")
    print("-" * 56)
    print(f"  Time/sample     : {metrics['ms/sample']:.4f} ms")
    print(f"  Frequency       : {metrics['Freq. [Hz]']:.1f} Hz")
    print(sep + "\n")

    if save_to_excel:
        save_results_to_excel(metrics, xlsx_path=excel_path)
        print()

    has_history     = history is not None
    has_importances = feature_importances is not None and feature_names is not None
    has_power       = voltage is not None
    n_extra = int(has_history) + int(has_importances) + int(has_power)
    n_rows  = 2 + int(np.ceil(n_extra / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=(18, 6 * n_rows))
    axes = axes.flatten()
    fig.suptitle(f'Model evaluation: {model_name}', fontsize=18, fontweight='bold', y=1.01)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    ax = axes[0]
    ax.scatter(yt_f, yp_f, alpha=0.2, color='steelblue', edgecolors='none', s=8)
    lo, hi = min(yt_f.min(), yp_f.min()), max(yt_f.max(), yp_f.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Ideal (y=x)')
    ax.set_title('1. Actual vs Predicted (all motors)', fontsize=13)
    ax.set_xlabel('Actual current [A]'); ax.set_ylabel('Predicted current [A]')
    ax.legend(); ax.grid(True, linestyle=':', alpha=0.6)

    ax = axes[1]
    n = min(subset_size, yt.shape[0])
    ax.plot(yt[:n, 0], label=f'Actual ({motor_names[0]})', color='black', lw=1.5, alpha=0.8)
    ax.plot(yp[:n, 0], label='Prediction', color='lime', linestyle='--', lw=1.5)
    ax.set_title(f'2. Current tracking — {motor_names[0]} (first {n} samples)', fontsize=13)
    ax.set_xlabel('Sample index'); ax.set_ylabel('Current [A]')
    ax.legend(); ax.grid(True, linestyle=':', alpha=0.6)

    ax = axes[2]
    ax.scatter(yp_f, residuals, alpha=0.15, color='mediumpurple', edgecolors='none', s=8)
    ax.axhline(0, color='red', lw=2)
    ax.set_title('3. Residuals (>0 = under-prediction = dangerous)', fontsize=13)
    ax.set_xlabel('Predicted current [A]'); ax.set_ylabel('Error [A]')
    ax.grid(True, linestyle=':', alpha=0.6)

    ax = axes[3]
    q_lo, q_hi = np.percentile(residuals, 0.5), np.percentile(residuals, 99.5)
    clipped = residuals[(residuals > q_lo) & (residuals < q_hi)]
    sns.histplot(clipped, bins=60, ax=ax, color='darkorange', kde=True)
    ax.axvline(0, color='red', linestyle='--', lw=2, label='Zero')
    ax.axvline(np.median(residuals), color='blue', linestyle=':', lw=2,
               label=f'Median: {np.median(residuals):.1f} A')
    ax.set_title('4. Error distribution [A]', fontsize=13)
    ax.set_xlabel('Error [A]'); ax.set_ylabel('Count'); ax.legend(fontsize=9)

    nx = 4
    if has_power:
        ax = axes[nx]; nx += 1
        U = np.asarray(voltage).flatten()
        p_true = U * yt.sum(axis=1); p_pred = U * yp.sum(axis=1)
        n = min(subset_size, len(p_true))
        ax.plot(p_true[:n] / 1000, label='Actual power', color='black', lw=1.5, alpha=0.8)
        ax.plot(p_pred[:n] / 1000, label='Predicted power', color='lime', linestyle='--', lw=1.5)
        ax.axhline(power_limit_w / 1000, color='red', lw=2, label=f'{power_limit_w/1000:.0f} kW limit')
        ax.set_title('5. DC power vs limit (P = U·ΣI)', fontsize=13)
        ax.set_xlabel('Sample index'); ax.set_ylabel('Power [kW]')
        ax.legend(fontsize=9); ax.grid(True, linestyle=':', alpha=0.6)

    if has_history:
        ax = axes[nx]; nx += 1
        _plot_learning_curve(ax, history, history_metric, history_val_key)

    if has_importances:
        ax = axes[nx]; nx += 1
        _plot_feature_importance(ax, feature_importances, feature_names)

    for i in range(nx, len(axes)):
        axes[i].set_visible(False)
    plt.tight_layout(); plt.show()
    return metrics


def _plot_learning_curve(ax, history, metric_key, val_key):
    if isinstance(history, dict) and all(isinstance(v, dict) for v in history.values()):
        keys = list(history.keys())
        inner = list(list(history.values())[0].keys())
        m = metric_key if metric_key in inner else inner[0]
        ax.plot(history[keys[0]][m], label=f'Train ({m.upper()})')
        if len(keys) > 1:
            ax.plot(history[keys[1]][m], '--', label=f'Validation ({m.upper()})')
        ax.set_xlabel('Iterations (trees)')
    else:
        if metric_key in history:
            ax.plot(history[metric_key], label=f'Train ({metric_key})')
        v = val_key or f'val_{metric_key}'
        if v in history:
            ax.plot(history[v], '--', label=f'Validation ({v})')
        ax.set_xlabel('Epoch')
    ax.set_title('Learning curve', fontsize=13)
    ax.set_ylabel('Loss'); ax.legend(); ax.grid(True, linestyle=':', alpha=0.6)


def _plot_feature_importance(ax, importances, names):
    importances = np.array(importances); names = np.array(names)
    idx = np.argsort(importances)[-15:]
    ax.barh(names[idx], importances[idx], color='teal')
    ax.set_title('Feature importance', fontsize=13)
    ax.set_xlabel('Importance'); ax.grid(True, linestyle=':', alpha=0.4, axis='x')


def compare_models(results_list, sort_by='RMSE [A]', ascending=True):
    """Table + comparison charts for multiple models."""
    df = pd.DataFrame(results_list).set_index('model')
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)
    print("\n" + "=" * 70); print("  MODEL COMPARISON"); print("=" * 70)
    print(df.to_string(float_format=lambda x: f'{x:.3f}'))
    print("=" * 70 + "\n")

    cols = [c for c in ['R2', 'RMSE [A]', 'MAE [A]', 'Underpred. [%]', 'ms/sample', 'Violation recall [%]']
            if c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()
    fig.suptitle('Model comparison', fontsize=16, fontweight='bold')
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
    for i, col in enumerate(cols):
        ax = axes[i]
        bars = ax.bar(df.index, df[col], color=colors)
        ax.set_title(col, fontsize=12); ax.set_ylabel(col)
        ax.tick_params(axis='x', rotation=20); ax.grid(True, linestyle=':', alpha=0.5, axis='y')
        for bar, val in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    for i in range(len(cols), len(axes)):
        axes[i].set_visible(False)
    plt.tight_layout(); plt.show()
    return df
