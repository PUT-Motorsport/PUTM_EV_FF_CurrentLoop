# src/functions/evaluation.py
# Moduł ewaluacji dla predykcji PRĄDU KAŻDEGO SILNIKA z osobna (multi-output)
# oraz wynikającej z niego MOCY DC (P = U_batt * sum(I_silnikow)).
#
# Wzorowany na evaluation.py z projektu FF_Motor, rozszerzony o:
#   - obsługę wielu wyjść (wektor prądów silników),
#   - agregację prądów -> moc -> metryki przekroczenia limitu 80 kW,
#   - metryki asymetrii błędu (niedoszacowanie = kierunek groźny dla limitera),
#   - czas inferencji / częstotliwość (pod wdrożenie na VCU).

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

POWER_LIMIT_W = 80_000  # limit mocy z regulaminu Formula Student [W]


def _as_2d(a):
    """Zwraca tablicę 2D (n_probek, n_silnikow). Akceptuje 1D (1 silnik) lub 2D."""
    a = np.asarray(a)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def _compute_metrics(model_name, y_true, y_pred, inference_time,
                     voltage=None, power_limit_w=POWER_LIMIT_W, motor_names=None):
    """Metryki dla predykcji prądu per silnik + (opcjonalnie) mocy DC.

    y_true, y_pred : (n, n_silnikow) – prądy poszczególnych silników [A]
    voltage        : (n,) – napięcie baterii [V]; jeśli podane -> metryki MOCY
    """
    yt = _as_2d(y_true)
    yp = _as_2d(y_pred)
    n_motors = yt.shape[1]
    if motor_names is None:
        motor_names = [f"M{i+1}" for i in range(n_motors)]

    # Metryki liczone na spłaszczonym wektorze (łącznie dla wszystkich silników)
    yt_f, yp_f = yt.flatten(), yp.flatten()
    residuals = yt_f - yp_f  # >0 => niedoszacowanie (groźne)

    r2    = r2_score(yt_f, yp_f)
    rmse  = np.sqrt(mean_squared_error(yt_f, yp_f))
    mae   = mean_absolute_error(yt_f, yp_f)
    medae = median_absolute_error(yt_f, yp_f)
    mask  = yt_f != 0
    mape  = np.mean(np.abs(residuals[mask] / yt_f[mask])) * 100 if mask.any() else float('nan')

    ms_per_sample = (inference_time / yt.shape[0]) * 1000  # czas na PRÓBKĘ (cały wektor silników)
    freq_hz       = 1000 / ms_per_sample if ms_per_sample > 0 else float('inf')

    under = residuals > 0
    underpred_rate = float(np.mean(under)) * 100
    mean_under_err = float(np.mean(residuals[under])) if under.any() else 0.0

    metrics = {
        'model':           model_name,
        'R2':              r2,
        'RMSE [A]':        rmse,
        'MAE [A]':         mae,
        'MedAE [A]':       medae,
        'MAPE [%]':        mape,
        'Niedoszac. [%]':  underpred_rate,
        'Śr.niedosz. [A]': mean_under_err,
        'ms/próbkę':       ms_per_sample,
        'Częst. [Hz]':     freq_hz,
    }

    # R2 per silnik (diagnostyka, który silnik trudniejszy)
    for i, nm in enumerate(motor_names):
        metrics[f'R2_{nm}'] = r2_score(yt[:, i], yp[:, i])

    # --- METRYKI MOCY DC ---
    if voltage is not None:
        U = np.asarray(voltage).flatten()
        i_sum_true = yt.sum(axis=1)   # suma prądów silników [A]
        i_sum_pred = yp.sum(axis=1)
        p_true = U * i_sum_true       # moc DC [W]
        p_pred = U * i_sum_pred

        over_true = p_true > power_limit_w
        over_pred = p_pred > power_limit_w
        n_over = int(over_true.sum())
        fn = int((over_true & ~over_pred).sum())   # przeoczone przekroczenie (KRYTYCZNE)
        fp = int((~over_true & over_pred).sum())   # zbędne cięcie
        recall_over = (1 - fn / n_over) * 100 if n_over > 0 else float('nan')
        p_rmse_kw = np.sqrt(mean_squared_error(p_true, p_pred)) / 1000

        metrics.update({
            'Moc RMSE [kW]':      p_rmse_kw,
            'Przekr. real [n]':   n_over,
            'FN (przeoczone)':    fn,
            'FP (zbędne cięcia)': fp,
            'Recall przekr. [%]': recall_over,
        })

    return metrics


def save_results_to_excel(metrics, xlsx_path=None):
    """Zapis metryk do wspólnego Excela (jeden wiersz na model, nadpisywanie)."""
    if xlsx_path is None:
        xlsx_path = RESULTS_XLSX_PATH
    new_row = dict(metrics)
    new_row['Zaktualizowano'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if os.path.exists(xlsx_path):
        try:
            df = pd.read_excel(xlsx_path)
        except Exception as e:
            print(f"  [Excel] Ostrzeżenie: nie wczytano pliku ({e}). Tworzę nowy.")
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
        action = "zaktualizowano"
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        action = "dodano"

    ordered = columns + [c for c in df.columns if c not in columns]
    df = df[ordered]
    try:
        df.to_excel(xlsx_path, index=False)
        print(f"  [Excel] Wynik {action} → {xlsx_path}")
    except PermissionError:
        print(f"  [Excel] BŁĄD: plik '{xlsx_path}' otwarty w innym programie. Zamknij i ponów.")
    except Exception as e:
        print(f"  [Excel] BŁĄD zapisu: {e}")
    return xlsx_path


def display_model_results(
    model_name, y_true, y_pred, inference_time,
    voltage=None, power_limit_w=POWER_LIMIT_W, motor_names=None,
    history=None, history_metric='rmse', history_val_key=None,
    feature_importances=None, feature_names=None,
    subset_size=300, save_to_excel=True, excel_path=None,
):
    """Ujednolicone wyświetlanie wyników modelu predykcji PRĄDU PER SILNIK.

    Podaj `voltage` (napięcie baterii), aby policzyć metryki MOCY DC i przekroczeń 80 kW.
    `y_true`/`y_pred` mogą być 1D (jeden silnik) lub 2D (n_probek, n_silnikow).
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

    # --- WYDRUK ---
    sep = "=" * 56
    print(sep); print(f"  WYNIKI MODELU: {model_name}"); print(sep)
    print(f"  R2 (łącznie)    : {metrics['R2']:.4f}")
    print(f"  RMSE            : {metrics['RMSE [A]']:.2f} A")
    print(f"  MAE             : {metrics['MAE [A]']:.2f} A")
    print(f"  MedAE           : {metrics['MedAE [A]']:.2f} A")
    print(f"  MAPE            : {metrics['MAPE [%]']:.2f} %")
    print("  R2 per silnik   : " + "  ".join(f"{nm}={metrics[f'R2_{nm}']:.3f}" for nm in motor_names))
    print("-" * 56)
    print(f"  Niedoszacowania : {metrics['Niedoszac. [%]']:.1f} % próbek")
    print(f"  Śr. niedoszac.  : {metrics['Śr.niedosz. [A]']:.2f} A   (kierunek groźny!)")
    if voltage is not None:
        print("-" * 56)
        print(f"  Moc RMSE        : {metrics['Moc RMSE [kW]']:.2f} kW")
        print(f"  Realne przekr.  : {metrics['Przekr. real [n]']} zdarzeń > {power_limit_w/1000:.0f} kW")
        print(f"  Przeoczone (FN) : {metrics['FN (przeoczone)']}   <- KRYTYCZNE")
        print(f"  Zbędne cięcia FP: {metrics['FP (zbędne cięcia)']}")
        print(f"  Recall przekr.  : {metrics['Recall przekr. [%]']:.1f} %")
    print("-" * 56)
    print(f"  Czas/próbkę     : {metrics['ms/próbkę']:.4f} ms")
    print(f"  Częstotliwość   : {metrics['Częst. [Hz]']:.1f} Hz")
    print(sep + "\n")

    if save_to_excel:
        save_results_to_excel(metrics, xlsx_path=excel_path)
        print()

    # --- WYKRESY ---
    has_history     = history is not None
    has_importances = feature_importances is not None and feature_names is not None
    has_power       = voltage is not None
    n_extra = int(has_history) + int(has_importances) + int(has_power)
    n_rows  = 2 + int(np.ceil(n_extra / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=(18, 6 * n_rows))
    axes = axes.flatten()
    fig.suptitle(f'Ewaluacja modelu: {model_name}', fontsize=18, fontweight='bold', y=1.01)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # 1 – scatter rzeczywistość vs predykcja (wszystkie silniki)
    ax = axes[0]
    ax.scatter(yt_f, yp_f, alpha=0.2, color='steelblue', edgecolors='none', s=8)
    lo, hi = min(yt_f.min(), yp_f.min()), max(yt_f.max(), yp_f.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Ideał (y=x)')
    ax.set_title('1. Rzeczywistość vs Predykcja (wszystkie silniki)', fontsize=13)
    ax.set_xlabel('Rzeczywisty prąd [A]'); ax.set_ylabel('Przewidywany prąd [A]')
    ax.legend(); ax.grid(True, linestyle=':', alpha=0.6)

    # 2 – przebieg czasowy prądu pierwszego silnika
    ax = axes[1]
    n = min(subset_size, yt.shape[0])
    ax.plot(yt[:n, 0], label=f'Prawdziwy ({motor_names[0]})', color='black', lw=1.5, alpha=0.8)
    ax.plot(yp[:n, 0], label='Predykcja', color='lime', linestyle='--', lw=1.5)
    ax.set_title(f'2. Śledzenie prądu silnika {motor_names[0]} (pierwsze {n})', fontsize=13)
    ax.set_xlabel('Numer próbki'); ax.set_ylabel('Prąd [A]')
    ax.legend(); ax.grid(True, linestyle=':', alpha=0.6)

    # 3 – reszty
    ax = axes[2]
    ax.scatter(yp_f, residuals, alpha=0.15, color='mediumpurple', edgecolors='none', s=8)
    ax.axhline(0, color='red', lw=2)
    ax.set_title('3. Reszty (>0 = niedoszacowanie = groźne)', fontsize=13)
    ax.set_xlabel('Przewidywany prąd [A]'); ax.set_ylabel('Błąd [A]')
    ax.grid(True, linestyle=':', alpha=0.6)

    # 4 – histogram błędów
    ax = axes[3]
    q_lo, q_hi = np.percentile(residuals, 0.5), np.percentile(residuals, 99.5)
    clipped = residuals[(residuals > q_lo) & (residuals < q_hi)]
    sns.histplot(clipped, bins=60, ax=ax, color='darkorange', kde=True)
    ax.axvline(0, color='red', linestyle='--', lw=2, label='Zero')
    ax.axvline(np.median(residuals), color='blue', linestyle=':', lw=2,
               label=f'Mediana: {np.median(residuals):.1f} A')
    ax.set_title('4. Rozkład błędu [A]', fontsize=13)
    ax.set_xlabel('Błąd [A]'); ax.set_ylabel('Liczba'); ax.legend(fontsize=9)

    nx = 4
    # 5 – moc DC vs limit
    if has_power:
        ax = axes[nx]; nx += 1
        U = np.asarray(voltage).flatten()
        p_true = U * yt.sum(axis=1); p_pred = U * yp.sum(axis=1)
        n = min(subset_size, len(p_true))
        ax.plot(p_true[:n] / 1000, label='Rzeczywista moc', color='black', lw=1.5, alpha=0.8)
        ax.plot(p_pred[:n] / 1000, label='Przewidywana moc', color='lime', linestyle='--', lw=1.5)
        ax.axhline(power_limit_w / 1000, color='red', lw=2, label=f'Limit {power_limit_w/1000:.0f} kW')
        ax.set_title('5. Moc DC vs limit (P=U·ΣI)', fontsize=13)
        ax.set_xlabel('Numer próbki'); ax.set_ylabel('Moc [kW]')
        ax.legend(fontsize=9); ax.grid(True, linestyle=':', alpha=0.6)

    # 6 – krzywa uczenia
    if has_history:
        ax = axes[nx]; nx += 1
        _plot_learning_curve(ax, history, history_metric, history_val_key)

    # 7 – feature importance
    if has_importances:
        ax = axes[nx]; nx += 1
        _plot_feature_importance(ax, feature_importances, feature_names)

    for i in range(nx, len(axes)):
        axes[i].set_visible(False)
    plt.tight_layout(); plt.show()
    return metrics


def _plot_learning_curve(ax, history, metric_key, val_key):
    """Obsługuje XGBoost evals_result oraz historię Keras (MLP)."""
    if isinstance(history, dict) and all(isinstance(v, dict) for v in history.values()):
        keys = list(history.keys())
        inner = list(list(history.values())[0].keys())
        m = metric_key if metric_key in inner else inner[0]
        ax.plot(history[keys[0]][m], label=f'Trening ({m.upper()})')
        if len(keys) > 1:
            ax.plot(history[keys[1]][m], '--', label=f'Walidacja ({m.upper()})')
        ax.set_xlabel('Iteracje (drzewa)')
    else:
        if metric_key in history:
            ax.plot(history[metric_key], label=f'Trening ({metric_key})')
        v = val_key or f'val_{metric_key}'
        if v in history:
            ax.plot(history[v], '--', label=f'Walidacja ({v})')
        ax.set_xlabel('Epoka')
    ax.set_title('Krzywa uczenia', fontsize=13)
    ax.set_ylabel('Błąd'); ax.legend(); ax.grid(True, linestyle=':', alpha=0.6)


def _plot_feature_importance(ax, importances, names):
    importances = np.array(importances); names = np.array(names)
    idx = np.argsort(importances)[-15:]
    ax.barh(names[idx], importances[idx], color='teal')
    ax.set_title('Ważność cech', fontsize=13)
    ax.set_xlabel('Ważność'); ax.grid(True, linestyle=':', alpha=0.4, axis='x')


def compare_models(results_list, sort_by='RMSE [A]', ascending=True):
    """Tabela + wykresy porównawcze."""
    df = pd.DataFrame(results_list).set_index('model')
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)
    print("\n" + "=" * 70); print("  PORÓWNANIE MODELI"); print("=" * 70)
    print(df.to_string(float_format=lambda x: f'{x:.3f}'))
    print("=" * 70 + "\n")

    cols = [c for c in ['R2', 'RMSE [A]', 'MAE [A]', 'Niedoszac. [%]', 'ms/próbkę', 'Recall przekr. [%]']
            if c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()
    fig.suptitle('Porównanie modeli', fontsize=16, fontweight='bold')
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
