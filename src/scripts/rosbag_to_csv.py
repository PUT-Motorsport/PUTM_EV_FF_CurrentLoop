"""
ETL: extract signals from .mcap rosbag files and write/append to a single CSV.

Incremental mode:
  A manifest file (training_data_manifest.json) tracks already-processed files
  by name + size. Each run processes ONLY new files and appends their rows to
  the existing CSV.

CSV columns (aligned to current_sensor time grid, ~5 Hz):
  timestamp_s    - absolute time [s]
  run_id         - unique recording index (monotonically increasing across runs)
  I_FL/FR/RL/RR  - inverter currents from /putm_vcl/current_sensor [raw uint16]
  T_FL/FR/RL/RR  - torque setpoints from /putm_vcl/setpoints [Nm, int32]
  v_FL/FR/RL/RR  - motor velocities from actual_values1 [RPM, int16]
  Iq_FL/FR/RL/RR - q-axis (torque) current from actual_values1 [raw int16]
  U_dc           - HV battery voltage from bms_hv_main [raw/10 = V]

CURRENT_SCALE: set according to AMK sensor documentation.
  Currently 1.0 (raw values). Candidate: 0.1 (raw digit = 0.1 A).

Usage:
  python src/rosbag_to_csv.py                # process only new recordings
  python src/rosbag_to_csv.py --include-old  # also include data/rosbag2/old/
  python src/rosbag_to_csv.py --rebuild      # clear manifest and CSV, rebuild from scratch
"""

import argparse
import json
import struct
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from mcap.reader import make_reader

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent
ROSBAG_DIR    = PROJECT_ROOT / 'data' / 'rosbag2'
OUT_CSV       = PROJECT_ROOT / 'data' / 'model' / 'training_data.csv'
MANIFEST_PATH = PROJECT_ROOT / 'data' / 'model' / 'training_data_manifest.json'

# ── scaling factors ───────────────────────────────────────────────────────────
CURRENT_SCALE  = 1.0    # raw uint16 -> A  (TBD: likely 0.1)
TORQUE_SCALE   = 1.0    # int32 -> Nm (already in Nm)
VELOCITY_SCALE = 1.0    # int16 -> RPM (already in RPM)
VOLTAGE_SCALE  = 0.1    # uint16 -> V  (verified: 5633 raw = 563.3 V)

# ── topics ───────────────────────────────────────────────────────────────────
MOTOR_POSITIONS = ['front/left', 'front/right', 'rear/left', 'rear/right']
MOTOR_SHORT     = ['FL', 'FR', 'RL', 'RR']

TOPIC_CS   = '/putm_vcl/current_sensor'
TOPIC_SP   = '/putm_vcl/setpoints'
TOPIC_BMS  = '/putm_vcl/bms_hv_main'
TOPICS_AV1 = {
    f'/putm_vcl/amk/{pos}/actual_values1': short
    for pos, short in zip(MOTOR_POSITIONS, MOTOR_SHORT)
}
ALL_TOPICS = {TOPIC_CS, TOPIC_SP, TOPIC_BMS, *TOPICS_AV1}


# ── manifest ──────────────────────────────────────────────────────────────────

def _file_key(path: Path) -> str:
    """Unique file key: name + size (detects replaced/modified files)."""
    return f'{path.name}:{path.stat().st_size}'


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {'processed': {}, 'next_run_id': 0}


def save_manifest(manifest: dict) -> None:
    manifest['last_updated'] = datetime.now().isoformat(timespec='seconds')
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


# ── CDR decoders ──────────────────────────────────────────────────────────────

def decode_current_sensor(data: bytes):
    """-> (FL, FR, RL, RR) uint16 raw inverter currents"""
    return struct.unpack_from('<4H', data, 4)


def decode_setpoints(data: bytes):
    """-> (FL, FR, RL, RR) int32 torque setpoints [Nm]"""
    return struct.unpack_from('<4i', data, 4)


def decode_actual_values1(data: bytes):
    """-> (velocity int16, torque_current int16, magnetizing_current int16)
    Layout: [4:12] AmkStatus (8 booleans), [12:18] 3x int16
    """
    vel, torq, mag = struct.unpack_from('<3h', data, 12)
    return vel, torq, mag


def decode_bms_hv_main(data: bytes):
    """-> voltage_sum uint16 raw  (/10 = Volts)"""
    return struct.unpack_from('<H', data, 4)[0]


# ── single file processing ────────────────────────────────────────────────────

def process_mcap(path: Path, run_id: int) -> pd.DataFrame | None:
    print(f'  [run {run_id}] {path.name}', end=' ... ', flush=True)

    raw_cs  = []
    raw_sp  = []
    raw_bms = []
    raw_av1 = {m: [] for m in MOTOR_SHORT}

    with open(path, 'rb') as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        available = {ch.topic for ch in summary.channels.values()}
        topics_to_read = list(ALL_TOPICS & available)

        if TOPIC_CS not in available:
            print('no current_sensor topic — skipping')
            return None

        for _schema, channel, message in reader.iter_messages(topics=topics_to_read):
            t  = message.log_time / 1e9
            tp = channel.topic
            d  = message.data

            if tp == TOPIC_CS:
                raw_cs.append((t, *decode_current_sensor(d)))
            elif tp == TOPIC_SP:
                raw_sp.append((t, *decode_setpoints(d)))
            elif tp == TOPIC_BMS:
                raw_bms.append((t, decode_bms_hv_main(d)))
            elif tp in TOPICS_AV1:
                m = TOPICS_AV1[tp]
                raw_av1[m].append((t, *decode_actual_values1(d)))

    if not raw_cs:
        print('no current_sensor data — skipping')
        return None

    df_cs = pd.DataFrame(raw_cs, columns=['t', 'I_FL', 'I_FR', 'I_RL', 'I_RR'])
    df_cs = df_cs.set_index('t').sort_index()

    df_sp = (
        pd.DataFrame(raw_sp, columns=['t', 'T_FL', 'T_FR', 'T_RL', 'T_RR'])
        .set_index('t').sort_index()
        if raw_sp else pd.DataFrame()
    )

    df_bms = (
        pd.DataFrame(raw_bms, columns=['t', 'U_raw'])
        .set_index('t').sort_index()
        if raw_bms else pd.DataFrame()
    )

    av1_dfs = {}
    for m in MOTOR_SHORT:
        if raw_av1[m]:
            av1_dfs[m] = pd.DataFrame(
                raw_av1[m],
                columns=['t', f'v_{m}', f'Iq_{m}', f'Id_{m}']
            ).set_index('t').sort_index()

    # Interpolate all streams onto the current_sensor time grid (~5 Hz)
    t_grid = df_cs.index.values

    def interp_cols(df: pd.DataFrame, cols: list[str]) -> dict:
        return {
            col: np.interp(t_grid, df.index.values, df[col].values)
            for col in cols if col in df.columns
        }

    result: dict = {'timestamp_s': t_grid, 'run_id': run_id}

    for col in ['I_FL', 'I_FR', 'I_RL', 'I_RR']:
        result[col] = df_cs[col].values * CURRENT_SCALE

    if not df_sp.empty:
        d = interp_cols(df_sp, ['T_FL', 'T_FR', 'T_RL', 'T_RR'])
        for col in ['T_FL', 'T_FR', 'T_RL', 'T_RR']:
            result[col] = d.get(col, np.full(len(t_grid), np.nan)) * TORQUE_SCALE
    else:
        for col in ['T_FL', 'T_FR', 'T_RL', 'T_RR']:
            result[col] = np.nan

    for m in MOTOR_SHORT:
        if m in av1_dfs:
            d = interp_cols(av1_dfs[m], [f'v_{m}', f'Iq_{m}'])
            result[f'v_{m}']  = d.get(f'v_{m}',  np.full(len(t_grid), np.nan)) * VELOCITY_SCALE
            result[f'Iq_{m}'] = d.get(f'Iq_{m}', np.full(len(t_grid), np.nan)) * CURRENT_SCALE
        else:
            result[f'v_{m}']  = np.nan
            result[f'Iq_{m}'] = np.nan

    if not df_bms.empty:
        result['U_dc'] = (
            np.interp(t_grid, df_bms.index.values, df_bms['U_raw'].values)
            * VOLTAGE_SCALE
        )
    else:
        result['U_dc'] = np.nan

    df_out = pd.DataFrame(result)
    print(f'{len(df_out)} samples  [{df_cs.index[0]:.1f}–{df_cs.index[-1]:.1f} s]')
    return df_out


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='ETL: .mcap rosbags -> training_data.csv')
    parser.add_argument('--include-old', action='store_true',
                        help='Also include files from data/rosbag2/old/')
    parser.add_argument('--rebuild', action='store_true',
                        help='Clear manifest and CSV, rebuild from scratch')
    args = parser.parse_args()

    mcap_files = sorted(ROSBAG_DIR.glob('*.mcap'))
    if args.include_old:
        mcap_files += sorted((ROSBAG_DIR / 'old').glob('*.mcap'))

    if not mcap_files:
        print(f'No .mcap files found in {ROSBAG_DIR}')
        return

    if args.rebuild:
        manifest = {'processed': {}, 'next_run_id': 0}
        if OUT_CSV.exists():
            OUT_CSV.unlink()
        print('--rebuild: cleared manifest and CSV.\n')
    else:
        manifest = load_manifest()

    processed: dict = manifest['processed']
    next_run_id: int = manifest['next_run_id']

    new_files = [p for p in mcap_files if _file_key(p) not in processed]

    if not new_files:
        print(f'No new recordings found. CSV is up to date ({OUT_CSV.name}).')
        print(f'Processed: {len(processed)} files, {next_run_id} runs total.')
        return

    print(f'Total .mcap files : {len(mcap_files)}')
    print(f'Already processed : {len(processed)}')
    print(f'New to process    : {len(new_files)}\n')

    t0 = time.time()
    new_frames = []

    for path in new_files:
        df = process_mcap(path, run_id=next_run_id)
        key = _file_key(path)
        if df is not None:
            new_frames.append(df)
            processed[key] = next_run_id
        else:
            processed[key] = None  # remember skipped files to avoid re-checking
        next_run_id += 1

    manifest['next_run_id'] = next_run_id

    if not new_frames:
        print('\nNo new data extracted (all new files lacked current_sensor topic).')
        save_manifest(manifest)
        return

    new_data = pd.concat(new_frames, ignore_index=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if OUT_CSV.exists() and not args.rebuild:
        new_data.to_csv(OUT_CSV, mode='a', header=False, index=False)
        total_rows = sum(1 for _ in open(OUT_CSV)) - 1
    else:
        new_data.to_csv(OUT_CSV, index=False)
        total_rows = len(new_data)

    save_manifest(manifest)

    print(f'\n{"="*60}')
    print(f'Added    : {len(new_data):,} rows from {len(new_frames)} recordings')
    print(f'CSV total: {total_rows:,} rows  ->  {OUT_CSV}')
    print(f'Time     : {time.time()-t0:.1f} s')
    print(f'\nNew recordings:')
    for df in new_frames:
        rid = int(df['run_id'].iloc[0])
        fname = next(p.name for p in new_files if processed.get(_file_key(p)) == rid)
        print(f'  run {rid}: {fname}  ({len(df)} samples)')


if __name__ == '__main__':
    main()
