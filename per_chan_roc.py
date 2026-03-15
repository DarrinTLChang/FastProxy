import numpy as np
import os
import sys
import pandas as pd
from sklearn.metrics import roc_curve, auc


# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════
BURST_PAD_BEFORE = 0.020   # 20 ms before burst_start
BURST_PAD_AFTER  = 0.000   # 0 ms after burst_end


def load_bursts(csv_path):
    df = pd.read_csv(csv_path)
    return df['burst_start_ms'].values / 1000.0, df['burst_end_ms'].values / 1000.0


def create_burst_labels(time_s, burst_starts, burst_ends):
    labels = np.zeros(len(time_s), dtype=int)
    for start, end in zip(burst_starts, burst_ends):
        mask = (time_s >= start - BURST_PAD_BEFORE) & (time_s <= end + BURST_PAD_AFTER)
        labels[mask] = 1
    return labels


def find_youden_j_threshold(fpr, tpr, thresholds):
    j_scores = tpr - fpr

    finite_mask = np.isfinite(thresholds)
    if np.any(finite_mask):
        valid_indices = np.where(finite_mask)[0]
        best_local_idx = np.argmax(j_scores[finite_mask])
        idx = valid_indices[best_local_idx]
    else:
        idx = np.argmax(j_scores)

    return {
        'threshold': thresholds[idx],
        'sensitivity': tpr[idx],
        'specificity': 1 - fpr[idx],
        'youden_j': j_scores[idx],
        'fpr': fpr[idx],
        'tpr': tpr[idx],
    }


def get_side_from_filename(filename):
    base = os.path.basename(filename)
    if '_L_' in base:
        return 'L'
    if '_R_' in base:
        return 'R'
    return None


def get_channel_columns(df):
    cols = []
    for i in range(1, 11):
        c = f'ch{i}'
        if c in df.columns:
            cols.append(c)
    return cols


def get_region_proxy_columns(df):
    return [c for c in df.columns if c.endswith('_median_proxy')]


def analyze_signal(signal, labels):
    valid_mask = np.isfinite(signal)
    signal = signal[valid_mask]
    labels_valid = labels[valid_mask]

    n_pos = int(np.sum(labels_valid))
    n_neg = len(labels_valid) - n_pos

    if len(signal) == 0:
        return None, "all values are NaN/inf"
    if n_pos == 0 or n_neg == 0:
        return None, "no positive or no negative labels after filtering"

    fpr, tpr, thresholds = roc_curve(labels_valid, signal)
    auc_score = auc(fpr, tpr)
    opt = find_youden_j_threshold(fpr, tpr, thresholds)

    return {
        'auc': auc_score,
        'threshold_youden_j': opt['threshold'],
        'sensitivity': opt['sensitivity'],
        'specificity': opt['specificity'],
        'youden_j': opt['youden_j'],
        'n_valid_bins': len(signal),
        'n_burst_bins': n_pos,
        'n_nonburst_bins': n_neg,
        # 'pct_burst': 100.0 * n_pos / len(signal),
    }, None


def build_output_paths(output_arg, input_folder):
    if output_arg is None:
        out_dir = os.path.join(input_folder, 'validate')
    elif output_arg.lower().endswith('.csv'):
        out_dir = os.path.dirname(output_arg) or '.'
    else:
        out_dir = output_arg

    os.makedirs(out_dir, exist_ok=True)

    return {
        'channel_L': os.path.join(out_dir, 'channel_L.csv'),
        'channel_R': os.path.join(out_dir, 'channel_R.csv'),
        'region_L': os.path.join(out_dir, 'region_L.csv'),
        'region_R': os.path.join(out_dir, 'region_R.csv'),
    }


def make_result_row(fname, side, signal_col, stats):
    return {
        'file': fname,
        'side': side,
        'signal': signal_col,
        'auc': round(stats['auc'], 6),
        'threshold_youden_j': f"{stats['threshold_youden_j']:.6e}",
        'sensitivity': round(stats['sensitivity'], 6),
        'specificity': round(stats['specificity'], 6),
        'youden_j': round(stats['youden_j'], 6),
        'n_valid_bins': int(stats['n_valid_bins']),
        'n_burst_bins': int(stats['n_burst_bins']),
        'n_nonburst_bins': int(stats['n_nonburst_bins']),
        # 'pct_burst': round(stats['pct_burst'], 6),
    }


def main():
    if len(sys.argv) < 4:
        print("Usage:")
        print("  python per_chan_roc.py <input_folder> <burst_left_csv> <burst_right_csv> [output_dir]")
        print("  Use 'none' if one side is missing.")
        sys.exit(1)

    input_folder = sys.argv[1]
    burst_L_csv = None if sys.argv[2].lower() == 'none' else sys.argv[2]
    burst_R_csv = None if sys.argv[3].lower() == 'none' else sys.argv[3]
    output_arg = sys.argv[4] if len(sys.argv) > 4 else None

    if not os.path.isdir(input_folder):
        print(f"ERROR: input folder does not exist: {input_folder}")
        sys.exit(1)

    output_paths = build_output_paths(output_arg, input_folder)

    print(f"Input folder: {input_folder}")
    print(f"Left burst CSV:  {burst_L_csv}")
    print(f"Right burst CSV: {burst_R_csv}")
    print(f"Padding: -{BURST_PAD_BEFORE*1000:.0f} ms / +{BURST_PAD_AFTER*1000:.0f} ms")
    print("Outputs:")
    for k, v in output_paths.items():
        print(f"  {k}: {v}")

    all_files = sorted(os.listdir(input_folder))
    csv_files = []

    for fname in all_files:
        full_path = os.path.join(input_folder, fname)
        fname_lower = fname.lower()

        if not os.path.isfile(full_path):
            continue
        if fname.startswith('._'):
            continue
        if fname_lower.endswith('.html'):
            continue
        if not fname_lower.endswith('.csv'):
            continue
        if fname_lower == 'hemisphere_neo_binned.csv':
            continue

        csv_files.append(full_path)

    if not csv_files:
        print("No eligible CSV files found.")
        sys.exit(1)

    print(f"Found {len(csv_files)} eligible CSV files.")

    channel_rows_L = []
    channel_rows_R = []
    region_rows_L = []
    region_rows_R = []

    for csv_path in csv_files:
        fname = os.path.basename(csv_path)
        side = get_side_from_filename(fname)

        if side is None:
            print(f"\nSkipping {fname}: could not infer side from filename.")
            continue

        burst_csv = burst_L_csv if side == 'L' else burst_R_csv
        if burst_csv is None or not os.path.isfile(burst_csv):
            print(f"\nSkipping {fname}: no burst CSV for side {side}.")
            continue

        print(f"\nProcessing: {fname}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  Skipping: could not read CSV ({e})")
            continue

        if 'time_s' not in df.columns:
            print("  Skipping: missing time_s column.")
            continue

        time_s = pd.to_numeric(df['time_s'], errors='coerce').values
        valid_time_mask = np.isfinite(time_s)
        if not np.all(valid_time_mask):
            df = df.loc[valid_time_mask].reset_index(drop=True)
            time_s = df['time_s'].values

        burst_starts, burst_ends = load_bursts(burst_csv)
        labels = create_burst_labels(time_s, burst_starts, burst_ends)

        channel_cols = get_channel_columns(df)
        region_cols = get_region_proxy_columns(df)

        if channel_cols:
            print(f"  Channels: {channel_cols}")
        if region_cols:
            print(f"  Region proxy: {region_cols}")

        for signal_col in channel_cols:
            signal = pd.to_numeric(df[signal_col], errors='coerce').values
            stats, err = analyze_signal(signal, labels)

            if err is not None:
                print(f"    {signal_col}: skipped ({err})")
                continue

            print(f"    {signal_col}: AUC={stats['auc']:.4f}")
            row = make_result_row(fname, side, signal_col, stats)
            if side == 'L':
                channel_rows_L.append(row)
            else:
                channel_rows_R.append(row)

        for signal_col in region_cols:
            signal = pd.to_numeric(df[signal_col], errors='coerce').values
            stats, err = analyze_signal(signal, labels)

            if err is not None:
                print(f"    {signal_col}: skipped ({err})")
                continue

            print(f"    {signal_col}: AUC={stats['auc']:.4f}")
            row = make_result_row(fname, side, signal_col, stats)
            if side == 'L':
                region_rows_L.append(row)
            else:
                region_rows_R.append(row)

    def write_rows(rows, path):
        if rows:
            df = pd.DataFrame(rows).sort_values(by=['file', 'signal']).reset_index(drop=True)
            df.to_csv(path, index=False)
            print(f"Wrote: {path}")
        else:
            print(f"No rows for: {path}")

    print()
    write_rows(channel_rows_L, output_paths['channel_L'])
    write_rows(channel_rows_R, output_paths['channel_R'])
    write_rows(region_rows_L, output_paths['region_L'])
    write_rows(region_rows_R, output_paths['region_R'])

    print("Done!")


if __name__ == '__main__':
    main()