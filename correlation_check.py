import numpy as np
import os
import re
import sys
import csv
from collections import defaultdict
import h5py
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════
OUTLIER_THRESH  = 3.0       # MADs below group median correlation to flag
HIGHPASS_CUTOFF = 350       # Hz
HIGHPASS_ORDER  = 2


# ──────────────────────────────────────────────
# Biquad highpass (same as neo_pipeline)
# ──────────────────────────────────────────────

def compute_biquad_highpass_coeffs(fs, cutoff):
    wc = 2 * np.pi * cutoff
    T = 1 / fs
    wc_warped = (2 / T) * np.tan(wc * T / 2)
    K = wc_warped * T / 2
    denom = K**2 + np.sqrt(2) * K + 1
    b0 = 1.0 / denom
    b1 = -2.0 / denom
    b2 = 1.0 / denom
    a1 = (2 * K**2 - 2) / denom
    a2 = (K**2 - np.sqrt(2) * K + 1) / denom
    return b0, b1, b2, a1, a2


def highpass_filter(signal, fs, cutoff=HIGHPASS_CUTOFF, order=HIGHPASS_ORDER):
    """
    Apply biquad highpass filter (causal, single pass).
    Cascades order//2 biquad sections.
    """
    n_sections = order // 2
    b0, b1, b2, a1, a2 = compute_biquad_highpass_coeffs(fs, cutoff)

    out = signal.copy()
    for _ in range(n_sections):
        x1 = x2 = y1 = y2 = 0.0
        filtered = np.empty(len(out))
        for n in range(len(out)):
            x = out[n]
            y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2 = x1
            x1 = x
            y2 = y1
            y1 = y
            filtered[n] = y
        out = filtered

    return out


# ──────────────────────────────────────────────
# File parsing
# ──────────────────────────────────────────────

def parse_mat_filename(filename):
    name = os.path.splitext(filename)[0]
    match = re.match(r'^micro([A-Za-z0-9]+)_([LR])_(\d+)$', name)
    if not match:
        return None
    region, side, channel = match.groups()
    return region, side, int(channel)


def group_mat_files(folder):
    groups = defaultdict(list)
    for fname in os.listdir(folder):
        if not fname.lower().endswith('.mat'):
            continue
        parsed = parse_mat_filename(fname)
        if parsed is None:
            continue
        region, side, channel = parsed
        groups[(region, side)].append((channel, os.path.join(folder, fname)))
    for key in groups:
        groups[key].sort(key=lambda x: x[0])
    return groups


def discover_drive(drive_root):
    discoveries = []
    for entry in sorted(os.listdir(drive_root)):
        patient_dir = os.path.join(drive_root, entry)
        if not os.path.isdir(patient_dir):
            continue
        match = re.match(r'^(s\d+)', entry, re.IGNORECASE)
        if not match:
            continue
        patient_id = match.group(1)

        micro_root = os.path.join(patient_dir, 'Mat Data', 'Voluntary', 'micro')
        if not os.path.isdir(micro_root):
            for alt in ['Mat data', 'mat data', 'MatData']:
                test = os.path.join(patient_dir, alt, 'Voluntary', 'micro')
                if os.path.isdir(test):
                    micro_root = test
                    break
            else:
                continue

        for period_entry in sorted(os.listdir(micro_root)):
            period_path = os.path.join(micro_root, period_entry)
            if not os.path.isdir(period_path):
                continue
            if not re.match(r'^period\d+$', period_entry, re.IGNORECASE):
                continue

            mat_folder = period_path
            raw_folder = os.path.join(period_path, 'Raw')
            if os.path.isdir(raw_folder):
                test_mats = [f for f in os.listdir(raw_folder) if f.lower().endswith('.mat')]
                if test_mats:
                    mat_folder = raw_folder

            mat_files = [f for f in os.listdir(mat_folder) if f.lower().endswith('.mat')]
            if not mat_files:
                continue

            discoveries.append({
                'patient': patient_id,
                'period': period_entry.lower(),
                'micro_path': mat_folder,
            })
    return discoveries


# ──────────────────────────────────────────────
# Load signal
# ──────────────────────────────────────────────

def load_signal(filepath):
    signal = None
    fs = None
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            dataset = f[key]
            if dataset.ndim == 0 or (dataset.ndim >= 1 and dataset.size == 1):
                if fs is None:
                    fs = int(np.squeeze(dataset[()]))
            elif dataset.ndim >= 1 and dataset.size > 1000:
                if signal is None:
                    signal = np.squeeze(dataset[()]).astype(float)
    if signal is None or fs is None:
        raise ValueError(f"Could not load signal/fs from {filepath}")
    return signal, fs


# ──────────────────────────────────────────────
# Correlation analysis
# ──────────────────────────────────────────────

def compute_correlation_matrix(signals):
    """
    Compute pairwise Pearson correlation between all channels.

    signals: list of 1D arrays (all same length)
    Returns: (n_channels x n_channels) correlation matrix
    """
    # Stack into (n_channels, n_samples) matrix
    matrix = np.vstack(signals)
    # np.corrcoef gives the full correlation matrix
    return np.corrcoef(matrix)


def find_correlation_outliers(corr_matrix, channels, threshold_mads=OUTLIER_THRESH):
    """
    For each channel, compute its average correlation with all OTHER channels.
    Flag channels whose average correlation is significantly lower than peers.

    Uses median + MAD (looking for LOW outliers only — channels that
    don't correlate with their neighbors).
    """
    n = len(channels)
    avg_corrs = np.zeros(n)

    for i in range(n):
        # Average correlation with all other channels (exclude self = 1.0)
        others = [corr_matrix[i, j] for j in range(n) if j != i]
        avg_corrs[i] = np.mean(others)

    group_median = np.median(avg_corrs)
    mad = np.median(np.abs(avg_corrs - group_median))

    if mad == 0:
        mad = 1e-30

    results = []
    for i in range(n):
        deviation = (avg_corrs[i] - group_median) / mad
        # Flag if significantly BELOW median (low correlation = bad channel)
        is_outlier = deviation < -threshold_mads

        results.append({
            'channel': channels[i],
            'avg_correlation': round(avg_corrs[i], 4),
            'deviation_mads': round(deviation, 2),
            'is_outlier': is_outlier,
        })

    return results, avg_corrs, group_median, mad


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def plot_correlation_matrix(corr_matrix, channels, title, output_path):
    """Plot the pairwise correlation matrix as a heatmap."""
    labels = [f'ch{ch}' for ch in channels]
    n = len(channels)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.6), max(5, n * 0.5)))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8, rotation=45)
    ax.set_yticklabels(labels, fontsize=8)

    # Annotate each cell
    for i in range(n):
        for j in range(n):
            val = corr_matrix[i, j]
            color = 'white' if abs(val) > 0.7 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color)

    fig.colorbar(im, ax=ax, label='Pearson r')
    ax.set_title(title, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"      -> Corr matrix: {output_path}")


def plot_avg_correlation_bar(results, group_median, mad, threshold_mads,
                              title, output_path):
    """Bar chart of average correlation per channel."""
    channels = [f'ch{r["channel"]}' for r in results]
    avg_corrs = [r['avg_correlation'] for r in results]
    outliers = [r['is_outlier'] for r in results]
    colors = ['red' if o else 'steelblue' for o in outliers]

    fig, ax = plt.subplots(figsize=(max(8, len(channels) * 0.8), 5))
    ax.bar(channels, avg_corrs, color=colors, alpha=0.8)

    # Threshold line (below median)
    thresh_val = group_median - threshold_mads * mad
    ax.axhline(thresh_val, color='red', linestyle='--', linewidth=1,
               label=f'Outlier threshold (-{threshold_mads} MADs)')
    ax.axhline(group_median, color='gray', linestyle='-', linewidth=1,
               label='Group median')

    ax.set_ylabel('Avg Correlation with Other Channels')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"      -> Bar chart: {output_path}")


# ──────────────────────────────────────────────
# Analyze one period folder
# ──────────────────────────────────────────────

def analyze_period(micro_path, output_dir, patient, period,
                   threshold_mads=OUTLIER_THRESH, save_plots=True):
    groups = group_mat_files(micro_path)
    if not groups:
        return {}, []

    os.makedirs(output_dir, exist_ok=True)
    summary = {}
    csv_rows = []

    for (region, side), channel_files in sorted(groups.items()):
        side_label = 'Left' if side == 'L' else 'Right'
        group_key = f'{region}_{side}'
        print(f"    {region} {side_label} ({len(channel_files)} channels)")

        # Load all signals
        channels = []
        signals = []
        fs_val = None
        for channel, filepath in channel_files:
            signal, fs = load_signal(filepath)
            channels.append(channel)
            signals.append(signal)
            fs_val = fs

        # Trim all to same length
        min_len = min(len(s) for s in signals)
        signals = [s[:min_len] for s in signals]

        # Highpass filter before correlation (spike-band comparison)
        print(f"      Highpass filtering at {HIGHPASS_CUTOFF}Hz (order {HIGHPASS_ORDER})...")
        signals = [highpass_filter(s, fs_val) for s in signals]

        # Compute correlation matrix
        corr_matrix = compute_correlation_matrix(signals)

        # Find outliers
        results, avg_corrs, group_median, mad = \
            find_correlation_outliers(corr_matrix, channels, threshold_mads)

        outlier_channels = []
        clean_channels = []

        for r in results:
            ch = r['channel']
            dev = r['deviation_mads']
            avg_c = r['avg_correlation']
            status = "OUTLIER" if r['is_outlier'] else "OK"

            print(f"      ch{ch}: {status} ({dev:+.1f} MADs)  avg_corr={avg_c:.4f}")

            if r['is_outlier']:
                outlier_channels.append(ch)
            else:
                clean_channels.append(ch)

            csv_rows.append({
                'patient': patient,
                'period': period,
                'region': region,
                'side': side_label,
                'channel': ch,
                'avg_correlation': avg_c,
                'group_median': round(group_median, 4),
                'mad': round(mad, 4),
                'deviation_mads': dev,
                'is_outlier': r['is_outlier'],
            })

        print(f"      Group median corr: {group_median:.4f}, MAD: {mad:.4f}")
        if outlier_channels:
            print(f"      ** Outliers: ch{outlier_channels}")
        else:
            print(f"      All channels well-correlated")

        summary[group_key] = {
            'results': results,
            'outlier_channels': outlier_channels,
            'clean_channels': clean_channels,
            'total': len(channel_files),
            'group_median': group_median,
            'mad': mad,
        }

        if save_plots:
            plot_correlation_matrix(
                corr_matrix, channels,
                title=f'{patient} {period} {group_key} — Pairwise Correlation',
                output_path=os.path.join(output_dir, f'{group_key}_corr_matrix.png')
            )

            plot_avg_correlation_bar(
                results, group_median, mad, threshold_mads,
                title=f'{patient} {period} {group_key} — Avg Correlation per Channel',
                output_path=os.path.join(output_dir, f'{group_key}_corr_bars.png')
            )

    write_summary(summary, output_dir, csv_rows)
    return summary, csv_rows


# ──────────────────────────────────────────────
# Per-period summary CSV (all channels)
# ──────────────────────────────────────────────

def write_summary(summary, output_dir, period_csv_rows):
    """Write a CSV with ALL channels for this period."""
    csv_path = os.path.join(output_dir, 'correlation_check_summary.csv')

    fieldnames = [
        'patient', 'period', 'region', 'side', 'channel',
        'status', 'avg_correlation', 'group_median', 'mad', 'deviation_mads',
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in period_csv_rows:
            row_out = dict(row)
            row_out['status'] = 'OUTLIER' if row['is_outlier'] else 'OK'
            writer.writerow(row_out)

    print(f"      -> Summary CSV: {csv_path}")


# ──────────────────────────────────────────────
# Master CSV (outliers only)
# ──────────────────────────────────────────────

def write_master_csv(all_rows, output_path):
    outlier_rows = [r for r in all_rows if r['is_outlier']]

    if not outlier_rows:
        print(f"\nNo correlation outliers found — no CSV written.")
        return

    fieldnames = [
        'patient', 'period', 'region', 'side', 'channel',
        'avg_correlation', 'group_median', 'mad', 'deviation_mads',
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in outlier_rows:
            writer.writerow(row)

    print(f"\nMaster CSV: {output_path}")
    print(f"  {len(outlier_rows)} channels with low inter-channel correlation")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Drive scan:  python correlation_check.py <drive_root> <output_root> [--threshold=3.0]")
        print("  One folder:  python correlation_check.py --folder <mat_folder> <output_folder> [--threshold=3.0]")
        print()
        print(f"  --threshold : MADs below median correlation to flag (default: {OUTLIER_THRESH})")
        sys.exit(1)

    threshold_mads = OUTLIER_THRESH
    for a in sys.argv[1:]:
        if a.startswith('--threshold='):
            threshold_mads = float(a.split('=')[1])

    args = [a for a in sys.argv[1:] if not a.startswith('--')]

    # ── Single folder mode ──
    if '--folder' in sys.argv:
        if len(args) < 2:
            print("Usage: python correlation_check.py --folder <mat_folder> <output_folder>")
            sys.exit(1)

        mat_folder = args[0]
        output_folder = args[1]

        if not os.path.isdir(mat_folder):
            print(f"Error: '{mat_folder}' is not a valid directory.")
            sys.exit(1)

        print(f"Analyzing folder: {mat_folder}")
        print(f"Outlier threshold: -{threshold_mads} MADs\n")

        summary, csv_rows = analyze_period(
            mat_folder, output_folder,
            patient='unknown', period='unknown',
            threshold_mads=threshold_mads
        )

        master_csv_path = os.path.join(output_folder, 'channels_low_correlation.csv')
        write_master_csv(csv_rows, master_csv_path)
        print("Done!")
        return

    # ── Drive scan mode ──
    drive_root = args[0]
    output_root = args[1]

    if not os.path.isdir(drive_root):
        print(f"Error: '{drive_root}' is not a valid directory.")
        sys.exit(1)

    print(f"Scanning {drive_root} ...")
    print(f"Outlier threshold: -{threshold_mads} MADs\n")
    discoveries = discover_drive(drive_root)

    if not discoveries:
        print("No patient/period folders found.")
        sys.exit(1)

    print(f"Found {len(discoveries)} patient-period(s):\n")
    for d in discoveries:
        print(f"  {d['patient']} / {d['period']}")
    print()

    all_csv_rows = []

    for d in discoveries:
        patient = d['patient']
        period = d['period']
        micro_path = d['micro_path']

        print(f"── {patient} / {period} ──")

        report_dir = os.path.join(output_root, patient, period, '_correlation')
        summary, csv_rows = analyze_period(
            micro_path, report_dir,
            patient=patient, period=period,
            threshold_mads=threshold_mads
        )
        all_csv_rows.extend(csv_rows)

        total_outliers = sum(len(v['outlier_channels']) for v in summary.values())
        total_ch = sum(v['total'] for v in summary.values())
        print(f"    => {total_outliers}/{total_ch} outliers\n")

    master_csv_path = os.path.join(output_root, 'channels_low_correlation.csv')
    write_master_csv(all_csv_rows, master_csv_path)

    print("\nDone!")


if __name__ == '__main__':
    main()