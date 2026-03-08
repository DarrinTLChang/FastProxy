import numpy as np
import os
import re
import sys
import csv
from collections import defaultdict
import h5py
from scipy.signal import welch
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════
HARMONICS       = (60, 120, 180, 240, 300, 360)
BANDWIDTH       = 2.0       # Hz around each harmonic to measure peak
OUTLIER_THRESH  = 3.0       # STDs above group mean to flag


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
# PSD & noise measurement
# ──────────────────────────────────────────────

def compute_psd(signal, fs, nperseg=None):
    if nperseg is None:
        nperseg = min(len(signal), fs * 2)
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    return freqs, psd


def measure_harmonic_power(freqs, psd, harmonics=HARMONICS, bandwidth=BANDWIDTH):
    harm_powers = {}
    total = 0.0
    for harm_freq in harmonics:
        peak_mask = (freqs >= harm_freq - bandwidth / 2) & (freqs <= harm_freq + bandwidth / 2)
        if not np.any(peak_mask):
            continue
        peak_power = np.max(psd[peak_mask])
        harm_powers[harm_freq] = peak_power
        total += peak_power
    return harm_powers, total


# ──────────────────────────────────────────────
# Outlier detection (STD-based)
# ──────────────────────────────────────────────

def find_outlier_channels(channel_data, threshold_std=OUTLIER_THRESH):
    """
    A channel is an outlier if its total harmonic noise power is
    more than threshold_std standard deviations above the group mean.
    """
    totals = np.array([d['total_noise'] for d in channel_data])

    group_mean = np.mean(totals)
    group_std = np.std(totals)

    if group_std == 0:
        group_std = 1e-30

    results = []
    for i, d in enumerate(channel_data):
        deviation = (totals[i] - group_mean) / group_std
        is_outlier = deviation > threshold_std

        results.append({
            'channel': d['channel'],
            'total_noise': d['total_noise'],
            'harm_powers': d['harm_powers'],
            'deviation_std': round(deviation, 2),
            'is_outlier': is_outlier,
        })

    return results, group_mean, group_std


# ──────────────────────────────────────────────
# Analyze one folder
# ──────────────────────────────────────────────

def analyze_period(micro_path, output_dir, patient, period,
                   threshold_std=OUTLIER_THRESH, save_plots=True):
    """
    Run PSD analysis on all channels in a folder.
    Returns (summary dict, list of rows for master CSV).
    """
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

        channel_data = []
        all_freqs = []
        all_psds = []

        for channel, filepath in channel_files:
            signal, fs = load_signal(filepath)
            freqs, psd = compute_psd(signal, fs)
            harm_powers, total_noise = measure_harmonic_power(freqs, psd)

            channel_data.append({
                'channel': channel,
                'harm_powers': harm_powers,
                'total_noise': total_noise,
            })
            all_freqs.append(freqs)
            all_psds.append(psd)

        results, group_mean, group_std = find_outlier_channels(channel_data, threshold_std)

        outlier_channels = []
        clean_channels = []

        for r in results:
            ch = r['channel']
            dev = r['deviation_std']
            status = "OUTLIER" if r['is_outlier'] else "OK"

            harm_str = ", ".join(
                f"{freq}Hz: {power:.2e}"
                for freq, power in sorted(r['harm_powers'].items())
            )
            print(f"      ch{ch}: {status} ({dev:+.1f} STDs)  total={r['total_noise']:.2e}  [{harm_str}]")

            if r['is_outlier']:
                outlier_channels.append(ch)
            else:
                clean_channels.append(ch)

            # Build CSV row for every channel
            csv_rows.append({
                'patient': patient,
                'period': period,
                'region': region,
                'side': side_label,
                'channel': ch,
                'total_noise': r['total_noise'],
                'group_mean': group_mean,
                'group_std': group_std,
                'deviation_std': dev,
                'is_outlier': r['is_outlier'],
                **{f'harm_{freq}Hz': r['harm_powers'].get(freq, 0.0) for freq in HARMONICS},
            })

        print(f"      Group mean: {group_mean:.2e}, STD: {group_std:.2e}")
        if outlier_channels:
            print(f"      ** Outliers: ch{outlier_channels}")
        else:
            print(f"      All channels within normal range")

        summary[group_key] = {
            'results': results,
            'outlier_channels': outlier_channels,
            'clean_channels': clean_channels,
            'total': len(channel_files),
            'group_mean': group_mean,
            'group_std': group_std,
        }

        if save_plots:
            all_labels = [f'ch{r["channel"]}' for r in results]
            all_outlier = [r['is_outlier'] for r in results]

            plot_psd_grid(
                all_freqs, all_psds, all_labels, all_outlier,
                title=f'{patient} {period} {group_key} PSD — Individual Channels',
                output_path=os.path.join(output_dir, f'{group_key}_psd_grid.png')
            )

            plot_psd_overlay(
                all_freqs, all_psds, all_labels, all_outlier,
                title=f'{patient} {period} {group_key} PSD — Channel Comparison',
                output_path=os.path.join(output_dir, f'{group_key}_psd_overlay.png')
            )

            plot_noise_bar(
                results, group_mean, group_std, threshold_std,
                title=f'{patient} {period} {group_key} — Harmonic Noise per Channel',
                output_path=os.path.join(output_dir, f'{group_key}_noise_bars.png')
            )

    write_summary(summary, output_dir)
    return summary, csv_rows


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def plot_psd_grid(all_freqs, all_psds, all_labels, all_outlier, title, output_path,
                  freq_max=500):
    n = len(all_freqs)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        if idx >= n:
            ax.set_visible(False)
            continue

        freqs = all_freqs[idx]
        psd = all_psds[idx]
        label = all_labels[idx]
        outlier = all_outlier[idx]

        mask = freqs <= freq_max
        color = 'red' if outlier else 'steelblue'

        ax.semilogy(freqs[mask], psd[mask], linewidth=0.6, color=color)
        for harm in HARMONICS:
            if harm <= freq_max:
                ax.axvline(harm, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        status = "OUTLIER" if outlier else "OK"
        ax.set_title(f'{label} [{status}]', fontsize=10,
                     color='red' if outlier else 'black')
        ax.set_xlabel('Freq (Hz)', fontsize=8)
        ax.set_ylabel('PSD', fontsize=8)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"      -> Grid plot: {output_path}")


def plot_psd_overlay(all_freqs, all_psds, all_labels, all_outlier, title, output_path,
                     freq_max=500):
    fig, ax = plt.subplots(figsize=(12, 5))

    for i in range(len(all_freqs)):
        freqs = all_freqs[i]
        psd = all_psds[i]
        label = all_labels[i]
        outlier = all_outlier[i]

        mask = freqs <= freq_max
        color = 'red' if outlier else 'steelblue'
        alpha = 0.9 if outlier else 0.4
        lw = 1.0 if outlier else 0.5

        ax.semilogy(freqs[mask], psd[mask], linewidth=lw, color=color,
                     alpha=alpha, label=label)

    for harm in HARMONICS:
        if harm <= freq_max:
            ax.axvline(harm, color='orange', linestyle=':', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=7, ncol=4, loc='upper right')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"      -> Overlay: {output_path}")


def plot_noise_bar(results, group_mean, group_std, threshold_std, title, output_path):
    channels = [f'ch{r["channel"]}' for r in results]
    totals = [r['total_noise'] for r in results]
    outliers = [r['is_outlier'] for r in results]
    colors = ['red' if o else 'steelblue' for o in outliers]

    fig, ax = plt.subplots(figsize=(max(8, len(channels) * 0.8), 5))
    ax.bar(channels, totals, color=colors, alpha=0.8)

    thresh_val = group_mean + threshold_std * group_std
    ax.axhline(thresh_val, color='red', linestyle='--', linewidth=1,
               label=f'Outlier threshold ({threshold_std} STDs)')
    ax.axhline(group_mean, color='gray', linestyle='-', linewidth=1,
               label='Group mean')

    ax.set_ylabel('Total Harmonic Noise Power')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=8)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"      -> Bar chart: {output_path}")


# ──────────────────────────────────────────────
# Summary text report (per period)
# ──────────────────────────────────────────────

def write_summary(summary, output_dir):
    txt_path = os.path.join(output_dir, 'channel_quality_summary.txt')

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("Channel Quality Report — Outlier Detection (Relative Comparison)\n")
        f.write(f"Outlier threshold: {OUTLIER_THRESH} STDs above group mean\n")
        f.write(f"Harmonics checked: {HARMONICS}\n")
        f.write("=" * 60 + "\n\n")

        for group_key, info in sorted(summary.items()):
            outliers = info['outlier_channels']
            clean = info['clean_channels']
            total = info['total']
            gm = info['group_mean']
            gs = info['group_std']

            f.write(f"{group_key}: {len(outliers)}/{total} channels are outliers\n")
            f.write(f"  Group mean noise: {gm:.2e},  STD: {gs:.2e}\n")

            if outliers:
                f.write(f"  OUTLIER channels: {outliers}\n")
                f.write(f"  CLEAN channels:   {clean}\n")
            else:
                f.write(f"  All channels similar — no outliers\n")

            f.write("\n  Per-channel breakdown:\n")
            for r in info['results']:
                ch = r['channel']
                status = "OUTLIER" if r['is_outlier'] else "OK    "
                dev = r['deviation_std']
                total_n = r['total_noise']
                f.write(f"    ch{ch}: {status}  {dev:+.1f} STDs  total={total_n:.2e}\n")
            f.write("\n")

    print(f"      -> Summary: {txt_path}")


# ──────────────────────────────────────────────
# Master CSV (all patients, all periods)
# ──────────────────────────────────────────────

def write_master_csv(all_rows, output_path):
    """
    Write a CSV containing ONLY the outlier channels that should be removed.
    """
    outlier_rows = [r for r in all_rows if r['is_outlier']]

    if not outlier_rows:
        print(f"\nNo outlier channels found — no CSV written.")
        return

    fieldnames = [
        'patient', 'period', 'region', 'side', 'channel',
        'total_noise', 'group_mean', 'group_std', 'deviation_std',
    ]
    for freq in HARMONICS:
        fieldnames.append(f'harm_{freq}Hz')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in outlier_rows:
            writer.writerow(row)

    print(f"\nMaster CSV: {output_path}")
    print(f"  {len(outlier_rows)} outlier channels to remove")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Drive scan:  python psd_remove.py <drive_root> <output_root> [--threshold=2.0]")
        print("  One folder:  python psd_remove.py --folder <mat_folder> <output_folder> [--threshold=2.0]")
        print()
        print(f"  --threshold : STDs above group mean to flag (default: {OUTLIER_THRESH})")
        sys.exit(1)

    threshold_std = OUTLIER_THRESH
    for a in sys.argv[1:]:
        if a.startswith('--threshold='):
            threshold_std = float(a.split('=')[1])

    args = [a for a in sys.argv[1:] if not a.startswith('--')]

    # ── Single folder mode ──
    if '--folder' in sys.argv:
        if len(args) < 2:
            print("Usage: python psd_remove.py --folder <mat_folder> <output_folder>")
            sys.exit(1)

        mat_folder = args[0]
        output_folder = args[1]

        if not os.path.isdir(mat_folder):
            print(f"Error: '{mat_folder}' is not a valid directory.")
            sys.exit(1)

        print(f"Analyzing folder: {mat_folder}")
        print(f"Outlier threshold: {threshold_std} STDs\n")

        summary, csv_rows = analyze_period(
            mat_folder, output_folder,
            patient='unknown', period='unknown',
            threshold_std=threshold_std
        )

        master_csv_path = os.path.join(output_folder, 'channels_to_remove.csv')
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
    print(f"Outlier threshold: {threshold_std} STDs\n")
    discoveries = discover_drive(drive_root)

    if not discoveries:
        print("No patient/period folders found.")
        sys.exit(1)

    print(f"Found {len(discoveries)} patient-period(s):\n")
    for d in discoveries:
        print(f"  {d['patient']} / {d['period']}")
    print()

    # Collect all CSV rows across all patients/periods
    all_csv_rows = []

    for d in discoveries:
        patient = d['patient']
        period = d['period']
        micro_path = d['micro_path']

        print(f"── {patient} / {period} ──")

        report_dir = os.path.join(output_root, patient, period, '_quality')
        summary, csv_rows = analyze_period(
            micro_path, report_dir,
            patient=patient, period=period,
            threshold_std=threshold_std
        )
        all_csv_rows.extend(csv_rows)

        total_outliers = sum(len(v['outlier_channels']) for v in summary.values())
        total_ch = sum(v['total'] for v in summary.values())
        print(f"    => {total_outliers}/{total_ch} outliers\n")

    # Write master CSV with ALL channels from ALL patients
    master_csv_path = os.path.join(output_root, 'channels_to_remove.csv')
    write_master_csv(all_csv_rows, master_csv_path)

    print("\nDone!")


if __name__ == '__main__':
    main()