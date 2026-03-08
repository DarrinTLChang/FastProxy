import numpy as np
import os
import re
import sys
from collections import defaultdict
import h5py
from scipy.signal import welch
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
# File parsing (same as neo_pipeline)
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
            mat_files = [f for f in os.listdir(period_path) if f.lower().endswith('.mat')]
            if not mat_files:
                continue
            discoveries.append({
                'patient': patient_id,
                'period': period_entry.lower(),
                'micro_path': period_path,
            })
    return discoveries


# ──────────────────────────────────────────────
# Load raw signal from .mat
# ──────────────────────────────────────────────

def load_signal(filepath):
    """Load signal and fs from a v7.3 HDF5 .mat file."""
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
# PSD analysis for 60 Hz detection
# ──────────────────────────────────────────────

def compute_psd(signal, fs, nperseg=None):
    """Compute PSD using Welch's method."""
    if nperseg is None:
        nperseg = min(len(signal), fs * 2)  # 2-second windows
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    return freqs, psd


def check_60hz(freqs, psd, harmonics=(60, 120, 180), bandwidth=2.0, threshold_db=10):
    """
    Check for 60 Hz line noise (and harmonics) in the PSD.

    Compares the power at each harmonic frequency to the surrounding
    baseline. If the peak exceeds the baseline by more than threshold_db,
    it is flagged.

    Returns:
        results: list of dicts with info about each harmonic
        is_bad: True if any harmonic exceeds the threshold
    """
    results = []
    is_bad = False

    for harm_freq in harmonics:
        # Find the peak power in a narrow band around the harmonic
        peak_mask = (freqs >= harm_freq - bandwidth / 2) & (freqs <= harm_freq + bandwidth / 2)
        if not np.any(peak_mask):
            continue

        peak_power = np.max(psd[peak_mask])

        # Baseline: power in surrounding bands, excluding the peak band
        base_lo_mask = (freqs >= harm_freq - 20) & (freqs < harm_freq - bandwidth)
        base_hi_mask = (freqs > harm_freq + bandwidth) & (freqs <= harm_freq + 20)
        base_mask = base_lo_mask | base_hi_mask

        if not np.any(base_mask):
            continue

        baseline_power = np.median(psd[base_mask])

        # Compare in dB
        if baseline_power > 0:
            peak_db = 10 * np.log10(peak_power / baseline_power)
        else:
            peak_db = 0.0

        flagged = peak_db > threshold_db

        results.append({
            'frequency': harm_freq,
            'peak_power': peak_power,
            'baseline_power': baseline_power,
            'peak_above_baseline_db': round(peak_db, 1),
            'flagged': flagged,
        })

        if flagged:
            is_bad = True

    return results, is_bad


# ──────────────────────────────────────────────
# Analyze one period folder
# ──────────────────────────────────────────────

def analyze_period(micro_path, output_dir, threshold_db=10, save_plots=True):
    """
    Run PSD analysis on all channels in a period folder.
    Returns a summary dict of results.
    """
    groups = group_mat_files(micro_path)
    if not groups:
        return {}

    os.makedirs(output_dir, exist_ok=True)
    summary = {}

    for (region, side), channel_files in sorted(groups.items()):
        side_label = 'Left' if side == 'L' else 'Right'
        group_key = f'{region}_{side}'
        print(f"    {region} {side_label} ({len(channel_files)} channels)")

        group_results = []
        bad_channels = []

        # Collect PSD data for plotting
        all_freqs = []
        all_psds = []
        all_labels = []
        all_flagged = []

        for channel, filepath in channel_files:
            signal, fs = load_signal(filepath)
            freqs, psd = compute_psd(signal, fs)
            harm_results, is_bad = check_60hz(freqs, psd, threshold_db=threshold_db)

            status = "BAD" if is_bad else "OK"
            harm_str = ", ".join(
                f"{r['frequency']}Hz: {r['peak_above_baseline_db']:+.1f}dB"
                for r in harm_results
            )
            print(f"      ch{channel}: {status}  [{harm_str}]")

            group_results.append({
                'channel': channel,
                'is_bad': is_bad,
                'harmonics': harm_results,
            })

            if is_bad:
                bad_channels.append(channel)

            all_freqs.append(freqs)
            all_psds.append(psd)
            all_labels.append(f'ch{channel}')
            all_flagged.append(is_bad)

        summary[group_key] = {
            'channels': group_results,
            'bad_channels': bad_channels,
            'total': len(channel_files),
        }

        if save_plots:
            plot_psd_grid(
                all_freqs, all_psds, all_labels, all_flagged,
                title=f'{group_key} PSD Analysis',
                output_path=os.path.join(output_dir, f'{group_key}_psd.png')
            )

    # Write summary text file
    write_summary(summary, output_dir)

    return summary


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def plot_psd_grid(all_freqs, all_psds, all_labels, all_flagged, title, output_path,
                  freq_max=500):
    """
    Plot PSD for each channel in a grid, highlighting bad channels in red.
    """
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
        flagged = all_flagged[idx]

        mask = freqs <= freq_max
        color = 'red' if flagged else 'steelblue'

        ax.semilogy(freqs[mask], psd[mask], linewidth=0.6, color=color)

        # Mark 60 Hz harmonics
        for harm in [60, 120, 180]:
            ax.axvline(harm, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        status = "BAD" if flagged else "OK"
        ax.set_title(f'{label} [{status}]', fontsize=10,
                     color='red' if flagged else 'black')
        ax.set_xlabel('Freq (Hz)', fontsize=8)
        ax.set_ylabel('PSD', fontsize=8)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"      -> Plot: {output_path}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Summary report
# ──────────────────────────────────────────────

def write_summary(summary, output_dir):
    """Write a text summary of channel quality."""
    txt_path = os.path.join(output_dir, 'channel_quality_summary.txt')

    with open(txt_path, 'w') as f:
        f.write("Channel Quality Report — 60 Hz Noise Detection\n")
        f.write("=" * 50 + "\n\n")

        for group_key, info in sorted(summary.items()):
            bad = info['bad_channels']
            total = info['total']
            f.write(f"{group_key}: {len(bad)}/{total} channels flagged\n")

            if bad:
                f.write(f"  Bad channels: {bad}\n")
            else:
                f.write(f"  All channels clean\n")

            for ch_info in info['channels']:
                ch = ch_info['channel']
                status = "BAD" if ch_info['is_bad'] else "OK "
                harms = ch_info['harmonics']
                harm_str = ", ".join(
                    f"{r['frequency']}Hz: {r['peak_above_baseline_db']:+.1f}dB"
                    for r in harms
                )
                f.write(f"    ch{ch}: {status}  {harm_str}\n")
            f.write("\n")

    print(f"      -> Summary: {txt_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage: python channel_quality.py <drive_root> <output_root> [--threshold=10]")
        print()
        print("  drive_root  : root of the drive (e.g. F:\\)")
        print("  output_root : where to save reports (e.g. F:\\fastProxy_outputs)")
        print("  --threshold : dB above baseline to flag (default: 10)")
        sys.exit(1)

    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    drive_root = args[0]
    output_root = args[1]

    # Parse optional threshold
    threshold_db = 10
    for a in sys.argv[1:]:
        if a.startswith('--threshold='):
            threshold_db = float(a.split('=')[1])

    if not os.path.isdir(drive_root):
        print(f"Error: '{drive_root}' is not a valid directory.")
        sys.exit(1)

    print(f"Scanning {drive_root} ...")
    discoveries = discover_drive(drive_root)

    if not discoveries:
        print("No patient/period folders found.")
        sys.exit(1)

    print(f"Found {len(discoveries)} patient-period(s):\n")
    for d in discoveries:
        print(f"  {d['patient']} / {d['period']}")

    print()

    for d in discoveries:
        patient = d['patient']
        period = d['period']
        micro_path = d['micro_path']

        print(f"── {patient} / {period} ──")

        report_dir = os.path.join(output_root, patient, period, '_quality')
        summary = analyze_period(micro_path, report_dir, threshold_db=threshold_db)

        # Print quick summary
        total_bad = sum(len(v['bad_channels']) for v in summary.values())
        total_ch = sum(v['total'] for v in summary.values())
        print(f"    => {total_bad}/{total_ch} channels flagged\n")

    print("Done!")


if __name__ == '__main__':
    main()