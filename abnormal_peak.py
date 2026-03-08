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
PEAK_THRESH_STD = 50         # flag if any sample exceeds this many STDs above mean amplitude


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
# Amplitude analysis
# ──────────────────────────────────────────────

def analyze_amplitude(signal, fs, threshold_std=PEAK_THRESH_STD):
    """
    Analyze the raw signal amplitude.

    Returns dict with:
      - mean, std of the absolute amplitude
      - max absolute amplitude
      - threshold value
      - number of samples exceeding threshold
      - percentage of samples exceeding threshold
      - times (in seconds) of the peaks
      - has_abnormal: True if any peaks found
    """
    amp = np.abs(signal)
    amp_mean = np.mean(amp)
    amp_std = np.std(amp)
    amp_max = np.max(amp)

    threshold = amp_mean + threshold_std * amp_std

    # Find samples exceeding threshold
    peak_mask = amp > threshold
    n_peaks = np.sum(peak_mask)
    pct_peaks = 100.0 * n_peaks / len(signal)

    # Get times of peaks
    peak_indices = np.where(peak_mask)[0]
    peak_times = peak_indices / fs

    # Group nearby peaks into events (within 10ms of each other)
    events = []
    if len(peak_indices) > 0:
        gap_samples = int(fs * 0.01)  # 10ms
        event_start = peak_indices[0]
        event_end = peak_indices[0]

        for idx in peak_indices[1:]:
            if idx - event_end <= gap_samples:
                event_end = idx
            else:
                events.append({
                    'start_s': event_start / fs,
                    'end_s': event_end / fs,
                    'duration_ms': (event_end - event_start) / fs * 1000,
                    'max_amp': np.max(amp[event_start:event_end + 1]),
                    'max_std': np.max(amp[event_start:event_end + 1]) / amp_std if amp_std > 0 else 0,
                })
                event_start = idx
                event_end = idx

        # Last event
        events.append({
            'start_s': event_start / fs,
            'end_s': event_end / fs,
            'duration_ms': (event_end - event_start) / fs * 1000,
            'max_amp': np.max(amp[event_start:event_end + 1]),
            'max_std': np.max(amp[event_start:event_end + 1]) / amp_std if amp_std > 0 else 0,
        })

    return {
        'amp_mean': amp_mean,
        'amp_std': amp_std,
        'amp_max': amp_max,
        'threshold': threshold,
        'n_peaks': n_peaks,
        'pct_peaks': pct_peaks,
        'n_events': len(events),
        'events': events,
        'has_abnormal': n_peaks > 0,
        'max_std_deviation': amp_max / amp_std if amp_std > 0 else 0,
    }


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def plot_amplitude_overview(signal, fs, analysis, label, output_path):
    """
    Plot the full raw signal with threshold lines and peak markers.
    """
    t = np.arange(len(signal)) / fs
    amp = np.abs(signal)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, amp, linewidth=0.2, color='steelblue', alpha=0.6)

    # Threshold line
    ax.axhline(analysis['threshold'], color='red', linestyle='--', linewidth=0.8,
               label=f'Threshold ({PEAK_THRESH_STD} STDs)')
    ax.axhline(analysis['amp_mean'], color='gray', linestyle='-', linewidth=0.5,
               label='Mean')

    status = "ABNORMAL" if analysis['has_abnormal'] else "CLEAN"
    ax.set_title(f'{label} [{status}] — {analysis["n_events"]} events, '
                 f'max={analysis["max_std_deviation"]:.1f} STDs',
                 fontweight='bold',
                 color='red' if analysis['has_abnormal'] else 'black')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|Amplitude|')
    ax.legend(fontsize=8, loc='upper right')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"      -> Plot: {output_path}")


# ──────────────────────────────────────────────
# Analyze one period folder
# ──────────────────────────────────────────────

def analyze_period(micro_path, output_dir, patient, period,
                   threshold_std=PEAK_THRESH_STD, save_plots=True):
    groups = group_mat_files(micro_path)
    if not groups:
        return [], []

    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    csv_rows = []

    for (region, side), channel_files in sorted(groups.items()):
        side_label = 'Left' if side == 'L' else 'Right'
        group_key = f'{region}_{side}'
        print(f"    {region} {side_label} ({len(channel_files)} channels)")

        for channel, filepath in channel_files:
            signal, fs = load_signal(filepath)
            analysis = analyze_amplitude(signal, fs, threshold_std=threshold_std)

            status = "ABNORMAL" if analysis['has_abnormal'] else "OK"
            print(f"      ch{channel}: {status}  "
                  f"mean={analysis['amp_mean']:.2e}  "
                  f"std={analysis['amp_std']:.2e}  "
                  f"max={analysis['max_std_deviation']:.1f} STDs  "
                  f"events={analysis['n_events']}")

            all_results.append({
                'patient': patient,
                'period': period,
                'region': region,
                'side': side_label,
                'channel': channel,
                'analysis': analysis,
            })

            # Only add to CSV if abnormal
            if analysis['has_abnormal']:
                csv_rows.append({
                    'patient': patient,
                    'period': period,
                    'region': region,
                    'side': side_label,
                    'channel': channel,
                    'amp_mean': analysis['amp_mean'],
                    'amp_std': analysis['amp_std'],
                    'amp_max': analysis['amp_max'],
                    'max_std_deviation': round(analysis['max_std_deviation'], 2),
                    'n_events': analysis['n_events'],
                    'n_peak_samples': analysis['n_peaks'],
                    'pct_peak_samples': round(analysis['pct_peaks'], 4),
                })

            if save_plots:
                plot_amplitude_overview(
                    signal, fs, analysis,
                    label=f'{patient} {period} {group_key} ch{channel}',
                    output_path=os.path.join(output_dir, f'{group_key}_ch{channel}_amplitude.png')
                )

    # Write per-period summary
    write_period_summary(all_results, output_dir)

    return all_results, csv_rows


# ──────────────────────────────────────────────
# Per-period summary
# ──────────────────────────────────────────────

def write_period_summary(all_results, output_dir):
    """Write a CSV summary of ALL channels for this period."""
    csv_path = os.path.join(output_dir, 'amplitude_check_summary.csv')

    fieldnames = [
        'patient', 'period', 'region', 'side', 'channel',
        'status', 'amp_mean', 'amp_std', 'amp_max',
        'max_std_deviation', 'n_events', 'n_peak_samples', 'pct_peak_samples',
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in all_results:
            a = r['analysis']
            writer.writerow({
                'patient': r['patient'],
                'period': r['period'],
                'region': r['region'],
                'side': r['side'],
                'channel': r['channel'],
                'status': 'ABNORMAL' if a['has_abnormal'] else 'OK',
                'amp_mean': f"{a['amp_mean']:.6e}",
                'amp_std': f"{a['amp_std']:.6e}",
                'amp_max': f"{a['amp_max']:.6e}",
                'max_std_deviation': round(a['max_std_deviation'], 2),
                'n_events': a['n_events'],
                'n_peak_samples': a['n_peaks'],
                'pct_peak_samples': round(a['pct_peaks'], 4),
            })

    print(f"      -> Summary CSV: {csv_path}")


# ──────────────────────────────────────────────
# Master CSV (outliers only)
# ──────────────────────────────────────────────

def write_master_csv(all_rows, output_path):
    """Write CSV with ONLY channels that have abnormal peaks."""
    if not all_rows:
        print(f"\nNo abnormal channels found — no CSV written.")
        return

    fieldnames = [
        'patient', 'period', 'region', 'side', 'channel',
        'amp_mean', 'amp_std', 'amp_max', 'max_std_deviation',
        'n_events', 'n_peak_samples', 'pct_peak_samples',
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"\nMaster CSV: {output_path}")
    print(f"  {len(all_rows)} channels with abnormal peaks")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Drive scan:  python amplitude_check.py <drive_root> <output_root> [--threshold=8]")
        print("  One folder:  python amplitude_check.py --folder <mat_folder> <output_folder> [--threshold=8]")
        print()
        print(f"  --threshold : STDs above mean amplitude to flag (default: {PEAK_THRESH_STD})")
        sys.exit(1)

    threshold_std = PEAK_THRESH_STD
    for a in sys.argv[1:]:
        if a.startswith('--threshold='):
            threshold_std = float(a.split('=')[1])

    args = [a for a in sys.argv[1:] if not a.startswith('--')]

    # ── Single folder mode ──
    if '--folder' in sys.argv:
        if len(args) < 2:
            print("Usage: python amplitude_check.py --folder <mat_folder> <output_folder>")
            sys.exit(1)

        mat_folder = args[0]
        output_folder = args[1]

        if not os.path.isdir(mat_folder):
            print(f"Error: '{mat_folder}' is not a valid directory.")
            sys.exit(1)

        print(f"Analyzing folder: {mat_folder}")
        print(f"Peak threshold: {threshold_std} STDs\n")

        all_results, csv_rows = analyze_period(
            mat_folder, output_folder,
            patient='unknown', period='unknown',
            threshold_std=threshold_std
        )

        master_csv_path = os.path.join(output_folder, 'abnormal_amplitude_channels.csv')
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
    print(f"Peak threshold: {threshold_std} STDs\n")
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

        report_dir = os.path.join(output_root, patient, period, '_amplitude')
        all_results, csv_rows = analyze_period(
            micro_path, report_dir,
            patient=patient, period=period,
            threshold_std=threshold_std
        )
        all_csv_rows.extend(csv_rows)

        n_abnormal = sum(1 for r in all_results if r['analysis']['has_abnormal'])
        print(f"    => {n_abnormal}/{len(all_results)} channels with abnormal peaks\n")

    master_csv_path = os.path.join(output_root, 'abnormal_amplitude_channels.csv')
    write_master_csv(all_csv_rows, master_csv_path)

    print("\nDone!")


if __name__ == '__main__':
    main()