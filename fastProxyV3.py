import numpy as np
import os
import re
import sys
from collections import defaultdict
import h5py
from scipy.signal import butter, lfilter, lfilter_zi
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
# Core DSP functions (bin-by-bin, causal)
# ──────────────────────────────────────────────

def make_highpass(fs, cutoff=350, order=3):
    """Create highpass filter coefficients and initial state."""
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype='high')
    zi = lfilter_zi(b, a)
    return b, a, zi


def process_single_file(filepath, cutoff=350, k=1, bin_ms=20):
    """
    Run the full pipeline on one .mat file (v7.3 HDF5), bin-by-bin.

    For each 20ms bin:
      1. Causal highpass filter (state carries across bins)
      2. NEO with k=1 (independent per bin, loses k samples at each edge)
      3. Mean of the NEO output for that bin

    Returns (array of per-bin means, fs).
    """
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
        with h5py.File(filepath, 'r') as f:
            keys = list(f.keys())
            shapes = {ky: f[ky].shape for ky in keys}
        raise ValueError(
            f"Could not auto-detect signal/fs in {filepath}. "
            f"Keys found: {shapes}"
        )

    bin_size = int(fs * bin_ms / 1000)
    n_bins = len(signal) // bin_size

    # --- Filter setup ---
    b, a, zi = make_highpass(fs, cutoff=cutoff)
    zi = zi * signal[0]

    bin_means = np.zeros(n_bins)

    for i in range(n_bins):
        chunk = signal[i * bin_size : (i + 1) * bin_size]
        filtered, zi = lfilter(b, a, chunk, zi=zi)

        x = filtered
        neo_out = x[k:-k] ** 2 - x[:-2*k] * x[2*k:]

        if len(neo_out) > 0:
            bin_means[i] = np.mean(neo_out)
        else:
            bin_means[i] = 0.0

    return bin_means, fs


# ──────────────────────────────────────────────
# File parsing
# ──────────────────────────────────────────────

def parse_mat_filename(filename):
    """
    Parse filenames like 'microGPi1_L_1.mat'
    Returns (region, side, channel) e.g. ('GPi1', 'L', 1)
    """
    name = os.path.splitext(filename)[0]
    match = re.match(r'^micro([A-Za-z0-9]+)_([LR])_(\d+)$', name)
    if not match:
        return None
    region, side, channel = match.groups()
    return region, side, int(channel)


def group_mat_files(folder):
    """
    Find all .mat files in a folder and group by (region, side).
    Returns dict: (region, side) -> sorted list of (channel, filepath)
    """
    groups = defaultdict(list)
    for fname in os.listdir(folder):
        if not fname.lower().endswith('.mat'):
            continue
        parsed = parse_mat_filename(fname)
        if parsed is None:
            print(f"    Skipping unrecognized file: {fname}")
            continue
        region, side, channel = parsed
        groups[(region, side)].append((channel, os.path.join(folder, fname)))

    for key in groups:
        groups[key].sort(key=lambda x: x[0])

    return groups


# ──────────────────────────────────────────────
# Drive discovery
# ──────────────────────────────────────────────

def discover_drive(drive_root):
    """
    Walk the drive to find all patient/period/micro folders.

    Expected structure:
      F:\\s523_202408\\Mat Data\\Voluntary\\micro\\period1\\
      F:\\s524_202410\\Mat Data\\Voluntary\\micro\\period2\\

    Returns list of dicts:
      [
        {'patient': 's523', 'period': 'period1', 'micro_path': '...'},
        ...
      ]
    """
    discoveries = []

    for entry in sorted(os.listdir(drive_root)):
        patient_dir = os.path.join(drive_root, entry)
        if not os.path.isdir(patient_dir):
            continue

        # Extract patient ID: first part before underscore (e.g. 's523' from 's523_202408')
        match = re.match(r'^(s\d+)', entry, re.IGNORECASE)
        if not match:
            continue
        patient_id = match.group(1)

        # Look for Mat Data/Voluntary/micro/
        micro_root = os.path.join(patient_dir, 'Mat Data', 'Voluntary', 'micro')
        if not os.path.isdir(micro_root):
            # Try case variations
            for alt in ['Mat data', 'mat data', 'MatData']:
                test = os.path.join(patient_dir, alt, 'Voluntary', 'micro')
                if os.path.isdir(test):
                    micro_root = test
                    break
            else:
                continue

        # Find all period folders
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
# CSV output for one region/side group
# ──────────────────────────────────────────────

def build_csv(group_key, channel_files, output_dir, cutoff=350, k=1, bin_ms=20):
    """
    Process all channels for one (region, side) group and write a CSV.
    """
    region, side = group_key
    side_label = 'Left' if side == 'L' else 'Right'
    print(f"    {region} {side_label} ({len(channel_files)} channels)")

    all_binned = []
    channels = []
    fs_val = None

    for channel, filepath in channel_files:
        print(f"      ch{channel}: {os.path.basename(filepath)}")
        binned, fs = process_single_file(filepath, cutoff=cutoff, k=k, bin_ms=bin_ms)
        all_binned.append(binned)
        channels.append(channel)
        fs_val = fs

    min_len = min(len(b) for b in all_binned)
    all_binned = [b[:min_len] for b in all_binned]

    matrix = np.column_stack(all_binned)
    time_s = np.arange(min_len) * (bin_ms / 1000.0)
    median_proxy = np.median(matrix, axis=1)

    # Header
    header_parts = ['time_s']
    for ch in channels:
        header_parts.append(f'ch{ch}')
    header_parts.append(f'{region}_{side}_median_proxy')
    header = ','.join(header_parts)

    out_data = np.column_stack([time_s, matrix, median_proxy])

    os.makedirs(output_dir, exist_ok=True)
    csv_name = f'{region}_{side}_neo_binned.csv'
    csv_path = os.path.join(output_dir, csv_name)
    np.savetxt(csv_path, out_data, delimiter=',', header=header, comments='', fmt='%.6e')
    print(f"      -> {csv_path}")

    return csv_path


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def plot_median_proxy(csv_path, t_start=0, t_end=150):
    """
    Plot the median proxy column from a CSV over a given time range.
    """
    with open(csv_path, 'r') as f:
        header = f.readline().strip().split(',')

    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    time_s = data[:, 0]

    proxy_col = -1
    proxy_label = header[proxy_col]

    mask = (time_s >= t_start) & (time_s <= t_end)
    t = time_s[mask]
    y = data[mask, proxy_col]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, y, linewidth=0.5, color='steelblue')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('NEO Median Proxy')
    ax.set_title(f'{proxy_label}  [{t_start}–{t_end} s]')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    fig.tight_layout()

    png_path = csv_path.replace('.csv', f'_plot_{t_start}-{t_end}s.png')
    fig.savefig(png_path, dpi=150)
    print(f"      -> Plot: {png_path}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage: python neo_pipeline.py <drive_root> <output_root> [--plot]")
        print()
        print("  drive_root  : root of the drive (e.g. F:\\)")
        print("  output_root : where to save outputs (e.g. F:\\fastProxy_outputs)")
        print("  --plot      : also save plots of median proxy (0-150s)")
        print()
        print("Output structure:")
        print("  <output_root>/<patient>/<period>/<region_side>/<region>_<side>_neo_binned.csv")
        sys.exit(1)

    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    do_plot = '--plot' in sys.argv

    drive_root = args[0]
    output_root = args[1]

    if not os.path.isdir(drive_root):
        print(f"Error: '{drive_root}' is not a valid directory.")
        sys.exit(1)

    # Discover all patients/periods
    print(f"Scanning {drive_root} ...")
    discoveries = discover_drive(drive_root)

    if not discoveries:
        print("No patient/period folders found.")
        sys.exit(1)

    print(f"Found {len(discoveries)} patient-period(s):\n")
    for d in discoveries:
        print(f"  {d['patient']} / {d['period']}  ->  {d['micro_path']}")

    print()

    # Process each patient/period
    total_csvs = 0

    for d in discoveries:
        patient = d['patient']
        period = d['period']
        micro_path = d['micro_path']

        print(f"── {patient} / {period} ──")

        groups = group_mat_files(micro_path)
        if not groups:
            print("    No matching .mat files, skipping.")
            continue

        for (region, side), channel_files in sorted(groups.items()):
            # Output: fastProxy_outputs/s523/period1/GPi1_L/
            region_dir = os.path.join(output_root, patient, period, f'{region}_{side}')

            csv_path = build_csv(
                (region, side), channel_files, region_dir
            )
            total_csvs += 1

            if do_plot:
                plot_median_proxy(csv_path, t_start=0, t_end=150)

    print(f"\nDone! {total_csvs} CSV(s) created under {output_root}")


if __name__ == '__main__':
    main()