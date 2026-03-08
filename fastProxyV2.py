import numpy as np
import os
import re
import sys
from collections import defaultdict
import h5py
from scipy.signal import butter, lfilter, lfilter_zi


# ──────────────────────────────────────────────
# Core DSP functions (bin-by-bin, causal)
# ──────────────────────────────────────────────
# Emulates a real-time closed-loop system that receives
# 20ms packets of data and must produce one output value
# per packet: highpass -> NEO (k=1) -> mean.

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
    zi = zi * signal[0]  # scale to reduce startup transient

    # --- NEO has no overlap between bins ---
    # Each bin is independent. NEO with k=1 loses the first and last
    # sample of each bin (no context carried across bins).

    bin_means = np.zeros(n_bins)

    for i in range(n_bins):
        # 1. Extract this bin's raw samples
        chunk = signal[i * bin_size : (i + 1) * bin_size]

        # 2. Causal highpass filter (state persists)
        filtered, zi = lfilter(b, a, chunk, zi=zi)

        # 3. NEO on this bin only (loses k samples at each edge)
        x = filtered
        neo_out = x[k:-k] ** 2 - x[:-2*k] * x[2*k:]

        # 4. Mean of this bin's NEO output
        if len(neo_out) > 0:
            bin_means[i] = np.mean(neo_out)
        else:
            bin_means[i] = 0.0

    return bin_means, fs


# ──────────────────────────────────────────────
# File discovery & grouping
# ──────────────────────────────────────────────

def parse_filename(filename):
    """
    Parse filenames like 'microVIM_L_1.mat'
    Returns (region, side, channel) e.g. ('VIM', 'L', 1)
    """
    name = os.path.splitext(filename)[0]
    match = re.match(r'^micro([A-Za-z]+)_([LR])_(\d+)$', name)
    if not match:
        return None
    region, side, channel = match.groups()
    return region, side, int(channel)


def discover_and_group(folder):
    """
    Find all .mat files in folder and group by (region, side).
    Returns dict: (region, side) -> sorted list of (channel, filepath)
    """
    groups = defaultdict(list)
    for fname in os.listdir(folder):
        if not fname.lower().endswith('.mat'):
            continue
        parsed = parse_filename(fname)
        if parsed is None:
            print(f"  Skipping unrecognized file: {fname}")
            continue
        region, side, channel = parsed
        groups[(region, side)].append((channel, os.path.join(folder, fname)))

    for key in groups:
        groups[key].sort(key=lambda x: x[0])

    return groups


# ──────────────────────────────────────────────
# CSV output
# ──────────────────────────────────────────────

def build_csv(group_key, channel_files, output_dir, cutoff=350, k=1, bin_ms=20):
    """
    Process all channels for one (region, side) group and write a CSV.
    """
    region, side = group_key
    side_label = 'Left' if side == 'L' else 'Right'
    print(f"\nProcessing {region} {side_label} ({len(channel_files)} channels)...")

    all_binned = []
    channels = []
    fs_val = None

    for channel, filepath in channel_files:
        print(f"  Channel {channel}: {os.path.basename(filepath)}")
        binned, fs = process_single_file(filepath, cutoff=cutoff, k=k, bin_ms=bin_ms)
        all_binned.append(binned)
        channels.append(channel)
        fs_val = fs

    # Trim all to the shortest length
    min_len = min(len(b) for b in all_binned)
    all_binned = [b[:min_len] for b in all_binned]

    # Stack into (n_bins, n_channels) array
    matrix = np.column_stack(all_binned)

    # Time column in seconds
    time_s = np.arange(min_len) * (bin_ms / 1000.0)

    # Median across channels at each time bin
    median_proxy = np.median(matrix, axis=1)

    # Build header
    header_parts = ['time_s']
    for ch in channels:
        header_parts.append(f'ch{ch}')
    header_parts.append(f'{region}_{side}_median_proxy')
    header = ','.join(header_parts)

    # Combine all columns
    out_data = np.column_stack([time_s, matrix, median_proxy])

    # Write CSV
    csv_name = f'micro{region}_{side}_neo_binned.csv'
    csv_path = os.path.join(output_dir, csv_name)
    np.savetxt(csv_path, out_data, delimiter=',', header=header, comments='', fmt='%.6e')
    print(f"  -> Saved: {csv_path}")
    print(f"     {min_len} bins, {len(channels)} channels, fs={fs_val} Hz")

    return csv_path


import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def plot_median_proxy(csv_path, t_start=0, t_end=100):
    """
    Plot the median proxy column from a CSV over a given time range.
    Saves a .png next to the CSV.
    """
    # Load CSV, first row is header
    with open(csv_path, 'r') as f:
        header = f.readline().strip().split(',')

    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    time_s = data[:, 0]

    # Find the median proxy column (last column, name ends with _median_proxy)
    proxy_col = -1
    proxy_label = header[proxy_col]

    # Mask to the requested time range
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
    print(f"  -> Plot saved: {png_path}")
    plt.show()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python neo_pipeline.py <input_folder> [output_folder] [--plot]")
        print("  input_folder : path to folder with .mat files")
        print("  output_folder: (optional) where to save CSVs, defaults to input_folder")
        print("  --plot       : also plot the median proxy (0-100s) for each group")
        sys.exit(1)

    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    do_plot = '--plot' in sys.argv

    input_folder = args[0]
    output_folder = args[1] if len(args) > 1 else input_folder

    if not os.path.isdir(input_folder):
        print(f"Error: '{input_folder}' is not a valid directory.")
        sys.exit(1)

    os.makedirs(output_folder, exist_ok=True)

    # Discover and group files
    groups = discover_and_group(input_folder)

    if not groups:
        print("No matching .mat files found.")
        sys.exit(1)

    print(f"Found {sum(len(v) for v in groups.values())} files "
          f"in {len(groups)} region/side group(s):")
    for (region, side), files in groups.items():
        side_label = 'Left' if side == 'L' else 'Right'
        print(f"  {region} {side_label}: channels {[ch for ch, _ in files]}")

    # Process each group
    csv_paths = []
    for group_key, channel_files in groups.items():
        path = build_csv(group_key, channel_files, output_folder)
        csv_paths.append(path)

    print(f"\nDone! {len(csv_paths)} CSV(s) created.")

    # Plot if requested
    if do_plot:
        for csv_path in csv_paths:
            plot_median_proxy(csv_path, t_start=0, t_end=150)


if __name__ == '__main__':
    main()