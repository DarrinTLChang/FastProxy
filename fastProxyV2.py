import numpy as np
import os
import re
import sys
from collections import defaultdict
import h5py
from scipy.signal import butter, lfilter, lfilter_zi
import matplotlib.pyplot as plt

#HP (per bin causal), NEO (k=1), bin size 20 ms

HIGHPASS_CUTOFF = 350       # Hz
HIGHPASS_ORDER  = 2
NEO_K           = 1
NUM_SAMPLES     = 128        # samples per bin (the real-time packet size)


# ──────────────────────────────────────────────
# Core DSP functions (bin-by-bin, causal)
# ──────────────────────────────────────────────

def make_highpass(fs, cutoff=HIGHPASS_CUTOFF, order=HIGHPASS_ORDER):
    """Create highpass filter coefficients and initial state."""
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype='high')
    zi = lfilter_zi(b, a)
    return b, a, zi


def process_single_file(filepath, cutoff=HIGHPASS_CUTOFF, k=NEO_K, num_samples=NUM_SAMPLES):
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

    bin_size = num_samples
    n_bins = len(signal) // bin_size

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
# File discovery & grouping
# ──────────────────────────────────────────────

def parse_filename(filename):
    """
    Parse filenames like:
      - microVIM_L_9.mat
      - microVIM_L_9_CommonFiltered.mat

    The optional '_CommonFiltered' suffix is ignored so both map
    to the same (region, side, channel).
    """
    name = os.path.splitext(filename)[0]
    match = re.match(r'^micro([A-Za-z0-9]+)_([LR])_(\d+)(?:_CommonFiltered)?$', name)
    if not match:
        return None
    region, side, channel = match.groups()
    return region, side, int(channel)


def discover_and_group(folder):
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
# CSV output (per region)
# ──────────────────────────────────────────────

def build_csv(group_key, channel_files, output_dir,
              cutoff=HIGHPASS_CUTOFF, k=NEO_K, num_samples=NUM_SAMPLES):
    region, side = group_key
    side_label = 'Left' if side == 'L' else 'Right'
    print(f"\nProcessing {region} {side_label} ({len(channel_files)} channels)...")

    all_binned = []
    channels = []
    fs_val = None

    for channel, filepath in channel_files:
        print(f"  Channel {channel}: {os.path.basename(filepath)}")
        binned, fs = process_single_file(filepath, cutoff=cutoff, k=k, num_samples=num_samples)
        all_binned.append(binned)
        channels.append(channel)
        fs_val = fs

    min_len = min(len(b) for b in all_binned)
    all_binned = [b[:min_len] for b in all_binned]

    matrix = np.column_stack(all_binned)
    bin_duration_s = num_samples / fs_val
    time_s = np.arange(min_len) * bin_duration_s
    median_proxy = np.median(matrix, axis=1)

    header_parts = ['time_s']
    for ch in channels:
        header_parts.append(f'ch{ch}')
    header_parts.append(f'{region}_{side}_median_proxy')
    header = ','.join(header_parts)

    out_data = np.column_stack([time_s, matrix, median_proxy])

    os.makedirs(output_dir, exist_ok=True)
    csv_name = f'micro{region}_{side}_neo_binned.csv'
    csv_path = os.path.join(output_dir, csv_name)
    np.savetxt(csv_path, out_data, delimiter=',', header=header, comments='', fmt='%.6e')
    bin_ms = num_samples / fs_val * 1000
    print(f"  -> Saved: {csv_path}")
    print(f"     {min_len} bins, {len(channels)} channels, fs={fs_val} Hz, {bin_ms:.2f} ms/bin")

    return csv_path, all_binned, channels, fs_val


# ──────────────────────────────────────────────
# Hemisphere CSV output
# ──────────────────────────────────────────────

def build_hemisphere_csv(hemisphere_bins, output_dir, fs, num_samples=NUM_SAMPLES):
    """
    Build a single CSV: time_s, hemisphere_L_median_proxy, hemisphere_R_median_proxy
    Median across ALL channels on each side.
    """
    sides_with_data = [s for s in ['L', 'R'] if hemisphere_bins.get(s)]
    if not sides_with_data:
        return None

    side_medians = {}
    min_len = float('inf')

    for side in sides_with_data:
        bins = hemisphere_bins[side]
        side_min = min(len(b) for b in bins)
        min_len = min(min_len, side_min)

    min_len = int(min_len)

    for side in sides_with_data:
        bins = hemisphere_bins[side]
        trimmed = [b[:min_len] for b in bins]
        matrix = np.column_stack(trimmed)
        side_medians[side] = np.median(matrix, axis=1)

    bin_duration_s = num_samples / fs
    time_s = np.arange(min_len) * bin_duration_s

    header_parts = ['time_s']
    columns = [time_s]

    for side in ['L', 'R']:
        if side in side_medians:
            header_parts.append(f'hemisphere_{side}_median_proxy')
            columns.append(side_medians[side])

    header = ','.join(header_parts)
    out_data = np.column_stack(columns)

    os.makedirs(output_dir, exist_ok=True)
    csv_name = 'hemisphere_neo_binned.csv'
    csv_path = os.path.join(output_dir, csv_name)
    np.savetxt(csv_path, out_data, delimiter=',', header=header, comments='', fmt='%.6e')

    for side in sides_with_data:
        side_label = 'Left' if side == 'L' else 'Right'
        n_ch = len(hemisphere_bins[side])
        print(f"\n  Hemisphere {side_label}: {n_ch} channels")
    print(f"  -> Hemisphere CSV: {csv_path}")
    print(f"     {min_len} bins, {num_samples / fs * 1000:.2f} ms/bin")

    return csv_path


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def plot_median_proxy(csv_path, t_start=0, t_end=100):
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
    print(f"  -> Plot saved: {png_path}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python neo_pipeline.py <input_folder> [output_folder] [--plot]")
        sys.exit(1)

    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    do_plot = '--plot' in sys.argv

    input_folder = args[0]
    output_folder = args[1] if len(args) > 1 else input_folder

    if not os.path.isdir(input_folder):
        print(f"Error: '{input_folder}' is not a valid directory.")
        sys.exit(1)

    os.makedirs(output_folder, exist_ok=True)

    print(f"Config: cutoff={HIGHPASS_CUTOFF}Hz, order={HIGHPASS_ORDER}, "
          f"k={NEO_K}, num_samples={NUM_SAMPLES}\n")

    groups = discover_and_group(input_folder)

    if not groups:
        print("No matching .mat files found.")
        sys.exit(1)

    print(f"Found {sum(len(v) for v in groups.values())} files "
          f"in {len(groups)} region/side group(s):")
    for (region, side), files in groups.items():
        side_label = 'Left' if side == 'L' else 'Right'
        print(f"  {region} {side_label}: channels {[ch for ch, _ in files]}")

    csv_paths = []
    hemisphere_bins = {'L': [], 'R': []}
    fs_val = None

    for group_key, channel_files in groups.items():
        region, side = group_key
        path, all_binned, channels, fs = build_csv(
            group_key, channel_files, output_folder
        )
        csv_paths.append(path)
        fs_val = fs

        for binned in all_binned:
            hemisphere_bins[side].append(binned)

    # Build single hemisphere CSV
    if fs_val:
        hemi_path = build_hemisphere_csv(hemisphere_bins, output_folder, fs=fs_val)
        if hemi_path:
            csv_paths.append(hemi_path)

    print(f"\nDone! {len(csv_paths)} CSV(s) created.")

    if do_plot:
        for csv_path in csv_paths:
            plot_median_proxy(csv_path, t_start=0, t_end=150)


if __name__ == '__main__':
    main()