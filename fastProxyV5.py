import numpy as np
import os
import re
import sys
from collections import defaultdict
import h5py
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════
# GLOBAL CONFIG — change these once here
# ══════════════════════════════════════════════
HIGHPASS_CUTOFF = 350       # Hz
NEO_K           = 1
NUM_SAMPLES     = 128       # samples per bin (the real-time packet size)


# ──────────────────────────────────────────────
# Biquad IIR Highpass Filter (no scipy needed at runtime)
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


class BiquadHighpass:
    def __init__(self, fs, cutoff):
        self.b0, self.b1, self.b2, self.a1, self.a2 = \
            compute_biquad_highpass_coeffs(fs, cutoff)
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0

    def reset(self):
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def process_sample(self, x):
        y = (self.b0 * x
             + self.b1 * self.x1
             + self.b2 * self.x2
             - self.a1 * self.y1
             - self.a2 * self.y2)
        self.x2 = self.x1
        self.x1 = x
        self.y2 = self.y1
        self.y1 = y
        return y

    def process_chunk(self, chunk):
        out = np.empty(len(chunk))
        for n in range(len(chunk)):
            out[n] = self.process_sample(chunk[n])
        return out


class CascadedHighpass:
    def __init__(self, fs, cutoff, order=2):
        if order % 2 != 0:
            raise ValueError("Biquad cascade requires even order (2, 4, 6, ...)")
        n_sections = order // 2
        self.sections = [BiquadHighpass(fs, cutoff) for _ in range(n_sections)]
        self.order = order

    def reset(self):
        for s in self.sections:
            s.reset()

    def process_sample(self, x):
        y = x
        for s in self.sections:
            y = s.process_sample(y)
        return y

    def process_chunk(self, chunk):
        out = chunk.copy()
        for s in self.sections:
            out = s.process_chunk(out)
        return out


# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────

def process_single_file(filepath, cutoff=HIGHPASS_CUTOFF, k=NEO_K,
                        num_samples=NUM_SAMPLES, filter_order=2):
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

    hp = CascadedHighpass(fs, cutoff, order=filter_order)

    bin_means = np.zeros(n_bins)

    for i in range(n_bins):
        chunk = signal[i * bin_size : (i + 1) * bin_size]
        filtered = hp.process_chunk(chunk)

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
    name = os.path.splitext(filename)[0]
    match = re.match(r'^micro([A-Za-z0-9]+)_([LR])_(\d+)$', name)
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
# CSV output
# ──────────────────────────────────────────────

def build_csv(group_key, channel_files, output_dir,
              cutoff=HIGHPASS_CUTOFF, k=NEO_K, num_samples=NUM_SAMPLES, filter_order=2):
    region, side = group_key
    side_label = 'Left' if side == 'L' else 'Right'
    print(f"\nProcessing {region} {side_label} ({len(channel_files)} channels)...")

    all_binned = []
    channels = []
    fs_val = None

    for channel, filepath in channel_files:
        print(f"  Channel {channel}: {os.path.basename(filepath)}")
        binned, fs = process_single_file(filepath, cutoff=cutoff, k=k,
                                         num_samples=num_samples,
                                         filter_order=filter_order)
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
# Print coefficients (for hardcoding on hardware)
# ──────────────────────────────────────────────

def print_coefficients(fs, cutoff, order=2):
    n_sections = order // 2
    print(f"\n{'='*50}")
    print(f"Biquad Highpass Coefficients")
    print(f"  fs={fs} Hz, cutoff={cutoff} Hz, order={order} ({n_sections} section(s))")
    print(f"{'='*50}")

    b0, b1, b2, a1, a2 = compute_biquad_highpass_coeffs(fs, cutoff)

    print(f"\n  // Per biquad section:")
    print(f"  const float b0 = {b0:.15f};")
    print(f"  const float b1 = {b1:.15f};")
    print(f"  const float b2 = {b2:.15f};")
    print(f"  const float a1 = {a1:.15f};")
    print(f"  const float a2 = {a2:.15f};")
    print(f"\n  // y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]")
    print(f"  // Cascade {n_sections} section(s) in series for order {order}")
    print()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python neo_pipeline.py <input_folder> [output_folder] [--plot] [--order=2] [--samples=128] [--coeffs]")
        print()
        print("  --order=N    : filter order (must be even: 2, 4, 6). Default: 2")
        print("  --samples=N  : samples per bin. Default: 128")
        print("  --coeffs     : print hardcoded biquad coefficients and exit")
        sys.exit(1)

    # Parse flags
    filter_order = 2
    num_samples = NUM_SAMPLES
    for a in sys.argv[1:]:
        if a.startswith('--order='):
            filter_order = int(a.split('=')[1])
        elif a.startswith('--samples='):
            num_samples = int(a.split('=')[1])

    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    do_plot = '--plot' in sys.argv
    do_coeffs = '--coeffs' in sys.argv

    # Coeffs-only mode
    if do_coeffs:
        if args:
            input_folder = args[0]
            if os.path.isdir(input_folder):
                for f in os.listdir(input_folder):
                    if f.lower().endswith('.mat'):
                        sig, fs = None, None
                        with h5py.File(os.path.join(input_folder, f), 'r') as hf:
                            for key in hf.keys():
                                ds = hf[key]
                                if ds.ndim == 0 or (ds.ndim >= 1 and ds.size == 1):
                                    fs = int(np.squeeze(ds[()]))
                        if fs:
                            print_coefficients(fs, HIGHPASS_CUTOFF, filter_order)
                            return
        print("Could not determine fs. Printing for common rates:")
        for fs in [24000, 30000, 44100, 48000]:
            print_coefficients(fs, HIGHPASS_CUTOFF, filter_order)
        return

    input_folder = args[0]
    output_folder = args[1] if len(args) > 1 else input_folder

    if not os.path.isdir(input_folder):
        print(f"Error: '{input_folder}' is not a valid directory.")
        sys.exit(1)

    os.makedirs(output_folder, exist_ok=True)

    print(f"Config: cutoff={HIGHPASS_CUTOFF}Hz, biquad order={filter_order} "
          f"({filter_order//2} section(s)), k={NEO_K}, num_samples={num_samples}\n")

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
            group_key, channel_files, output_folder,
            num_samples=num_samples, filter_order=filter_order
        )
        csv_paths.append(path)
        fs_val = fs

        for binned in all_binned:
            hemisphere_bins[side].append(binned)

    # Build single hemisphere CSV
    if fs_val:
        hemi_path = build_hemisphere_csv(hemisphere_bins, output_folder,
                                          fs=fs_val, num_samples=num_samples)
        if hemi_path:
            csv_paths.append(hemi_path)

    print(f"\nDone! {len(csv_paths)} CSV(s) created.")

    if do_plot:
        for csv_path in csv_paths:
            plot_median_proxy(csv_path, t_start=0, t_end=150)


if __name__ == '__main__':
    main()