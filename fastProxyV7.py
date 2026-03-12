import numpy as np
import os
import re
import sys
from collections import defaultdict
import h5py
import matplotlib.pyplot as plt
#added blacklist

# ══════════════════════════════════════════════
# GLOBAL CONFIG — change these once here
# ══════════════════════════════════════════════
HIGHPASS_CUTOFF  = 350       # Hz
NEO_K            = 1
NUM_SAMPLES      = 489       # samples per bin (the real-time packet size)


# ══════════════════════════════════════════════
# BLACKLIST CONFIG — toggle each CSV on/off
# ══════════════════════════════════════════════
BLACKLIST_CSVS = {
    'amplitude':   r"F:\fastProxy_outputs\abnormal_peak\abnormal_amplitude_channels.csv",
    'correlation': r"F:\fastProxy_outputs\correlation_check\channels_low_correlation.csv",
    'psd_high':    r"F:\fastProxy_outputs\psd_std3\channels_to_remove_HIGH",
    'psd_low':     r"F:\fastProxy_outputs\psd_std3\channels_to_remove_LOW",
}

# Set to True/False to enable/disable each blacklist source
BLACKLIST_ENABLE = {
    'amplitude':   False,
    'correlation': False,
    'psd_high':    False,
    'psd_low':     False,
}

# Current run: only channels blacklisted for THIS patient/period are excluded.
# Set to None to skip blacklist filtering (no channels excluded by blacklist).
CURRENT_PATIENT = "s523"   # e.g. "s523"
CURRENT_PERIOD = "period1" # e.g. "1" or "period1" (must match how period appears in CSV)


# ──────────────────────────────────────────────
# Blacklist loader
# ──────────────────────────────────────────────

def _normalize_period(period_str):
    """Allow '1' to match 'period1' and vice versa."""
    s = (period_str or "").strip().lower()
    if s.isdigit():
        return s, f"period{s}"
    if s.startswith("period") and s[6:].isdigit():
        return s[6:], s
    return s, s


def load_blacklist(patient, period):
    """
    Load enabled blacklist CSVs and return a set of (region, side, channel)
    to exclude **only for the given patient and period**.

    Each CSV must have columns: patient, period, region, side, channel.
    Rows from other patients/periods are ignored so we only blacklist
    channels that match the current run.
    """
    import csv as csv_module

    blacklisted = set()
    side_map = {'Left': 'L', 'Right': 'R', 'L': 'L', 'R': 'R'}

    patient = (patient or "").strip()
    period_alt1, period_alt2 = _normalize_period(period)

    for name, enabled in BLACKLIST_ENABLE.items():
        if not enabled:
            continue

        csv_path = BLACKLIST_CSVS.get(name, "")
        if not csv_path:
            continue
        # Allow no extension or .csv
        if not os.path.isfile(csv_path):
            csv_path = csv_path.rstrip("/\\") + ".csv"
        if not os.path.isfile(csv_path):
            print(f"  Blacklist '{name}': file not found, skipping")
            continue

        count = 0
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                row_patient = (row.get('patient') or '').strip()
                row_period = (row.get('period') or '').strip().lower()
                region = (row.get('region') or '').strip()
                side = side_map.get((row.get('side') or '').strip(), '')
                ch_str = (row.get('channel') or '').strip()

                if not region or not side or not ch_str:
                    continue
                # Only blacklist if patient and period match this run
                if row_patient != patient:
                    continue
                if row_period != period_alt1 and row_period != period_alt2:
                    continue

                try:
                    blacklisted.add((region, side, int(ch_str)))
                    count += 1
                except ValueError:
                    continue

        if count > 0:
            print(f"  Blacklist '{name}': {count} channel(s) for {patient} {period}")

    return blacklisted


# ──────────────────────────────────────────────
# Biquad IIR Highpass Filter (no scipy needed at runtime)
# ──────────────────────────────────────────────
#
# A single biquad (order=2) implements:
#   y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
#
# Cost: 5 multiplies + 4 adds per sample
# State: 4 values (x[n-1], x[n-2], y[n-1], y[n-2])
#
# For higher orders, cascade multiple biquads in series.
# Order 2 = 1 biquad  (12 dB/oct)
# Order 4 = 2 biquads (24 dB/oct)
# Order 6 = 3 biquads (36 dB/oct)

def compute_biquad_highpass_coeffs(fs, cutoff):
    """
    Compute biquad highpass coefficients using the bilinear transform.
    This is what scipy.signal.butter(2, ..., btype='high') computes.

    On real hardware you'd precompute these once and hardcode them.

    Returns (b0, b1, b2, a1, a2) with a0 normalized to 1.
    """
    # Prewarp the cutoff frequency
    wc = 2 * np.pi * cutoff
    T = 1 / fs
    wc_warped = (2 / T) * np.tan(wc * T / 2)

    # Second-order Butterworth analog prototype: s^2 + sqrt(2)*s + 1
    # Bilinear transform substitution: s = (2/T)*(z-1)/(z+1)
    K = wc_warped * T / 2

    # Denominator: K^2 + sqrt(2)*K + 1
    denom = K**2 + np.sqrt(2) * K + 1

    b0 = 1.0 / denom
    b1 = -2.0 / denom
    b2 = 1.0 / denom

    a1 = (2 * K**2 - 2) / denom
    a2 = (K**2 - np.sqrt(2) * K + 1) / denom

    return b0, b1, b2, a1, a2


class BiquadHighpass:
    """
    A single biquad highpass filter section.
    Cascade multiple instances for higher orders.

    Per-sample cost: 5 multiplies, 4 adds.
    Memory: 4 floats (x1, x2, y1, y2).
    """

    def __init__(self, fs, cutoff):
        self.b0, self.b1, self.b2, self.a1, self.a2 = \
            compute_biquad_highpass_coeffs(fs, cutoff)
        # State
        self.x1 = 0.0   # x[n-1]
        self.x2 = 0.0   # x[n-2]
        self.y1 = 0.0   # y[n-1]
        self.y2 = 0.0   # y[n-2]

    def reset(self):
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def process_sample(self, x):
        """Filter one sample. This is the hot loop on hardware."""
        y = (self.b0 * x
             + self.b1 * self.x1
             + self.b2 * self.x2
             - self.a1 * self.y1
             - self.a2 * self.y2)

        # Shift state
        self.x2 = self.x1
        self.x1 = x
        self.y2 = self.y1
        self.y1 = y

        return y

    def process_chunk(self, chunk):
        """Filter a numpy array of samples. Used for simulation."""
        out = np.empty(len(chunk))
        for n in range(len(chunk)):
            out[n] = self.process_sample(chunk[n])
        return out


class CascadedHighpass:
    """
    Cascade multiple biquad sections for higher-order filtering.

    Order 2 = 1 biquad  (12 dB/oct)  — 5 mult/sample
    Order 4 = 2 biquads (24 dB/oct)  — 10 mult/sample
    Order 6 = 3 biquads (36 dB/oct)  — 15 mult/sample

    Only even orders supported (each biquad = order 2).
    """

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
        """Pass one sample through all biquad sections in series."""
        y = x
        for s in self.sections:
            y = s.process_sample(y)
        return y

    def process_chunk(self, chunk):
        """Filter a numpy array through the cascade."""
        out = chunk.copy()
        for s in self.sections:
            out = s.process_chunk(out)
        return out


# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────

def load_signal_from_mat(filepath):
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
        with h5py.File(filepath, 'r') as f:
            keys = list(f.keys())
            shapes = {ky: f[ky].shape for ky in keys}
        raise ValueError(
            f"Could not auto-detect signal/fs in {filepath}. "
            f"Keys found: {shapes}"
        )

    return signal, fs


def apply_car_inplace(signals, n_bins, num_samples):
    """
    Common Average Reference: for each bin, subtract the mean across channels
    from every channel. Operates on raw signals before highpass filtering.

    signals: list of 1D arrays (one per channel, all same length or longer)
    n_bins:  number of bins to process
    num_samples: samples per bin

    Modifies signals in-place.
    """
    n_ch = len(signals)
    for i in range(n_bins):
        start = i * num_samples
        end = start + num_samples
        # Compute mean across channels for this bin
        car = np.zeros(num_samples)
        for c in range(n_ch):
            car += signals[c][start:end]
        car /= n_ch
        # Subtract from each channel
        for c in range(n_ch):
            signals[c][start:end] -= car


def process_signal(signal, fs, n_bins, cutoff=HIGHPASS_CUTOFF, k=NEO_K,
                   num_samples=NUM_SAMPLES, filter_order=2):
    """
    Run HP -> NEO -> mean per bin on an already-loaded (and CAR'd) signal.

    For each bin of num_samples:
      1. Biquad highpass filter (state carries across bins)
      2. NEO with k=1, using lookback from previous bin (N-1 values per N-sample bin)
      3. Mean of the NEO output for that bin

    Returns 1D array of per-bin means.
    """
    hp = CascadedHighpass(fs, cutoff, order=filter_order)
    bin_means = np.zeros(n_bins)
    prev = 0.0

    for i in range(n_bins):
        chunk = signal[i * num_samples : (i + 1) * num_samples]
        filtered = hp.process_chunk(chunk)

        ext = np.empty(len(filtered) + 1)
        ext[0] = prev
        ext[1:] = filtered
        neo_out = ext[1:-1] ** 2 - ext[:-2] * ext[2:]

        bin_means[i] = np.mean(neo_out) if len(neo_out) > 0 else 0.0
        prev = filtered[-1]

    return bin_means


# ──────────────────────────────────────────────
# File discovery & grouping
# ──────────────────────────────────────────────

def parse_filename(filename):
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
# CSV output
# ──────────────────────────────────────────────

def build_csv(group_key, channel_files, output_dir, blacklist=None,
              cutoff=HIGHPASS_CUTOFF, k=NEO_K, num_samples=NUM_SAMPLES, filter_order=2):
    region, side = group_key
    side_label = 'Left' if side == 'L' else 'Right'

    # Keep full list for loading and CAR; blacklist only affects which channels go to proxy/CSV
    channel_files_full = list(channel_files)
    n_total = len(channel_files_full)

    n_included = n_total
    if blacklist:
        n_excluded = sum(1 for ch, _ in channel_files_full if (region, side, ch) in blacklist)
        n_included = n_total - n_excluded
        if n_excluded > 0:
            print(f"\n  {region} {side_label}: {n_excluded} channel(s) blacklisted, "
                  f"{n_included} used for proxy (CAR still uses all {n_total} channels)")
        if n_included == 0:
            print(f"\n  {region} {side_label}: ALL channels blacklisted, skipping.")
            return None, [], [], None

    print(f"\nProcessing {region} {side_label} ({n_total} channels for CAR, "
          f"{n_included if blacklist else n_total} for proxy)...")

    # Step 1: Load all signals (every channel in the group)
    print(f"  Loading signals...")
    signals = []
    fs_val = None
    for channel, filepath in channel_files_full:
        print(f"    ch{channel}: {os.path.basename(filepath)}")
        signal, fs = load_signal_from_mat(filepath)
        signals.append(signal)
        fs_val = fs

    n_bins = min(len(s) // num_samples for s in signals)

    # Step 2: CAR using ALL channels (mean over all, subtract from all)
    print(f"  Applying Common Average Reference (all {len(signals)} channels)...")
    apply_car_inplace(signals, n_bins, num_samples)

    # Step 3: Which channels to include in proxy/CSV (exclude blacklisted)
    if blacklist:
        included_indices = [i for i in range(n_total)
                           if (region, side, channel_files_full[i][0]) not in blacklist]
    else:
        included_indices = list(range(n_total))

    # Step 4: HP -> NEO -> mean only for included channels
    print(f"  Running HP + NEO on {len(included_indices)} channel(s)...")
    all_binned = []
    channels = []
    for i in included_indices:
        ch = channel_files_full[i][0]
        channels.append(ch)
        binned = process_signal(signals[i], fs_val, n_bins, cutoff=cutoff, k=k,
                                num_samples=num_samples, filter_order=filter_order)
        all_binned.append(binned)

    matrix = np.column_stack(all_binned)
    bin_duration_s = num_samples / fs_val
    time_s = np.arange(n_bins) * bin_duration_s
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
    print(f"     {n_bins} bins, {len(channels)} channels, fs={fs_val} Hz, {bin_ms:.2f} ms/bin")

    return csv_path, all_binned, channels, fs_val


# ──────────────────────────────────────────────
# Hemisphere CSV output
# ──────────────────────────────────────────────

def build_hemisphere_csv(hemisphere_bins, output_dir, fs, num_samples=NUM_SAMPLES):
    """
    Build a single CSV with columns: time_s, hemisphere_L_median_proxy, hemisphere_R_median_proxy
    Takes the median across ALL channels on each side.

    hemisphere_bins: dict {'L': [list of 1D arrays], 'R': [list of 1D arrays]}
    """
    # Find which sides have data
    sides_with_data = [s for s in ['L', 'R'] if hemisphere_bins.get(s)]
    if not sides_with_data:
        return None

    # Compute median per side
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

    # Build columns: time, L median (if exists), R median (if exists)
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
    """Print the biquad coefficients you'd hardcode on your device."""
    n_sections = order // 2
    print(f"\n{'='*50}")
    print(f"Biquad Highpass Coefficients")
    print(f"  fs={fs} Hz, cutoff={cutoff} Hz, order={order} ({n_sections} section(s))")
    print(f"{'='*50}")

    b0, b1, b2, a1, a2 = compute_biquad_highpass_coeffs(fs, cutoff)

    print(f"\n  // Per biquad section (all sections identical for Butterworth):")
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
        print("Usage: python fastProxyV6.py <input_folder> [output_folder] [--plot] [--order=2] [--coeffs]")
        print("       [--patient=ID] [--period=P]")
        print()
        print("  --order=N     : filter order (must be even: 2, 4, 6). Default: 2")
        print("  --patient=ID  : override CURRENT_PATIENT for blacklist (e.g. s523)")
        print("  --period=P    : override CURRENT_PERIOD for blacklist (e.g. 1 or period1)")
        print("  --coeffs      : print hardcoded biquad coefficients and exit")
        sys.exit(1)

    # Parse flags
    filter_order = 2
    for a in sys.argv[1:]:
        if a.startswith('--order='):
            filter_order = int(a.split('=')[1])

    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    do_plot = '--plot' in sys.argv
    do_coeffs = '--coeffs' in sys.argv

    # Coeffs-only mode: just print and exit
    if do_coeffs:
        # Need fs — use a dummy or load from first file
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

    # Resolve patient/period for blacklist (only exclude channels for THIS run)
    patient = getattr(sys.modules[__name__], 'CURRENT_PATIENT', None)
    period = getattr(sys.modules[__name__], 'CURRENT_PERIOD', None)
    for a in sys.argv[1:]:
        if a.startswith('--patient='):
            patient = a.split('=', 1)[1].strip()
        elif a.startswith('--period='):
            period = a.split('=', 1)[1].strip()

    print(f"Config: cutoff={HIGHPASS_CUTOFF}Hz, biquad order={filter_order} "
          f"({filter_order//2} section(s)), k={NEO_K}, num_samples={NUM_SAMPLES}\n")

    # Load channel blacklist for this patient/period only
    blacklist = set()
    if patient and period:
        print(f"Blacklist filter: patient={patient}, period={period}")
        print("Loading blacklists...")
        blacklist = load_blacklist(patient, period)
        if blacklist:
            print(f"  Total blacklisted for this run: {len(blacklist)} channel(s)\n")
        else:
            print("  No channels blacklisted for this patient/period\n")
    else:
        print("  Blacklist disabled (CURRENT_PATIENT or CURRENT_PERIOD not set)\n")

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

    # Collect per-channel bin data by side for hemisphere computation
    hemisphere_bins = {'L': [], 'R': []}
    fs_val = None

    for group_key, channel_files in groups.items():
        region, side = group_key
        path, all_binned, channels, fs = build_csv(
            group_key, channel_files, output_folder,
            blacklist=blacklist, filter_order=filter_order
        )

        if path is None:
            continue

        csv_paths.append(path)
        fs_val = fs

        # Accumulate for hemisphere
        for binned in all_binned:
            hemisphere_bins[side].append(binned)

    # Build single hemisphere CSV (both sides in one file)
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