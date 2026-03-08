import numpy as np
import os
import sys
import h5py
from scipy.signal import butter, lfilter, lfilter_zi, welch
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════
HIGHPASS_CUTOFF = 350       # Hz
HIGHPASS_ORDER  = 3
BIN_MS          = 20        # ms per bin


# ──────────────────────────────────────────────
# Load signal
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
# Causal highpass (bin-by-bin, same as neo_pipeline)
# ──────────────────────────────────────────────

def apply_highpass_binwise(signal, fs, cutoff=HIGHPASS_CUTOFF, order=HIGHPASS_ORDER,
                           bin_ms=BIN_MS):
    """
    Apply causal highpass filter bin-by-bin, identical to how the
    real-time system processes 20ms packets.
    Returns the full filtered signal (concatenated bins).
    """
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype='high')
    zi = lfilter_zi(b, a) * signal[0]

    bin_size = int(fs * bin_ms / 1000)
    n_bins = len(signal) // bin_size

    filtered = np.zeros(n_bins * bin_size)

    for i in range(n_bins):
        chunk = signal[i * bin_size : (i + 1) * bin_size]
        filt_chunk, zi = lfilter(b, a, chunk, zi=zi)
        filtered[i * bin_size : (i + 1) * bin_size] = filt_chunk

    return filtered[:n_bins * bin_size], signal[:n_bins * bin_size]


# ──────────────────────────────────────────────
# PSD comparison plot
# ──────────────────────────────────────────────

def plot_before_after_psd(raw, filtered, fs, label, output_path, freq_max=2000):
    """Plot PSD of raw vs filtered signal on the same axes."""
    nperseg = min(len(raw), fs * 2)
    freqs_raw, psd_raw = welch(raw, fs=fs, nperseg=nperseg)
    freqs_filt, psd_filt = welch(filtered, fs=fs, nperseg=nperseg)

    fig, ax = plt.subplots(figsize=(12, 5))

    mask_r = freqs_raw <= freq_max
    mask_f = freqs_filt <= freq_max

    ax.semilogy(freqs_raw[mask_r], psd_raw[mask_r],
                linewidth=0.6, color='gray', alpha=0.7, label='Raw')
    ax.semilogy(freqs_filt[mask_f], psd_filt[mask_f],
                linewidth=0.6, color='steelblue', label=f'After HP {HIGHPASS_CUTOFF}Hz (order {HIGHPASS_ORDER})')

    ax.axvline(HIGHPASS_CUTOFF, color='red', linestyle='--', linewidth=0.8,
               alpha=0.7, label=f'Cutoff: {HIGHPASS_CUTOFF} Hz')

    for harm in [60, 120, 180, 240, 300, 360, 420, 480]:
        ax.axvline(harm, color='orange', linestyle=':', linewidth=0.4, alpha=0.5)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.set_title(f'{label} — PSD Before & After Highpass')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  -> {output_path}")
    # plt.show()


# ──────────────────────────────────────────────
# Time domain comparison plot
# ──────────────────────────────────────────────

def plot_before_after_time(raw, filtered, fs, label, output_path,
                           t_start=0, t_end=0.1):
    """Plot a short time window of raw vs filtered signal."""
    n_start = int(t_start * fs)
    n_end = int(t_end * fs)
    t = np.arange(n_start, n_end) / fs

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    axes[0].plot(t, raw[n_start:n_end], linewidth=0.4, color='gray')
    axes[0].set_ylabel('Raw')
    axes[0].set_title(f'{label} — Time Domain [{t_start}–{t_end} s]')

    axes[1].plot(t, filtered[n_start:n_end], linewidth=0.4, color='steelblue')
    axes[1].set_ylabel(f'HP {HIGHPASS_CUTOFF}Hz')
    axes[1].set_xlabel('Time (s)')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  -> {output_path}")
    # plt.show()


# ──────────────────────────────────────────────
# Process one file
# ──────────────────────────────────────────────

def process_file(filepath, output_dir):
    """Load, filter, and plot PSD + time domain for one .mat file."""
    basename = os.path.splitext(os.path.basename(filepath))[0]
    print(f"\n  {basename}")

    signal, fs = load_signal(filepath)
    print(f"    {len(signal)} samples, fs={fs} Hz")

    filtered, raw_trimmed = apply_highpass_binwise(signal, fs)

    os.makedirs(output_dir, exist_ok=True)

    # PSD comparison
    plot_before_after_psd(
        raw_trimmed, filtered, fs,
        label=basename,
        output_path=os.path.join(output_dir, f'{basename}_psd_compare.png')
    )

    # Time domain comparison (first 100ms)
    plot_before_after_time(
        raw_trimmed, filtered, fs,
        label=basename,
        output_path=os.path.join(output_dir, f'{basename}_time_compare.png')
    )


# ──────────────────────────────────────────────
# Process a folder of .mat files
# ──────────────────────────────────────────────

def process_folder(mat_folder, output_dir):
    """Process every .mat file in a folder."""
    mat_files = sorted([f for f in os.listdir(mat_folder) if f.lower().endswith('.mat')])

    if not mat_files:
        print(f"No .mat files found in {mat_folder}")
        return

    print(f"Found {len(mat_files)} .mat files in {mat_folder}")
    print(f"Config: cutoff={HIGHPASS_CUTOFF}Hz, order={HIGHPASS_ORDER}, bin={BIN_MS}ms\n")

    for fname in mat_files:
        filepath = os.path.join(mat_folder, fname)
        process_file(filepath, output_dir)

    print(f"\nDone! Plots saved to {output_dir}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Single file:  python highpass_check.py <file.mat> <output_folder>")
        print("  Whole folder: python highpass_check.py <mat_folder> <output_folder>")
        print()
        print(f"Current config: cutoff={HIGHPASS_CUTOFF}Hz, order={HIGHPASS_ORDER}")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    if os.path.isfile(input_path):
        # Single file mode
        process_file(input_path, output_dir)
    elif os.path.isdir(input_path):
        # Folder mode
        process_folder(input_path, output_dir)
    else:
        print(f"Error: '{input_path}' is not a valid file or directory.")
        sys.exit(1)


if __name__ == '__main__':
    main()