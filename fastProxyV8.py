"""
Fast proxy: 350 Hz biquad HP (fs=24414), NEO k=1, 128 samples/bin, median across channels.
Config: --channels=0-9,20-25  (comma-separated ranges, inclusive)
"""
#V7 but with printouts of channel selected + progress 
import numpy as np
import os
import sys
import h5py

# Biquad HP coeffs: 350 Hz, fs=24414, order 2, a0=1
B0 =  0.938290861982229
B1 = -1.876581723964458
B2 =  0.938290861982229
A1 = -1.872770074080853
A2 =  0.880393373848063

BIN = 128
FS  = 24414


class BiquadHP:
    __slots__ = ('x1', 'x2', 'y1', 'y2')

    def __init__(self):
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def process_chunk(self, chunk):
        out = np.empty(len(chunk))
        x1, x2, y1, y2 = self.x1, self.x2, self.y1, self.y2
        for n in range(len(chunk)):
            x = chunk[n]
            y = B0*x + B1*x1 + B2*x2 - A1*y1 - A2*y2
            x2, x1 = x1, x
            y2, y1 = y1, y
            out[n] = y
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        return out


def parse_channels(spec):
    """Parse '0-9,20-25' into sorted list of ints [0,1,...,9,20,...,25]."""
    indices = []
    for part in spec.split(','):
        if '-' in part:
            lo, hi = part.split('-')
            indices.extend(range(int(lo), int(hi) + 1))
        else:
            indices.append(int(part))
    return sorted(set(indices))


def load_signal(path):
    with h5py.File(path, 'r') as f:
        for key in f.keys():
            ds = f[key]
            if ds.ndim >= 1 and ds.size > 1000:
                return np.squeeze(ds[()]).astype(float)
    raise ValueError(f"No signal in {path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python fastProxy.py <folder> [output_folder] [--channels=0-9,20-25]")
        sys.exit(1)

    channel_indices = list(range(100))  # default: 0-99
    for a in sys.argv[1:]:
        if a.startswith('--channels='):
            channel_indices = parse_channels(a.split('=')[1])

    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    in_dir  = args[0]
    out_dir = args[1] if len(args) > 1 else in_dir

    # Map channel indices to files
    all_mat = sorted(f for f in os.listdir(in_dir) if f.lower().endswith('.mat'))
    channel_indices = [i for i in channel_indices if i < len(all_mat)]
    if not channel_indices:
        print(f"No valid channels ({len(all_mat)} .mat files in folder).")
        sys.exit(1)

    paths = [os.path.join(in_dir, all_mat[i]) for i in channel_indices]
    n_ch = len(paths)
    print(f"\n{n_ch} channels selected:")
    for idx, ch in enumerate(channel_indices):
        print(f"  ch {ch}: {all_mat[ch]}")

    # Load all signals
    print("\nLoading signals...")
    signals = [load_signal(p) for p in paths]
    n_bins = min(len(s) // BIN for s in signals)
    print(f"  {n_bins} bins x {n_ch} channels")

    # Process bin-by-bin across all channels (real-time order)
    filters = [BiquadHP() for _ in range(n_ch)]
    prevs   = [0.0] * n_ch
    median_proxy = np.zeros(n_bins)
    ch_vals = np.empty(n_ch)

    progress_step = max(1, n_bins // 10)
    for b in range(n_bins):
        if b % progress_step == 0:
            print(f"  bin {b}/{n_bins} ({100*b//n_bins}%)")
        start = b * BIN
        end   = start + BIN
        for c in range(n_ch):
            f = filters[c].process_chunk(signals[c][start:end])
            ext = np.empty(BIN + 1)
            ext[0] = prevs[c]
            ext[1:] = f
            neo = ext[1:-1]**2 - ext[:-2] * ext[2:]
            ch_vals[c] = np.mean(neo)
            prevs[c] = f[-1]
            #if we want to do 126 samples per bin, use this: (excluding the FIRST and LAST sample)
            # # neo = f[1:-1]**2 - f[:-2] * f[2:]
            # # ch_vals[c] = np.mean(neo) if len(neo) else 0.0
        median_proxy[b] = np.median(ch_vals)

    # Save
    time_s = np.arange(n_bins) * (BIN / FS)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'proxy_median.csv')
    np.savetxt(csv_path, np.column_stack([time_s, median_proxy]),
               delimiter=',', header='time_s,median_proxy', comments='', fmt='%.6e')
    print(f"  -> {csv_path}  ({n_bins} bins, {n_ch} channels)")


if __name__ == '__main__':
    main()