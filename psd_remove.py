import numpy as np
import os
import re
import sys
import csv
from collections import defaultdict
import h5py
from scipy.signal import welch
import plotly.graph_objects as go


# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════
HARMONICS       = (60, 120, 180, 240, 300, 360,420,480,540,600,660,720,780,840)
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


def list_mat_files(folder, recursive=False):
    """
    List .mat files in a folder. If recursive=True, walk subfolders too.
    Returns absolute file paths.
    """
    folder = os.path.abspath(folder)
    if not recursive:
        return [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if f.lower().endswith('.mat') and os.path.isfile(os.path.join(folder, f))
        ]

    paths = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith('.mat'):
                paths.append(os.path.join(root, f))
    return paths


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
        nperseg = min(len(signal), fs * 40)
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

def find_outlier_channels(channel_data, threshold_mads=OUTLIER_THRESH):
    """
    A channel is an outlier if its total harmonic noise power is
    more than threshold_mads * MAD above OR below the group median.

    Uses median + MAD (robust to outliers inflating the spread).
    """
    totals = np.array([d['total_noise'] for d in channel_data])

    group_median = np.median(totals)
    mad = np.median(np.abs(totals - group_median))

    if mad == 0:
        mad = 1e-30

    results = []
    for i, d in enumerate(channel_data):
        deviation = (totals[i] - group_median) / mad
        is_outlier_high = deviation > threshold_mads
        is_outlier_low = deviation < -threshold_mads

        results.append({
            'channel': d['channel'],
            'total_noise': d['total_noise'],
            'harm_powers': d['harm_powers'],
            'deviation_mads': round(deviation, 2),
            'is_outlier': is_outlier_high,
            'is_outlier_low': is_outlier_low,
        })

    return results, group_median, mad


# ──────────────────────────────────────────────
# Analyze one folder
# ──────────────────────────────────────────────

def analyze_period(micro_path, output_dir, patient, period,
                   threshold_std=OUTLIER_THRESH, save_plots=True, recursive=False):
    """
    Run PSD analysis on all channels in a folder.
    Returns (summary dict, list of rows for master CSV).
    """
    micro_path = os.path.abspath(micro_path)

    groups = group_mat_files(micro_path)
    all_mat_paths = list_mat_files(micro_path, recursive=recursive)
    if not all_mat_paths:
        return {}, []

    os.makedirs(output_dir, exist_ok=True)
    summary = {}
    csv_rows = []

    # If we didn't match any expected filenames, analyze everything as one group.
    if not groups:
        groups = {('unknown', 'U'): [(os.path.splitext(os.path.basename(p))[0], p) for p in all_mat_paths]}

    # Track which files were included in the nice (region,side,channel) grouping.
    grouped_paths = set()

    for (region, side), channel_files in sorted(groups.items()):
        side_label = 'Left' if side == 'L' else 'Right'
        if side == 'U':
            side_label = 'Unknown'
        group_key = f'{region}_{side}'
        print(f"    {region} {side_label} ({len(channel_files)} files)")

        channel_data = []
        all_freqs = []
        all_psds = []

        for channel, filepath in channel_files:
            grouped_paths.add(os.path.abspath(filepath))
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

        results, group_median, mad = find_outlier_channels(channel_data, threshold_std)

        outlier_channels = []
        outlier_low_channels = []
        clean_channels = []

        for r in results:
            ch = r['channel']
            dev = r['deviation_mads']

            if r['is_outlier']:
                status = "HIGH"
                outlier_channels.append(ch)
            elif r['is_outlier_low']:
                status = "LOW"
                outlier_low_channels.append(ch)
            else:
                status = "OK"
                clean_channels.append(ch)

            harm_str = ", ".join(
                f"{freq}Hz: {power:.2e}"
                for freq, power in sorted(r['harm_powers'].items())
            )
            print(f"      ch{ch}: {status} ({dev:+.1f} MADs)  total={r['total_noise']:.2e}  [{harm_str}]")

            # Build CSV row for every channel
            csv_rows.append({
                'patient': patient,
                'period': period,
                'region': region,
                'side': side_label,
                'channel': ch,
                'total_noise': r['total_noise'],
                'group_median': group_median,
                'mad': mad,
                'deviation_mads': dev,
                'is_outlier': r['is_outlier'],
                'is_outlier_low': r['is_outlier_low'],
                **{f'harm_{freq}Hz': r['harm_powers'].get(freq, 0.0) for freq in HARMONICS},
            })

        print(f"      Group median: {group_median:.2e}, MAD: {mad:.2e}")
        if outlier_channels:
            print(f"      ** High outliers: ch{outlier_channels}")
        if outlier_low_channels:
            print(f"      ** Low outliers:  ch{outlier_low_channels}")
        if not outlier_channels and not outlier_low_channels:
            print(f"      All channels within normal range")

        summary[group_key] = {
            'results': results,
            'outlier_channels': outlier_channels,
            'outlier_low_channels': outlier_low_channels,
            'clean_channels': clean_channels,
            'total': len(channel_files),
            'group_median': group_median,
            'mad': mad,
        }

        if save_plots:
            all_labels = [f'ch{r["channel"]}' for r in results]
            all_outlier = [r['is_outlier'] for r in results]

            ch_plot_dir = os.path.join(output_dir, f'{group_key}_channels')
            for idx in range(len(all_freqs)):
                plot_psd_single(
                    all_freqs[idx], all_psds[idx],
                    all_labels[idx], all_outlier[idx],
                    title_prefix=f'{patient} {period} {group_key}',
                    output_dir=ch_plot_dir,
                )

            plot_psd_overlay(
                all_freqs, all_psds, all_labels, all_outlier,
                title=f'{patient} {period} {group_key} PSD — Channel Comparison',
                output_path=os.path.join(output_dir, f'{group_key}_psd_overlay.png')
            )

            plot_noise_bar(
                results, group_median, mad, threshold_std,
                title=f'{patient} {period} {group_key} — Harmonic Noise per Channel',
                output_path=os.path.join(output_dir, f'{group_key}_noise_bars.png')
            )

    write_summary(summary, output_dir)
    return summary, csv_rows


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def plot_psd_single(freqs, psd, label, outlier, title_prefix, output_dir,
                    freq_max=5000):
    """Write one interactive HTML per channel."""
    os.makedirs(output_dir, exist_ok=True)

    mask = freqs <= freq_max
    color = 'red' if outlier else 'steelblue'
    status = "OUTLIER" if outlier else "OK"

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=freqs[mask], y=psd[mask],
        mode='lines', line=dict(width=1, color=color),
        name=label,
    ))

    for harm in HARMONICS:
        if harm <= freq_max:
            fig.add_vline(x=harm, line=dict(color='gray', width=0.5, dash='dash'), opacity=0.6)

    fig.update_layout(
        template='plotly_white',
        height=400,
        title=f'{title_prefix} — {label} [{status}]',
        xaxis_title='Frequency (Hz)',
        yaxis_title='PSD',
        yaxis_type='log',
        margin=dict(t=50, b=50, l=60, r=40),
    )

    html_path = os.path.join(output_dir, f'{label}_psd.html')
    fig.write_html(html_path, include_plotlyjs='cdn')
    print(f"      -> {html_path}")


def plot_psd_overlay(all_freqs, all_psds, all_labels, all_outlier, title, output_path,
                     freq_max=500):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig = go.Figure()

    for i in range(len(all_freqs)):
        freqs = all_freqs[i]
        psd = all_psds[i]
        label = all_labels[i]
        outlier = all_outlier[i]

        mask = freqs <= freq_max
        color = 'red' if outlier else 'steelblue'
        opacity = 0.9 if outlier else 0.4
        lw = 1.5 if outlier else 0.8

        fig.add_trace(go.Scattergl(
            x=freqs[mask], y=psd[mask],
            mode='lines',
            line=dict(width=lw, color=color),
            opacity=opacity,
            name=label,
        ))

    for harm in HARMONICS:
        if harm <= freq_max:
            fig.add_vline(x=harm, line=dict(color='orange', width=0.5, dash='dot'), opacity=0.5)

    fig.update_layout(
        template='plotly_white',
        height=500, width=1000,
        title=title,
        xaxis_title='Frequency (Hz)',
        yaxis_title='PSD',
        yaxis_type='log',
        legend=dict(font=dict(size=9)),
        margin=dict(t=50, b=50, l=60, r=40),
    )

    html_path = output_path.replace('.png', '.html')
    fig.write_html(html_path, include_plotlyjs='cdn')
    print(f"      -> Overlay: {html_path}")


def plot_noise_bar(results, group_median, mad, threshold_mads, title, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    channels = [f'ch{r["channel"]}' for r in results]
    totals = [r['total_noise'] for r in results]
    outliers = [r['is_outlier'] for r in results]
    colors = ['red' if o else 'steelblue' for o in outliers]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=channels, y=totals,
        marker_color=colors, opacity=0.8,
        name='Total Harmonic Noise',
    ))

    thresh_val = group_median + threshold_mads * mad
    fig.add_hline(y=thresh_val, line=dict(color='red', width=1, dash='dash'),
                  annotation_text=f'Outlier threshold ({threshold_mads} MADs)')
    fig.add_hline(y=group_median, line=dict(color='gray', width=1),
                  annotation_text='Group median')

    fig.update_layout(
        template='plotly_white',
        height=500, width=max(600, len(channels) * 60),
        title=title,
        yaxis_title='Total Harmonic Noise Power',
        yaxis_tickformat='.2e',
        margin=dict(t=50, b=50, l=80, r=40),
    )

    html_path = output_path.replace('.png', '.html')
    fig.write_html(html_path, include_plotlyjs='cdn')
    print(f"      -> Bar chart: {html_path}")


# ──────────────────────────────────────────────
# Summary text report (per period)
# ──────────────────────────────────────────────

def write_summary(summary, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, 'channel_quality_summary.txt')

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("Channel Quality Report — Outlier Detection (Median + MAD)\n")
        f.write(f"Outlier threshold: ±{OUTLIER_THRESH} MADs from group median\n")
        f.write(f"Harmonics checked: {HARMONICS}\n")
        f.write("=" * 60 + "\n\n")

        for group_key, info in sorted(summary.items()):
            outliers_high = info['outlier_channels']
            outliers_low = info['outlier_low_channels']
            clean = info['clean_channels']
            total = info['total']
            gm = info['group_median']
            m = info['mad']

            f.write(f"{group_key}: {len(outliers_high)} high, {len(outliers_low)} low / {total} channels\n")
            f.write(f"  Group median noise: {gm:.2e},  MAD: {m:.2e}\n")

            if outliers_high:
                f.write(f"  HIGH outliers: {outliers_high}\n")
            if outliers_low:
                f.write(f"  LOW outliers:  {outliers_low}\n")
            if not outliers_high and not outliers_low:
                f.write(f"  All channels within normal range\n")

            f.write("\n  Per-channel breakdown:\n")
            for r in info['results']:
                ch = r['channel']
                if r['is_outlier']:
                    status = "HIGH  "
                elif r['is_outlier_low']:
                    status = "LOW   "
                else:
                    status = "OK    "
                dev = r['deviation_mads']
                total_n = r['total_noise']
                f.write(f"    ch{ch}: {status}  {dev:+.1f} MADs  total={total_n:.2e}\n")
            f.write("\n")

    print(f"      -> Summary: {txt_path}")


# ──────────────────────────────────────────────
# Master CSVs (high and low outliers, separate files)
# ──────────────────────────────────────────────

def _write_csv(rows, output_path, label):
    """Helper to write a CSV from a list of row dicts."""
    if not rows:
        print(f"\nNo {label} channels found — no CSV written.")
        return

    fieldnames = [
        'patient', 'period', 'region', 'side', 'channel',
        'total_noise', 'group_median', 'mad', 'deviation_mads',
    ]
    for freq in HARMONICS:
        fieldnames.append(f'harm_{freq}Hz')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\n{label} CSV: {output_path}")
    print(f"  {len(rows)} channels")


def write_master_csvs(all_rows, output_dir):
    """Write two CSVs: one for high outliers, one for low outliers."""
    high_rows = [r for r in all_rows if r['is_outlier']]
    low_rows = [r for r in all_rows if r['is_outlier_low']]

    _write_csv(
        high_rows,
        os.path.join(output_dir, 'channels_to_remove_HIGH.csv'),
        'High outlier'
    )
    _write_csv(
        low_rows,
        os.path.join(output_dir, 'channels_to_remove_LOW.csv'),
        'Low outlier'
    )


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Drive scan:  python psd_remove.py <drive_root> <output_root> [--threshold=2.0]")
        print("  One folder:  python psd_remove.py --folder <mat_folder> <output_folder> [--threshold=2.0]")
        print()
        print(f"  --threshold : MADs above group median to flag (default: {OUTLIER_THRESH})")
        sys.exit(1)

    threshold_std = OUTLIER_THRESH
    recursive = '--recursive' in sys.argv
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
        print(f"Outlier threshold: ±{threshold_std} MADs\n")

        summary, csv_rows = analyze_period(
            mat_folder, output_folder,
            patient='unknown', period='unknown',
            threshold_std=threshold_std,
            recursive=recursive,
        )

        write_master_csvs(csv_rows, output_folder)
        print("Done!")
        return

    # ── Drive scan mode ──
    drive_root = args[0]
    output_root = args[1]

    if not os.path.isdir(drive_root):
        print(f"Error: '{drive_root}' is not a valid directory.")
        sys.exit(1)

    print(f"Scanning {drive_root} ...")
    print(f"Outlier threshold: ±{threshold_std} MADs\n")
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

        total_high = sum(len(v['outlier_channels']) for v in summary.values())
        total_low = sum(len(v['outlier_low_channels']) for v in summary.values())
        total_ch = sum(v['total'] for v in summary.values())
        print(f"    => {total_high} high, {total_low} low / {total_ch} channels\n")

    write_master_csvs(all_csv_rows, output_root)

    print("\nDone!")


if __name__ == '__main__':
    main()