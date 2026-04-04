"""
Batch Stimulation Analysis — runs stim_analysis across multiple alt proxy CSVs.

Takes a single labeled CSV (with FIRE/SKIP labels) and a folder of regional
proxy CSVs (e.g. microGPi1_L_neo_binned.csv). For each *_L_neo_binned.csv,
uses its _median_proxy column as the alt proxy signal.

Also processes hemisphere_neo_binned.csv if present (one run per side column).

Usage:
  python batch_stim_analysis.py <labeled_csv> <proxy_folder> <output_root>

Example:
  python batch_stim_analysis.py labeled.csv proxy_folder output_root

Output structure:
  <output_root>/
    microGPi1_L/
      stim_events.csv, postStim_Fast_Proxy.html, ...
    microGPi2_L/
      ...
    hemisphere_L/
      ...
    hemisphere_R/
      ...
"""

import numpy as np
import os
import sys
import glob
import pandas as pd

from postStim import (
    load_labeled_csv, load_stim_periods, load_burst_csv,
    extract_events, verify_stim_delivery, filter_events_by_burst,
    compute_metrics, write_events_csv,
    plot_postStim_Fast_Proxy, plot_individual_traces,
    plot_post_stim_boxplot, plot_ibi_boxplot, plot_full_trace,
    THRESHOLD, STIM_DURATION_MS, ANALYSIS_END_MS, EXTRA_DELAY_MS,
    ADC_CSV_PATH, BURST_CSV_PATH, BURST_FILTER_ENABLE, BURST_PROXIMITY_MS,
    VERIFY_STIM_DELIVERY, MAX_VALID_DELAY_MS,
    _extract_proxy_col,
)
from scipy.stats import mannwhitneyu, ttest_ind

# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════
SIDE_FILTER = '_L_neo_binned.csv'   # change to '_R_neo_binned.csv' for right side


def find_proxy_csvs(folder, suffix=SIDE_FILTER):
    pattern = os.path.join(folder, f'*{suffix}')
    files = sorted(glob.glob(pattern))
    # Exclude hemisphere file from the regional list
    files = [f for f in files if 'hemisphere' not in os.path.basename(f).lower()]
    print(f"Found {len(files)} regional files matching *{suffix} in {folder}")
    for f in files:
        print(f"  {os.path.basename(f)}")
    return files


def find_hemisphere_csv(folder):
    """Find hemisphere_neo_binned.csv and return list of (col_name, side_label) pairs."""
    path = os.path.join(folder, 'hemisphere_neo_binned.csv')
    if not os.path.isfile(path):
        return path, []
    df = pd.read_csv(path, nrows=0)
    sides = []
    for col in df.columns:
        if col.endswith('_median_proxy') and 'hemisphere' in col:
            # e.g. hemisphere_L_median_proxy -> L
            parts = col.replace('_median_proxy', '').split('_')
            side = parts[-1] if len(parts) >= 2 else col
            sides.append((col, side))
    if sides:
        print(f"Found hemisphere CSV with {len(sides)} side(s): {[s for _, s in sides]}")
    return path, sides


def load_alt_proxy(csv_path, proxy_col=None):
    """Load time_s and proxy from a CSV. If proxy_col specified, use that column."""
    df = pd.read_csv(csv_path)
    time_s = df['time_s'].values
    if proxy_col and proxy_col in df.columns:
        proxy = df[proxy_col].values
        col = proxy_col
    else:
        median_cols = [c for c in df.columns if c.endswith('_median_proxy')]
        if not median_cols:
            raise ValueError(f"No _median_proxy column in {csv_path}. Columns: {list(df.columns)}")
        col = median_cols[0]
        proxy = df[col].values
    return time_s, proxy, col


def run_single(alt_proxy_csv, output_dir,
               time_s_main, proxy_main, labels_main,
               stim_starts, stim_ends, burst_starts, burst_ends,
               proxy_col=None):
    """Run analysis for one alt proxy CSV (optionally targeting a specific column)."""

    stim_dur_s = STIM_DURATION_MS / 1000.0
    analysis_end_s = ANALYSIS_END_MS / 1000.0

    # Load alt proxy
    alt_time_s, alt_proxy, used_col = load_alt_proxy(alt_proxy_csv, proxy_col=proxy_col)
    print(f"  Alt proxy: {used_col}, {len(alt_time_s)} samples, {alt_time_s[0]:.2f} - {alt_time_s[-1]:.2f} s")

    # Extract events from main CSV labels
    events = extract_events(time_s_main, proxy_main, labels_main)
    if not events:
        print("  No FIRE/SKIP events found, skipping.")
        return

    # Remap to alt proxy time grid
    for ev in events:
        idx = np.searchsorted(alt_time_s, ev['time_s'], side='left')
        if idx >= len(alt_time_s):
            idx = len(alt_time_s) - 1
        ev['csv_index'] = idx
        ev['proxy_feat'] = alt_proxy[idx]

    time_s = alt_time_s
    proxy = alt_proxy

    # Verify stim delivery
    if VERIFY_STIM_DELIVERY and stim_starts is not None:
        events = verify_stim_delivery(events, stim_starts, MAX_VALID_DELAY_MS / 1000.0)
        if not events:
            print("  No events after stim verification, skipping.")
            return

    # Burst filter
    if BURST_FILTER_ENABLE and burst_starts is not None:
        proximity_s = BURST_PROXIMITY_MS / 1000.0
        events = filter_events_by_burst(events, burst_starts, burst_ends, proximity_s)
        if not events:
            print("  No events after burst filter, skipping.")
            return

    # Compute metrics
    events = compute_metrics(events, time_s, proxy, THRESHOLD, stim_dur_s, analysis_end_s,
                              burst_starts_s=burst_starts, burst_ends_s=burst_ends,
                              proximity_s=BURST_PROXIMITY_MS / 1000.0)

    # Stats
    fire_evts = [e for e in events if e['stim_on']]
    skip_evts = [e for e in events if not e['stim_on']]

    print(f"  FIRE: {len(fire_evts)}, SKIP: {len(skip_evts)}")

    metrics = [
        ('post_stim_mean', 'Post-Stim Mean (analysis window)'),
        ('ibi_s', 'Inter-Burst Interval (s)'),
        ('refractory_crossings', 'Refractory Crossings'),
    ]

    for key, label in metrics:
        fire_v = np.array([e[key] for e in fire_evts if not np.isnan(e.get(key, np.nan))])
        skip_v = np.array([e[key] for e in skip_evts if not np.isnan(e.get(key, np.nan))])
        if len(fire_v) < 2 or len(skip_v) < 2:
            continue
        _, p_tt = ttest_ind(fire_v, skip_v, equal_var=False)
        sig = "***" if p_tt<0.001 else "**" if p_tt<0.01 else "*" if p_tt<0.05 else "n.s."
        print(f"  {label}: FIRE={np.mean(fire_v):.2f}, SKIP={np.mean(skip_v):.2f}, p={p_tt:.4f} ({sig})")

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    write_events_csv(events, os.path.join(output_dir, 'stim_events.csv'))
    plot_postStim_Fast_Proxy(events, stim_dur_s, ANALYSIS_END_MS, THRESHOLD,
                            os.path.join(output_dir, 'postStim_Fast_Proxy.html'))
    plot_individual_traces(events, stim_dur_s, ANALYSIS_END_MS, THRESHOLD,
                          os.path.join(output_dir, 'postStim_FastProxy_indivLines.html'))
    plot_post_stim_boxplot(events, os.path.join(output_dir, 'post_stim_boxplot.html'))
    plot_ibi_boxplot(events, os.path.join(output_dir, 'ibi_boxplot.html'))
    # plot_refractory_crossings(events, os.path.join(output_dir, 'refractory_crossings.html'))
    plot_full_trace(time_s, proxy, events, THRESHOLD,
                    os.path.join(output_dir, 'full_trace.html'),
                    stim_starts_s=stim_starts, stim_ends_s=stim_ends,
                    burst_starts_s=burst_starts, burst_ends_s=burst_ends)


def main():
    if len(sys.argv) < 4:
        print("Usage: python batch_stim_analysis.py <labeled_csv> <proxy_folder> <output_root>")
        sys.exit(1)

    labeled_csv = sys.argv[1]
    proxy_folder = sys.argv[2]
    output_root = sys.argv[3]

    # Load main labeled CSV once
    print(f"Loading labeled CSV: {labeled_csv}")
    time_s_main, proxy_main, labels_main = load_labeled_csv(labeled_csv)
    print(f"  {len(time_s_main)} samples, {time_s_main[0]:.2f} - {time_s_main[-1]:.2f} s")

    n_fire = sum(1 for l in labels_main if l == 'FIRE')
    n_skip = sum(1 for l in labels_main if l == 'SKIP')
    print(f"  Labels: FIRE={n_fire}, SKIP={n_skip}")

    # Load ADC stim data once
    stim_starts = stim_ends = None
    if ADC_CSV_PATH and os.path.isfile(ADC_CSV_PATH):
        print(f"\nLoading ADC: {ADC_CSV_PATH}")
        stim_starts, stim_ends = load_stim_periods(ADC_CSV_PATH)
        if stim_starts is not None:
            print(f"  {len(stim_starts)} stim windows")
    else:
        print(f"\nADC CSV not found: {ADC_CSV_PATH}")

    # Load burst data once
    burst_starts = burst_ends = None
    if BURST_CSV_PATH and os.path.isfile(BURST_CSV_PATH):
        print(f"\nLoading bursts: {BURST_CSV_PATH}")
        burst_starts, burst_ends = load_burst_csv(BURST_CSV_PATH)
        print(f"  {len(burst_starts)} burst periods")
    else:
        print(f"\nBurst CSV not found: {BURST_CSV_PATH}")

    # Find all matching regional proxy CSVs
    proxy_csvs = find_proxy_csvs(proxy_folder)

    # Find hemisphere CSV
    hemi_path, hemi_sides = find_hemisphere_csv(proxy_folder)

    total_runs = len(proxy_csvs) + len(hemi_sides)
    if total_runs == 0:
        print("No proxy CSVs found, exiting.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Running analysis for {len(proxy_csvs)} regional + {len(hemi_sides)} hemisphere CSV(s)")
    print(f"{'='*60}")

    run_idx = 0

    # Regional CSVs
    for alt_csv in proxy_csvs:
        run_idx += 1
        basename = os.path.basename(alt_csv)
        region_name = basename.replace('_neo_binned.csv', '')
        out_dir = os.path.join(output_root, region_name)

        print(f"\n{'─'*60}")
        print(f"[{run_idx}/{total_runs}] {basename} -> {region_name}/")
        print(f"{'─'*60}")

        try:
            run_single(alt_csv, out_dir,
                       time_s_main, proxy_main, labels_main,
                       stim_starts, stim_ends, burst_starts, burst_ends)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Hemisphere CSV — one run per side column
    for col_name, side in hemi_sides:
        run_idx += 1
        out_dir = os.path.join(output_root, f'hemisphere_{side}')

        print(f"\n{'─'*60}")
        print(f"[{run_idx}/{total_runs}] hemisphere_neo_binned.csv ({col_name}) -> hemisphere_{side}/")
        print(f"{'─'*60}")

        try:
            run_single(hemi_path, out_dir,
                       time_s_main, proxy_main, labels_main,
                       stim_starts, stim_ends, burst_starts, burst_ends,
                       proxy_col=col_name)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"Batch complete. Results in: {output_root}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()