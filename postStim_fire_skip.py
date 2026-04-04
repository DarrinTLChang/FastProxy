"""
Stimulation Effect Analysis — TWO CSV version.
Compares events from two separate labeled proxy CSVs.

CSV_A provides FIRE events, CSV_B provides SKIP events.
Each CSV uses its own proxy trace for computing metrics/traces.

Usage:
  python stim_analysis_2csv.py <fire_csv> <skip_csv> <output_folder>
"""

import numpy as np
import os
import sys
import csv
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import mannwhitneyu, ttest_ind


# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════
THRESHOLD          = 95
STIM_DURATION_MS   = 200
ANALYSIS_END_MS    = 450

DELAY_M            = 0.0218285
DELAY_B            = 24
EXTRA_DELAY_MS     = 20

BURST_FILTER_ENABLE = False
BURST_CSV_PATH      = r"C:\Users\Maral\Downloads\day5_test_p8_network_burst_RS_left.csv"
BURST_PROXIMITY_MS  = 100

ADC_CSV_PATH        = r"F:\s531\processed data from 531\Mat Data\E\CL testing\period3\ADC1.csv"

VERIFY_STIM_DELIVERY = True
MAX_VALID_DELAY_MS   = 200

# Which label to pull from each CSV (default: FIRE from CSV_A, SKIP from CSV_B)
# Set to None to use ALL events from that CSV as that type
CSV_A_LABEL        = 'FIRE'     # only keep rows labeled FIRE from CSV_A
CSV_B_LABEL        = 'SKIP'     # only keep rows labeled SKIP from CSV_B


def load_labeled_csv(csv_path):
    df = pd.read_csv(csv_path)
    time_s = df['time_s'].values
    labels = df['trig_label'].fillna('').astype(str).str.strip().str.upper().values
    proxy = _extract_proxy_col(df)
    return time_s, proxy, labels


def _extract_proxy_col(df):
    for col in ('proxy_feature', 'feature_value'):
        if col in df.columns:
            return df[col].values
    median_cols = [c for c in df.columns if c.endswith('_median_proxy')]
    if median_cols:
        print(f"  Using '{median_cols[0]}' as proxy column")
        return df[median_cols[0]].values
    raise ValueError(f"No proxy column found. Columns: {list(df.columns)}")


def load_burst_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df['burst_start_ms'].values / 1000.0, df['burst_end_ms'].values / 1000.0


def load_stim_periods(adc_csv_path):
    df = pd.read_csv(adc_csv_path, low_memory=False)
    time_col = stim_col = None
    for col in df.columns:
        if col.lower() in ('time_s', 'time', 'timestamp'): time_col = col
        if col.lower() in ('stimulationbool', 'stim', 'stim_on'): stim_col = col
    if time_col is None or stim_col is None:
        return None, None
    adc_time = pd.to_numeric(df[time_col], errors="coerce").values
    stim_bool = pd.to_numeric(df[stim_col], errors="coerce").fillna(0).astype(int).values != 0
    starts, ends, in_stim = [], [], False
    for i in range(len(stim_bool)):
        if stim_bool[i] and not in_stim:
            starts.append(adc_time[i]); in_stim = True
        elif not stim_bool[i] and in_stim:
            ends.append(adc_time[i]); in_stim = False
    if in_stim:
        ends.append(adc_time[-1])
    return np.array(starts), np.array(ends)


def extract_events_from_csv(time_s, proxy, labels, keep_label, assign_type):
    """Extract events matching keep_label, assign them as assign_type (FIRE/SKIP)."""
    events = []
    for i in range(len(labels)):
        if keep_label is None or labels[i] == keep_label:
            if labels[i] not in ('FIRE', 'SKIP') and keep_label is None:
                continue
            events.append({
                'type': assign_type, 'stim_on': assign_type == 'FIRE',
                'time_s': time_s[i], 'csv_index': i, 'proxy_feat': proxy[i],
            })
    events.sort(key=lambda x: x['time_s'])
    print(f"  {assign_type}: {len(events)} events")
    return events


def filter_events_by_burst(events, burst_starts_s, burst_ends_s, proximity_s):
    kept = []
    for ev in events:
        t = ev['time_s']
        for bs, be in zip(burst_starts_s, burst_ends_s):
            if t >= (bs - proximity_s) and t <= (be + proximity_s):
                kept.append(ev)
                break
    n_fire = sum(1 for e in kept if e['stim_on'])
    n_skip = sum(1 for e in kept if not e['stim_on'])
    print(f"  Burst filter: {len(kept)}/{len(events)} kept, FIRE: {n_fire}, SKIP: {n_skip}")
    return kept


def verify_stim_delivery(events, stim_starts_s, max_delay_s):
    if stim_starts_s is None or len(stim_starts_s) == 0:
        print("  WARNING: No stim data for verification, keeping all events.")
        return events
    stim_arr = np.array(stim_starts_s)
    removed = []
    kept = []
    for ev in events:
        if not ev['stim_on']:
            kept.append(ev)
            continue
        t = ev['time_s']
        idx = np.searchsorted(stim_arr, t, side='left')
        if idx < len(stim_arr):
            delay_s = stim_arr[idx] - t
            if delay_s <= max_delay_s:
                ev['actual_delay_ms'] = round(delay_s * 1000, 4)
                kept.append(ev)
            else:
                removed.append((t, delay_s * 1000))
        else:
            removed.append((t, np.nan))
    if removed:
        print(f"\n  Excluded FIRE events:")
        for t, delay in removed:
            if np.isnan(delay):
                print(f"    t={t:.3f}s — no ADC stim found after this event")
            else:
                print(f"    t={t:.3f}s — actual delay={delay:.1f}ms (>{max_delay_s*1000:.0f}ms)")
    n_fire = sum(1 for e in kept if e['stim_on'])
    n_skip = sum(1 for e in kept if not e['stim_on'])
    print(f"\n  Stim verification: removed {len(removed)} FIRE events (no stim within {max_delay_s*1000:.0f}ms)")
    print(f"  After filter — FIRE: {n_fire}, SKIP: {n_skip}")
    return kept


def compute_metrics_separate(events, time_s_a, proxy_a, time_s_b, proxy_b,
                             threshold, stim_duration_s, analysis_end_s,
                             burst_starts_s=None, burst_ends_s=None, proximity_s=0.1):
    """Compute metrics using each event's own CSV data."""
    n_bursts = len(burst_starts_s) if burst_starts_s is not None else 0

    for ev in events:
        t_cross = ev['time_s']
        idx = ev['csv_index']

        # Pick the correct time/proxy arrays
        if ev['stim_on']:
            time_s, proxy = time_s_a, proxy_a
        else:
            time_s, proxy = time_s_b, proxy_b

        delay_ms = DELAY_M * t_cross + DELAY_B
        delay_s = delay_ms / 1000.0
        extra_s = EXTRA_DELAY_MS / 1000.0
        total_offset_s = delay_s + stim_duration_s + extra_s
        ev['processing_delay_ms'] = round(delay_ms, 4)
        ev['extra_delay_ms'] = EXTRA_DELAY_MS
        ev['total_offset_s'] = round(total_offset_s, 6)

        post_start = t_cross + total_offset_s
        post_end = t_cross + analysis_end_s
        remaining_ms = (post_end - post_start) * 1000
        ev['analysis_window_ms'] = round(remaining_ms, 2)

        if post_end > post_start:
            post_mask = (time_s >= post_start) & (time_s <= post_end)
            post_vals = proxy[post_mask]
            ev['post_stim_mean'] = float(np.mean(post_vals)) if len(post_vals) > 0 else np.nan
        else:
            ev['post_stim_mean'] = np.nan

        trace_end = t_cross + analysis_end_s + (512.0 / 24414.0625)
        trace_mask = (time_s >= t_cross) & (time_s <= trace_end)
        ev['trace_t'] = time_s[trace_mask] - t_cross
        ev['trace_v'] = proxy[trace_mask]

        ev['offline_burst_duration_s'] = np.nan
        ev['time_to_next_offline_burst_s'] = np.nan

        if burst_starts_s is not None:
            burst_durations = burst_ends_s - burst_starts_s
            for i in range(n_bursts):
                if t_cross >= (burst_starts_s[i] - proximity_s) and t_cross <= (burst_ends_s[i] + proximity_s):
                    ev['offline_burst_duration_s'] = float(burst_durations[i])
                    break
            for i in range(n_bursts):
                if burst_starts_s[i] > t_cross + 0.001:
                    ev['time_to_next_offline_burst_s'] = float(burst_starts_s[i] - t_cross)
                    break

    # IBI computed within each group separately
    fire_evts = [e for e in events if e['stim_on']]
    skip_evts = [e for e in events if not e['stim_on']]
    for group in [fire_evts, skip_evts]:
        for i in range(len(group) - 1):
            group[i]['ibi_s'] = group[i + 1]['time_s'] - group[i]['time_s']
        if group:
            group[-1]['ibi_s'] = np.nan

    return events


def write_events_csv(events, output_path):
    fields = ['event_num', 'type', 'stim_on', 'time_s', 'proxy_feat',
              'processing_delay_ms', 'total_offset_s', 'analysis_window_ms',
              'ibi_s',
              'offline_burst_duration_s', 'time_to_next_offline_burst_s',
              'post_stim_mean']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for i, e in enumerate(events):
            def fmt(key):
                v = e.get(key, np.nan)
                if isinstance(v, float) and np.isnan(v): return ''
                return round(v, 6)
            w.writerow({
                'event_num': i + 1, 'type': e['type'], 'stim_on': e['stim_on'],
                'time_s': round(e['time_s'], 6), 'proxy_feat': round(e['proxy_feat'], 6),
                'processing_delay_ms': e['processing_delay_ms'],
                'total_offset_s': e['total_offset_s'],
                'analysis_window_ms': e.get('analysis_window_ms', ''),
                'ibi_s': fmt('ibi_s'),
                'offline_burst_duration_s': fmt('offline_burst_duration_s'),
                'time_to_next_offline_burst_s': fmt('time_to_next_offline_burst_s'),
                'post_stim_mean': fmt('post_stim_mean'),
            })
    print(f"  -> {output_path}")


def plot_postStim_Fast_Proxy(events, stim_duration_s, analysis_end_ms, threshold, output_path):
    on = [e for e in events if e['stim_on'] and len(e.get('trace_t', [])) > 0]
    off = [e for e in events if not e['stim_on'] and len(e.get('trace_t', [])) > 0]
    if not on or not off:
        print("    Not enough events for postStim Fast Proxy.")
        return

    median_delay_ms = float(np.median([e['processing_delay_ms'] for e in events]))
    extra_ms = EXTRA_DELAY_MS
    stim_end_ms = median_delay_ms + stim_duration_s * 1000
    extra_end_ms = stim_end_ms + extra_ms

    sample_dt_ms = 512.0 / 24414.0625 * 1000

    def collect_by_sample(evts):
        max_len = max(len(e['trace_v']) for e in evts)
        sample_vals = [[] for _ in range(max_len)]
        for e in evts:
            if len(e['trace_v']) < 2:
                continue
            for j, v in enumerate(e['trace_v']):
                sample_vals[j].append(v)
        return sample_vals

    on_raw = collect_by_sample(on)
    off_raw = collect_by_sample(off)
    n_samples = min(len(on_raw), len(off_raw))

    sample_times_ms = np.arange(n_samples) * sample_dt_ms

    on_m = np.array([np.mean(on_raw[i]) if on_raw[i] else np.nan for i in range(n_samples)])
    on_se = np.array([np.std(on_raw[i]) / np.sqrt(len(on_raw[i])) if len(on_raw[i]) > 1 else 0.0 for i in range(n_samples)])
    off_m = np.array([np.mean(off_raw[i]) if off_raw[i] else np.nan for i in range(n_samples)])
    off_se = np.array([np.std(off_raw[i]) / np.sqrt(len(off_raw[i])) if len(off_raw[i]) > 1 else 0.0 for i in range(n_samples)])

    first_analysis_idx = int(np.ceil(extra_end_ms / sample_dt_ms))

    bin_pvals = []
    for i in range(n_samples):
        if i < first_analysis_idx:
            bin_pvals.append(np.nan)
        elif len(on_raw[i]) >= 2 and len(off_raw[i]) >= 2:
            _, p = ttest_ind(on_raw[i], off_raw[i], equal_var=False)
            bin_pvals.append(p)
        else:
            bin_pvals.append(np.nan)
    bin_pvals = np.array(bin_pvals)

    def get_traces(evts):
        traces = []
        for e in evts:
            if len(e['trace_v']) >= 2:
                traces.append(e['trace_v'][:n_samples])
        return traces

    on_traces = get_traces(on)
    off_traces = get_traces(off)

    cum_pvals = []
    for i in range(n_samples):
        if i < first_analysis_idx:
            cum_pvals.append(np.nan)
            continue
        on_means = [np.mean(tr[first_analysis_idx:i+1]) for tr in on_traces if len(tr) > i]
        off_means = [np.mean(tr[first_analysis_idx:i+1]) for tr in off_traces if len(tr) > i]
        if len(on_means) >= 2 and len(off_means) >= 2:
            _, p = ttest_ind(on_means, off_means, equal_var=False)
            cum_pvals.append(p)
        else:
            cum_pvals.append(np.nan)
    cum_pvals = np.array(cum_pvals)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=['postStim Fast Proxy (mean ± SEM)',
                                        f'Per-sample p-value (from sample {first_analysis_idx}, {extra_end_ms:.0f}ms)',
                                        f'Cumulative p-value (samples {first_analysis_idx}..i)'])

    fig.add_trace(go.Scatter(x=np.concatenate([sample_times_ms, sample_times_ms[::-1]]),
        y=np.concatenate([on_m + on_se, (on_m - on_se)[::-1]]),
        fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=sample_times_ms, y=on_m, mode='lines+markers',
        line=dict(color='red', width=2), marker=dict(size=4), name=f'FIRE (n={len(on)})'), row=1, col=1)

    fig.add_trace(go.Scatter(x=np.concatenate([sample_times_ms, sample_times_ms[::-1]]),
        y=np.concatenate([off_m + off_se, (off_m - off_se)[::-1]]),
        fill='toself', fillcolor='rgba(0,0,255,0.1)', line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=sample_times_ms, y=off_m, mode='lines+markers',
        line=dict(color='blue', width=2), marker=dict(size=4), name=f'SKIP (n={len(off)})'), row=1, col=1)

    fig.add_hline(y=threshold, line_dash='dash', line_color='orange', line_width=1.5,
        annotation_text=f'Threshold: {threshold}', annotation_position='top right', row=1, col=1)

    fig.add_vrect(x0=0, x1=median_delay_ms, fillcolor='rgba(150,150,150,0.1)', line_width=0,
                  annotation_text=f'Delay (~{median_delay_ms:.0f}ms)', annotation_position='top left', row=1, col=1)
    fig.add_vrect(x0=median_delay_ms, x1=stim_end_ms, fillcolor='rgba(0,200,0,0.1)', line_width=0,
                  annotation_text=f'Stim ({stim_duration_s*1000:.0f}ms)', annotation_position='top left', row=1, col=1)
    if extra_ms > 0:
        fig.add_vrect(x0=stim_end_ms, x1=extra_end_ms, fillcolor='rgba(200,100,0,0.1)', line_width=0,
                      annotation_text=f'Extra ({extra_ms:.0f}ms)', annotation_position='top left', row=1, col=1)
    fig.add_vrect(x0=extra_end_ms, x1=analysis_end_ms, fillcolor='rgba(200,200,0,0.1)', line_width=0,
                  annotation_text='Analysis', annotation_position='top left', row=1, col=1)

    bar_width = sample_dt_ms * 0.85
    pval_colors = ['red' if p < 0.05 else 'gray' for p in bin_pvals]
    fig.add_trace(go.Bar(x=sample_times_ms, y=bin_pvals, marker_color=pval_colors,
        name='p-value', showlegend=False, width=bar_width), row=2, col=1)
    fig.add_hline(y=0.05, line_dash='dash', line_color='red', line_width=1.5,
        annotation_text='p = 0.05', annotation_position='top right', row=2, col=1)

    fig.add_vrect(x0=0, x1=median_delay_ms, fillcolor='rgba(150,150,150,0.1)', line_width=0, row=2, col=1)
    fig.add_vrect(x0=median_delay_ms, x1=stim_end_ms, fillcolor='rgba(0,200,0,0.1)', line_width=0, row=2, col=1)
    if extra_ms > 0:
        fig.add_vrect(x0=stim_end_ms, x1=extra_end_ms, fillcolor='rgba(200,100,0,0.1)', line_width=0, row=2, col=1)
    fig.add_vrect(x0=extra_end_ms, x1=analysis_end_ms, fillcolor='rgba(200,200,0,0.1)', line_width=0, row=2, col=1)

    cum_colors = ['red' if p < 0.05 else 'gray' for p in cum_pvals]
    fig.add_trace(go.Bar(x=sample_times_ms, y=cum_pvals, marker_color=cum_colors,
        name='cumulative p', showlegend=False, width=bar_width), row=3, col=1)
    fig.add_hline(y=0.05, line_dash='dash', line_color='red', line_width=1.5,
        annotation_text='p = 0.05', annotation_position='top right', row=3, col=1)

    fig.add_vrect(x0=0, x1=median_delay_ms, fillcolor='rgba(150,150,150,0.1)', line_width=0, row=3, col=1)
    fig.add_vrect(x0=median_delay_ms, x1=stim_end_ms, fillcolor='rgba(0,200,0,0.1)', line_width=0, row=3, col=1)
    if extra_ms > 0:
        fig.add_vrect(x0=stim_end_ms, x1=extra_end_ms, fillcolor='rgba(200,100,0,0.1)', line_width=0, row=3, col=1)
    fig.add_vrect(x0=extra_end_ms, x1=analysis_end_ms, fillcolor='rgba(200,200,0,0.1)', line_width=0, row=3, col=1)

    folder_name = os.path.basename(os.path.dirname(output_path))
    fig.update_layout(
        title=f'{folder_name} — postStim fastProxy 2CSV (per-sample & cumulative t-test)',
        template='plotly_white', height=900)
    fig.update_yaxes(title_text='Proxy Value', range=[50, 150], row=1, col=1)
    fig.update_yaxes(title_text='p-value', range=[0, 1], row=2, col=1)
    fig.update_yaxes(title_text='p-value', range=[0, 1], row=3, col=1)
    fig.update_xaxes(title_text='Time from crossing (ms)', row=3, col=1)

    fig.write_html(output_path)
    print(f"  -> {output_path}")

    print(f"\n  Per-sample t-test results (dt={sample_dt_ms:.2f}ms):")
    print(f"  {'Sample':>6}  {'Time (ms)':>10}  {'FIRE mean':>10}  {'SKIP mean':>10}  {'p-value':>10}  {'sig':>5}  {'cum p':>10}  {'cum sig':>7}  {'n_fire':>6}  {'n_skip':>6}")
    for i in range(n_samples):
        p = bin_pvals[i]
        cp = cum_pvals[i]
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."
        csig = "***" if cp<0.001 else "**" if cp<0.01 else "*" if cp<0.05 else "n.s."
        print(f"  {i:6d}  {sample_times_ms[i]:10.2f}  {on_m[i]:10.2f}  {off_m[i]:10.2f}  {p:10.4f}  {sig:>5}  {cp:10.4f}  {csig:>7}  {len(on_raw[i]):6d}  {len(off_raw[i]):6d}")


def plot_post_stim_boxplot(events, output_path):
    fire_vals = [e['post_stim_mean'] for e in events if e['stim_on'] and not np.isnan(e.get('post_stim_mean', np.nan))]
    skip_vals = [e['post_stim_mean'] for e in events if not e['stim_on'] and not np.isnan(e.get('post_stim_mean', np.nan))]
    fig = go.Figure()
    fig.add_trace(go.Box(y=fire_vals, name='FIRE', marker_color='red', boxmean=True))
    fig.add_trace(go.Box(y=skip_vals, name='SKIP', marker_color='blue', boxmean=True))
    folder_name = os.path.basename(os.path.dirname(output_path))
    fig.update_layout(
        title=f'{folder_name} — Post-Stim Mean Proxy: FIRE (n={len(fire_vals)}) vs SKIP (n={len(skip_vals)})',
        yaxis_title='Mean Proxy in Analysis Window',
        template='plotly_white', height=450)
    fig.write_html(output_path)
    print(f"  -> {output_path}")


def plot_ibi_boxplot(events, output_path):
    fire_ibi = [e['ibi_s'] for e in events if e['stim_on'] and not np.isnan(e.get('ibi_s', np.nan))]
    skip_ibi = [e['ibi_s'] for e in events if not e['stim_on'] and not np.isnan(e.get('ibi_s', np.nan))]
    fig = go.Figure()
    fig.add_trace(go.Box(y=fire_ibi, name='FIRE', marker_color='red', boxmean=True))
    fig.add_trace(go.Box(y=skip_ibi, name='SKIP', marker_color='blue', boxmean=True))
    for label, vals in [('FIRE', fire_ibi), ('SKIP', skip_ibi)]:
        if vals:
            arr = np.array(vals)
            print(f"  IBI {label}: mean={arr.mean():.3f}s, std={arr.std():.3f}s, median={np.median(arr):.3f}s, n={len(vals)}")
    folder_name = os.path.basename(os.path.dirname(output_path))
    fig.update_layout(
        title=f'{folder_name} — Inter-Burst Interval: FIRE (n={len(fire_ibi)}) vs SKIP (n={len(skip_ibi)})',
        yaxis_title='IBI (seconds)',
        template='plotly_white', height=450)
    fig.write_html(output_path)
    print(f"  -> {output_path}")


def main():
    if len(sys.argv) < 4:
        print("Usage: python stim_analysis_2csv.py <fire_csv> <skip_csv> <output_folder>")
        sys.exit(1)

    fire_csv = sys.argv[1]
    skip_csv = sys.argv[2]
    output_dir = sys.argv[3]
    os.makedirs(output_dir, exist_ok=True)

    stim_dur_s = STIM_DURATION_MS / 1000.0
    analysis_end_s = ANALYSIS_END_MS / 1000.0

    # Load CSV A (FIRE source)
    print(f"Loading CSV_A (FIRE): {fire_csv}")
    time_s_a, proxy_a, labels_a = load_labeled_csv(fire_csv)
    print(f"  {len(time_s_a)} samples, {time_s_a[0]:.2f} - {time_s_a[-1]:.2f} s")

    # Load CSV B (SKIP source)
    print(f"\nLoading CSV_B (SKIP): {skip_csv}")
    time_s_b, proxy_b, labels_b = load_labeled_csv(skip_csv)
    print(f"  {len(time_s_b)} samples, {time_s_b[0]:.2f} - {time_s_b[-1]:.2f} s")

    # Extract events from each
    print(f"\nExtracting FIRE events from CSV_A (label filter: {CSV_A_LABEL})...")
    fire_events = extract_events_from_csv(time_s_a, proxy_a, labels_a, CSV_A_LABEL, 'FIRE')

    print(f"Extracting SKIP events from CSV_B (label filter: {CSV_B_LABEL})...")
    skip_events = extract_events_from_csv(time_s_b, proxy_b, labels_b, CSV_B_LABEL, 'SKIP')

    if not fire_events and not skip_events:
        print("No events found in either CSV.")
        sys.exit(0)

    events = fire_events + skip_events

    # Load ADC stim windows
    stim_starts = stim_ends = None
    if ADC_CSV_PATH and os.path.isfile(ADC_CSV_PATH):
        print(f"\nLoading ADC stim data: {ADC_CSV_PATH}")
        stim_starts, stim_ends = load_stim_periods(ADC_CSV_PATH)
        if stim_starts is not None:
            print(f"  {len(stim_starts)} stim windows, range: {stim_starts[0]:.2f} - {stim_ends[-1]:.2f} s")
    else:
        print(f"\nADC CSV not found: {ADC_CSV_PATH}")

    # Verify FIRE events
    if VERIFY_STIM_DELIVERY and stim_starts is not None:
        print(f"\nVerifying stim delivery (max delay: {MAX_VALID_DELAY_MS}ms)...")
        events = verify_stim_delivery(events, stim_starts, MAX_VALID_DELAY_MS / 1000.0)
        if not [e for e in events if e['stim_on']]:
            print("No FIRE events remain after stim verification.")
            sys.exit(0)

    # Load offline bursts
    burst_starts = burst_ends = None
    if BURST_CSV_PATH and os.path.isfile(BURST_CSV_PATH):
        print(f"\nLoading offline bursts: {BURST_CSV_PATH}")
        burst_starts, burst_ends = load_burst_csv(BURST_CSV_PATH)
        print(f"  {len(burst_starts)} burst periods")
    else:
        print(f"\nBurst CSV not found: {BURST_CSV_PATH}")

    if BURST_FILTER_ENABLE and burst_starts is not None:
        proximity_s = BURST_PROXIMITY_MS / 1000.0
        events = filter_events_by_burst(events, burst_starts, burst_ends, proximity_s)
        if not events:
            print("No events remain after filtering.")
            sys.exit(0)

    # Compute metrics — each event uses its own CSV's time/proxy
    print("\nComputing metrics...")
    events = compute_metrics_separate(events, time_s_a, proxy_a, time_s_b, proxy_b,
                                       THRESHOLD, stim_dur_s, analysis_end_s,
                                       burst_starts_s=burst_starts, burst_ends_s=burst_ends,
                                       proximity_s=BURST_PROXIMITY_MS / 1000.0)

    fire_evts = [e for e in events if e['stim_on']]
    skip_evts = [e for e in events if not e['stim_on']]

    print(f"\n{'='*60}")
    print(f"  FIRE: {len(fire_evts)} (from CSV_A), SKIP: {len(skip_evts)} (from CSV_B)")
    print(f"{'='*60}")

    metrics = [
        ('post_stim_mean', 'Post-Stim Mean (analysis window)'),
        ('ibi_s', 'Inter-Burst Interval (s)'),
        ('offline_burst_duration_s', 'Offline Burst Duration (s)'),
        ('time_to_next_offline_burst_s', 'Time to Next Offline Burst (s)'),
    ]

    for key, label in metrics:
        fire_v = np.array([e[key] for e in fire_evts if not np.isnan(e.get(key, np.nan))])
        skip_v = np.array([e[key] for e in skip_evts if not np.isnan(e.get(key, np.nan))])
        if len(fire_v) < 2 or len(skip_v) < 2:
            print(f"\n  {label}: not enough data (FIRE n={len(fire_v)}, SKIP n={len(skip_v)})")
            continue
        fire_mean = np.mean(fire_v)
        skip_mean = np.mean(skip_v)
        pct = 100 * (fire_mean - skip_mean) / skip_mean if skip_mean != 0 else np.nan
        _, p_mw = mannwhitneyu(fire_v, skip_v, alternative='two-sided')
        _, p_tt = ttest_ind(fire_v, skip_v, equal_var=False)
        sig_mw = "***" if p_mw<0.001 else "**" if p_mw<0.01 else "*" if p_mw<0.05 else "n.s."
        sig_tt = "***" if p_tt<0.001 else "**" if p_tt<0.01 else "*" if p_tt<0.05 else "n.s."
        print(f"\n  {label}:")
        print(f"    FIRE mean={fire_mean:.4e} (n={len(fire_v)})")
        print(f"    SKIP mean={skip_mean:.4e} (n={len(skip_v)})")
        print(f"    Δ = {pct:+.1f}%")
        print(f"    Mann-Whitney U: p={p_mw:.4f} ({sig_mw})")
        print(f"    Welch's t-test: p={p_tt:.4f} ({sig_tt})")

    print(f"\n{'='*60}")

    write_events_csv(events, os.path.join(output_dir, 'stim_events.csv'))
    plot_postStim_Fast_Proxy(events, stim_dur_s, ANALYSIS_END_MS, THRESHOLD,
                            os.path.join(output_dir, 'postStim_Fast_Proxy.html'))
    plot_post_stim_boxplot(events, os.path.join(output_dir, 'post_stim_boxplot.html'))
    plot_ibi_boxplot(events, os.path.join(output_dir, 'ibi_boxplot.html'))
    print("\nDone!")


if __name__ == '__main__':
    main()