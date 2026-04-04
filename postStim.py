"""
Stimulation Effect Analysis — reads the labeled proxy CSV
(with trig_label column from add_trig_labels.py).

Outputs: stim_events.csv, postStim_Fast_Proxy.html, post_stim_boxplot.html, full_trace.html

Usage:
  python stim_analysis.py <labeled_proxy_csv> <output_folder>
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
STIM_DURATION_MS   = 20
ANALYSIS_END_MS    = 250

DELAY_M            = 0.022
DELAY_B            = 5
EXTRA_DELAY_MS     = 40

REFRACTORY_WINDOW_MS = 250    # how far after crossing to count re-crossings (ms)

BURST_FILTER_ENABLE = False
BURST_CSV_PATH      = r"C:\Users\Maral\Downloads\day5_test_p8_network_burst_RS_left.csv"
BURST_PROXIMITY_MS  = 100

ADC_CSV_PATH        = r"F:\s531\processed data from 531\CL testing\macro\period7\ADC2.csv"

ALT_PROXY_CSV       = None

VERIFY_STIM_DELIVERY = True
MAX_VALID_DELAY_MS   = 200

SPIKETIME_MAT_PATH  = "F:\s531\processed data from 531\CL testing\spikes_v4_fixedClusters\micro_AverageFiltered\period8\SpikeClusters_3std_wav\spikeTime.mat"
MIN_SNR             = 1.2
MAX_SNR             = 10000
SIDE_TO_PLOT        = "L"
RASTER_DOT_SIZE     = 5
RASTER_OPACITY      = 0.8


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


def load_alt_proxy_csv(csv_path):
    df = pd.read_csv(csv_path)
    time_s = df['time_s'].values
    proxy = _extract_proxy_col(df)
    return time_s, proxy


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


def extract_events(time_s, proxy, labels):
    events = []
    for i in range(len(labels)):
        if labels[i] in ('FIRE', 'SKIP'):
            events.append({
                'type': labels[i], 'stim_on': labels[i] == 'FIRE',
                'time_s': time_s[i], 'csv_index': i, 'proxy_feat': proxy[i],
            })
    events.sort(key=lambda x: x['time_s'])
    n_fire = sum(1 for e in events if e['stim_on'])
    n_skip = sum(1 for e in events if not e['stim_on'])
    print(f"  FIRE: {n_fire}, SKIP: {n_skip}, Total: {len(events)}")
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


def compute_metrics(events, time_s, proxy, threshold,
                    stim_duration_s, analysis_end_s,
                    burst_starts_s=None, burst_ends_s=None, proximity_s=0.1):
    n_bursts = len(burst_starts_s) if burst_starts_s is not None else 0

    for ev in events:
        t_cross = ev['time_s']
        idx = ev['csv_index']

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

        # Count upward threshold crossings in refractory window
        # and measure peak proxy amplitude of each re-burst
        # Window: from total_offset (after delay+stim+extra) to REFRACTORY_WINDOW_MS from crossing
        refrac_start = t_cross + total_offset_s
        refrac_end = t_cross + REFRACTORY_WINDOW_MS / 1000.0
        refrac_mask = (time_s >= refrac_start) & (time_s <= refrac_end)
        refrac_vals = proxy[refrac_mask]
        n_crossings = 0
        crossing_peaks = []
        if len(refrac_vals) > 1:
            below = refrac_vals[:-1] < threshold
            above = refrac_vals[1:] >= threshold
            crossing_indices = np.where(below & above)[0]
            n_crossings = len(crossing_indices)
            # For each upward crossing, find peak before it drops back below threshold
            for ci in crossing_indices:
                peak = refrac_vals[ci + 1]
                for j in range(ci + 1, len(refrac_vals)):
                    if refrac_vals[j] > peak:
                        peak = refrac_vals[j]
                    if refrac_vals[j] < threshold:
                        break
                crossing_peaks.append(peak)
        ev['refractory_crossings'] = n_crossings
        ev['refractory_mean_peak'] = float(np.mean(crossing_peaks)) if crossing_peaks else np.nan

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

    for i in range(len(events) - 1):
        events[i]['ibi_s'] = events[i + 1]['time_s'] - events[i]['time_s']
    if events:
        events[-1]['ibi_s'] = np.nan

    return events


def write_events_csv(events, output_path):
    fields = ['event_num', 'type', 'stim_on', 'time_s', 'proxy_feat',
              'processing_delay_ms', 'total_offset_s', 'analysis_window_ms',
              'ibi_s',
              'offline_burst_duration_s', 'time_to_next_offline_burst_s',
              'post_stim_mean', 'refractory_crossings', 'refractory_mean_peak']
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
                'refractory_crossings': e.get('refractory_crossings', ''),
                'refractory_mean_peak': fmt('refractory_mean_peak'),
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
                        subplot_titles=['postStim Fast Proxy (mean +/- SEM)',
                                        f'Per-sample p-value (from sample {first_analysis_idx}, {extra_end_ms:.0f}ms)',
                                        f'Cumulative p-value (samples {first_analysis_idx}..i)'])

    # FIRE: only plot from 1 sample before analysis period
    fire_start = max(0, first_analysis_idx - 1)
    on_x = sample_times_ms[fire_start:]
    on_m_plot = on_m[fire_start:]
    on_se_plot = on_se[fire_start:]

    fig.add_trace(go.Scatter(x=np.concatenate([on_x, on_x[::-1]]),
        y=np.concatenate([on_m_plot + on_se_plot, (on_m_plot - on_se_plot)[::-1]]),
        fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(width=0), mode='lines', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=on_x, y=on_m_plot, mode='lines',
        line=dict(color='red', width=2), name=f'FIRE (n={len(on)})'), row=1, col=1)

    # SKIP: plot all samples
    fig.add_trace(go.Scatter(x=np.concatenate([sample_times_ms, sample_times_ms[::-1]]),
        y=np.concatenate([off_m + off_se, (off_m - off_se)[::-1]]),
        fill='toself', fillcolor='rgba(0,0,255,0.1)', line=dict(width=0), mode='lines', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=sample_times_ms, y=off_m, mode='lines',
        line=dict(color='blue', width=2), name=f'SKIP (n={len(off)})'), row=1, col=1)

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
        title=f'{folder_name} — postStim fastProxy (per-sample & cumulative t-test)',
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


def plot_individual_traces(events, stim_duration_s, analysis_end_ms, threshold, output_path):
    """Spaghetti plot: FIRE on top, SKIP on bottom, separate subplots."""
    on = [e for e in events if e['stim_on'] and len(e.get('trace_t', [])) > 0]
    off = [e for e in events if not e['stim_on'] and len(e.get('trace_t', [])) > 0]
    if not on and not off:
        print("    Not enough events for individual traces.")
        return

    median_delay_ms = float(np.median([e['processing_delay_ms'] for e in events]))
    extra_ms = EXTRA_DELAY_MS
    stim_end_ms = median_delay_ms + stim_duration_s * 1000
    extra_end_ms = stim_end_ms + extra_ms

    sample_dt_ms = 512.0 / 24414.0625 * 1000
    first_analysis_idx = int(np.ceil(extra_end_ms / sample_dt_ms))
    fire_start_idx = max(0, first_analysis_idx - 1)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[f'FIRE (n={len(on)})', f'SKIP (n={len(off)})'])

    # FIRE: start 1 sample before analysis period
    for i, e in enumerate(on):
        t_ms = e['trace_t'] * 1000
        fig.add_trace(go.Scattergl(x=t_ms[fire_start_idx:], y=e['trace_v'][fire_start_idx:], mode='lines',
            line=dict(color='rgba(255,0,0,0.2)', width=1),
            showlegend=False, hoverinfo='skip'), row=1, col=1)

    # SKIP: plot all points
    for i, e in enumerate(off):
        t_ms = e['trace_t'] * 1000
        fig.add_trace(go.Scattergl(x=t_ms, y=e['trace_v'], mode='lines',
            line=dict(color='rgba(0,0,255,0.2)', width=1),
            showlegend=False, hoverinfo='skip'), row=2, col=1)

    for row in [1, 2]:
        fig.add_hline(y=threshold, line_dash='dash', line_color='orange', line_width=1.5, row=row, col=1)
        fig.add_vrect(x0=0, x1=median_delay_ms, fillcolor='rgba(150,150,150,0.1)', line_width=0,
                      annotation_text=f'Delay (~{median_delay_ms:.0f}ms)',
                      annotation_position='top left', row=row, col=1)
        fig.add_vrect(x0=median_delay_ms, x1=stim_end_ms, fillcolor='rgba(0,200,0,0.1)', line_width=0,
                      annotation_text=f'Stim ({stim_duration_s*1000:.0f}ms)',
                      annotation_position='top left', row=row, col=1)
        if extra_ms > 0:
            fig.add_vrect(x0=stim_end_ms, x1=extra_end_ms, fillcolor='rgba(200,100,0,0.1)', line_width=0,
                          annotation_text=f'Extra Delay ({extra_ms:.0f}ms)',
                          annotation_position='top left', row=row, col=1)
        fig.add_vrect(x0=extra_end_ms, x1=analysis_end_ms, fillcolor='rgba(200,200,0,0.1)', line_width=0,
                      annotation_text='Analysis',
                      annotation_position='top left', row=row, col=1)

    folder_name = os.path.basename(os.path.dirname(output_path))
    fig.update_layout(
        title=f'{folder_name} — Individual Event Traces',
        template='plotly_white', height=800)
    fig.update_yaxes(title_text='Proxy Value', range=[50, 700], row=1, col=1)
    fig.update_yaxes(title_text='Proxy Value', range=[50, 700], row=2, col=1)
    fig.update_xaxes(title_text='Time from crossing (ms)', row=2, col=1)

    fig.write_html(output_path)
    print(f"  -> {output_path}")


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
        template='plotly_white', height=700)
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
        template='plotly_white', height=700)
    fig.write_html(output_path)
    print(f"  -> {output_path}")


def plot_refractory_crossings(events, output_path):
    fire_counts = [e['refractory_crossings'] for e in events if e['stim_on']]
    skip_counts = [e['refractory_crossings'] for e in events if not e['stim_on']]
    fire_peaks = [e['refractory_mean_peak'] for e in events if e['stim_on'] and not np.isnan(e.get('refractory_mean_peak', np.nan))]
    skip_peaks = [e['refractory_mean_peak'] for e in events if not e['stim_on'] and not np.isnan(e.get('refractory_mean_peak', np.nan))]

    folder_name = os.path.basename(os.path.dirname(output_path))

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        f'Re-crossing Count ({REFRACTORY_WINDOW_MS}ms)',
        f'Re-crossing Peak Amplitude ({REFRACTORY_WINDOW_MS}ms)'])

    fig.add_trace(go.Box(y=fire_counts, name='FIRE', marker_color='red', boxmean=True), row=1, col=1)
    fig.add_trace(go.Box(y=skip_counts, name='SKIP', marker_color='blue', boxmean=True), row=1, col=1)

    fig.add_trace(go.Box(y=fire_peaks, name='FIRE', marker_color='red', boxmean=True, showlegend=False), row=1, col=2)
    fig.add_trace(go.Box(y=skip_peaks, name='SKIP', marker_color='blue', boxmean=True, showlegend=False), row=1, col=2)

    for label, vals in [('FIRE', fire_counts), ('SKIP', skip_counts)]:
        if vals:
            arr = np.array(vals)
            print(f"  Refractory crossings {label}: mean={arr.mean():.2f}, median={np.median(arr):.0f}, n={len(vals)}")
    for label, vals in [('FIRE', fire_peaks), ('SKIP', skip_peaks)]:
        if vals:
            arr = np.array(vals)
            print(f"  Refractory peak amp {label}: mean={arr.mean():.2f}, median={np.median(arr):.2f}, n={len(vals)}")

    fig.update_yaxes(title_text='Number of Upward Crossings', row=1, col=1)
    fig.update_yaxes(title_text='Peak Proxy Value', row=1, col=2)
    fig.update_layout(
        title=f'{folder_name} — Refractory Analysis: FIRE (n={len(fire_counts)}) vs SKIP (n={len(skip_counts)})',
        template='plotly_white', height=700)
    fig.write_html(output_path)
    print(f"  -> {output_path}")


def plot_full_trace(time_s, proxy, events, threshold, output_path,
                    burst_starts_s=None, burst_ends_s=None,
                    stim_starts_s=None, stim_ends_s=None):
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=time_s, y=proxy, mode='lines',
        line=dict(color='steelblue', width=0.5), name='Proxy'))
    fig.add_hline(y=threshold, line_dash='dash', line_color='orange', line_width=1.5,
        annotation_text=f'Threshold: {threshold:.2f}', annotation_position='top right')
    if burst_starts_s is not None and burst_ends_s is not None and len(burst_starts_s) > 0:
        for bs, be in zip(burst_starts_s, burst_ends_s):
            fig.add_shape(type='rect', x0=bs, x1=be, y0=0, y1=1, yref='paper',
                          fillcolor='rgba(255,0,0,0.12)', line_width=0)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(size=10, color='rgba(255,0,0,0.3)', symbol='square'),
            name=f'Offline bursts ({len(burst_starts_s)})'))
    if stim_starts_s is not None and stim_ends_s is not None and len(stim_starts_s) > 0:
        for ss, se in zip(stim_starts_s, stim_ends_s):
            fig.add_shape(type='rect', x0=ss, x1=se, y0=0, y1=1, yref='paper',
                          fillcolor='rgba(0,180,0,0.15)', line_width=0)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(size=10, color='rgba(0,180,0,0.4)', symbol='square'),
            name=f'Stim delivered ({len(stim_starts_s)})'))
    for label, color, sym, filt in [('FIRE', 'red', 'triangle-up', True), ('SKIP', 'blue', 'triangle-down', False)]:
        evts = [e for e in events if e['stim_on'] == filt]
        fig.add_trace(go.Scatter(x=[e['time_s'] for e in evts],
            y=[proxy[e['csv_index']] for e in evts], mode='markers',
            marker=dict(color=color, size=8, symbol=sym), name=f'{label} ({len(evts)})'))
    folder_name = os.path.basename(os.path.dirname(output_path))
    fig.update_layout(title=f'{folder_name} — Full Proxy Trace with FIRE/SKIP Events',
        xaxis_title='Time (s)', yaxis_title='Proxy Value', template='plotly_white', height=500,
        yaxis=dict(range=[50, 500]))
    fig.write_html(output_path)
    print(f"  -> {output_path}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python stim_analysis.py <labeled_proxy_csv> <output_folder>")
        sys.exit(1)

    proxy_csv = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    stim_dur_s = STIM_DURATION_MS / 1000.0
    analysis_end_s = ANALYSIS_END_MS / 1000.0

    print(f"Loading labels: {proxy_csv}")
    time_s, proxy, labels = load_labeled_csv(proxy_csv)
    print(f"  {len(time_s)} samples, {time_s[-1]:.1f}s")
    print(f"  Proxy time range: {time_s[0]:.2f} - {time_s[-1]:.2f} s")

    if ALT_PROXY_CSV and os.path.isfile(ALT_PROXY_CSV):
        print(f"\nLoading alt proxy: {ALT_PROXY_CSV}")
        alt_time_s, alt_proxy = load_alt_proxy_csv(ALT_PROXY_CSV)
        print(f"  {len(alt_time_s)} samples, {alt_time_s[-1]:.1f}s")
        print("\nExtracting events (labels from main CSV)...")
        events = extract_events(time_s, proxy, labels)
        print("  Remapping events to alt proxy time grid...")
        for ev in events:
            idx = np.searchsorted(alt_time_s, ev['time_s'], side='left')
            if idx >= len(alt_time_s):
                idx = len(alt_time_s) - 1
            ev['csv_index'] = idx
            ev['proxy_feat'] = alt_proxy[idx]
        time_s = alt_time_s
        proxy = alt_proxy
    else:
        if ALT_PROXY_CSV:
            print(f"\nAlt proxy CSV not found: {ALT_PROXY_CSV}, using main CSV")
        print("\nExtracting events...")
        events = extract_events(time_s, proxy, labels)
    if not events:
        print("No FIRE/SKIP events found.")
        sys.exit(0)

    burst_starts = burst_ends = None
    if BURST_CSV_PATH and os.path.isfile(BURST_CSV_PATH):
        print(f"\nLoading offline bursts: {BURST_CSV_PATH}")
        burst_starts, burst_ends = load_burst_csv(BURST_CSV_PATH)
        print(f"  {len(burst_starts)} burst periods, range: {burst_starts[0]:.2f} - {burst_ends[-1]:.2f} s")
    else:
        print(f"\nBurst CSV not found: {BURST_CSV_PATH}")

    stim_starts = stim_ends = None
    if ADC_CSV_PATH and os.path.isfile(ADC_CSV_PATH):
        print(f"\nLoading ADC stim data: {ADC_CSV_PATH}")
        stim_starts, stim_ends = load_stim_periods(ADC_CSV_PATH)
        if stim_starts is not None:
            print(f"  {len(stim_starts)} stim windows, range: {stim_starts[0]:.2f} - {stim_ends[-1]:.2f} s")
    else:
        print(f"\nADC CSV not found: {ADC_CSV_PATH}")

    if VERIFY_STIM_DELIVERY and stim_starts is not None:
        print(f"\nVerifying stim delivery (max delay: {MAX_VALID_DELAY_MS}ms)...")
        events = verify_stim_delivery(events, stim_starts, MAX_VALID_DELAY_MS / 1000.0)
        if not events:
            print("No events remain after stim verification.")
            sys.exit(0)

    if BURST_FILTER_ENABLE and burst_starts is not None:
        proximity_s = BURST_PROXIMITY_MS / 1000.0
        events = filter_events_by_burst(events, burst_starts, burst_ends, proximity_s)
        if not events:
            print("No events remain after filtering.")
            sys.exit(0)

    print("\nComputing metrics...")
    events = compute_metrics(events, time_s, proxy, THRESHOLD, stim_dur_s, analysis_end_s,
                              burst_starts_s=burst_starts, burst_ends_s=burst_ends,
                              proximity_s=BURST_PROXIMITY_MS / 1000.0)

    fire_evts = [e for e in events if e['stim_on']]
    skip_evts = [e for e in events if not e['stim_on']]

    print(f"\n{'='*60}")
    print(f"  FIRE: {len(fire_evts)}, SKIP: {len(skip_evts)}")
    print(f"{'='*60}")

    metrics = [
        ('post_stim_mean', 'Post-Stim Mean (analysis window)'),
        ('ibi_s', 'Inter-Burst Interval (s)'),
        ('refractory_crossings', f'Threshold Re-crossings (0-{REFRACTORY_WINDOW_MS}ms)'),
        ('refractory_mean_peak', f'Re-crossing Mean Peak Amplitude (0-{REFRACTORY_WINDOW_MS}ms)'),
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
    plot_individual_traces(events, stim_dur_s, ANALYSIS_END_MS, THRESHOLD,
                          os.path.join(output_dir, 'postStim_FastProxy_indivLines.html'))
    plot_post_stim_boxplot(events, os.path.join(output_dir, 'post_stim_boxplot.html'))
    plot_ibi_boxplot(events, os.path.join(output_dir, 'ibi_boxplot.html'))
    plot_refractory_crossings(events, os.path.join(output_dir, 'refractory_crossings.html'))
    plot_full_trace(time_s, proxy, events, THRESHOLD, os.path.join(output_dir, 'full_trace.html'),
                    burst_starts_s=burst_starts, burst_ends_s=burst_ends,
                    stim_starts_s=stim_starts, stim_ends_s=stim_ends)
    print("\nDone!")


if __name__ == '__main__':
    main()