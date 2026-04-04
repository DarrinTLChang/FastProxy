import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ============================================================
# USER SETTINGS
# ============================================================
ADC_CSV_PATH   = r"F:\s531\processed data from 531\CL testing\macro\period7\ADC2.csv"
# ADC_CSV_PATH   = r"F:\s531\processed data from 531\Mat Data\E\CL testing\period3\ADC1.csv"
PROXY_CSV_PATH = r"F:\s531_binary\period7\test_proxy_with_labels.csv"
# PROXY_CSV_PATH = r"F:\darrin_USB_drive\s531_data\Day1\ClosedLoopTesting\recorded binary files\proxy_feature_record3.csv"
SAVE_OUTPUT = True
OUTPUT_CSV_PATH = r"F:\s531_binary\period7\delay_result_no_align.csv"

PLOT_DELAY_SCATTER = True
DELAY_PLOT_MAX_MS = 200
PLOT_DELAY_SAVE_HTML = True
PLOT_DELAY_SAVE_POINTS_CSV = True
PLOT_DELAY_SHOW = True


# ============================================================
# HELPERS
# ============================================================

def find_col(df, names):
    lower_map = {col.lower(): col for col in df.columns}
    for n in names:
        if n.lower() in lower_map:
            return lower_map[n.lower()]
    return None


def to_bool_series(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int) != 0
    s = series.astype(str).str.strip().str.lower()
    return s.isin({"true", "1", "high", "yes", "on"})


def rising_edge_times(time_s, bool_signal):
    prev = bool_signal.shift(1, fill_value=False)
    return time_s[(~prev) & bool_signal].reset_index(drop=True)


def plot_delay_scatter(results_df, max_ms, save_html=None, save_csv=None, show=True):
    sub = results_df.dropna(subset=["delay_ms"])
    sub = sub[sub["delay_ms"] < max_ms]
    if len(sub) == 0:
        print(f"\nNo points with delay_ms < {max_ms}.")
        return

    if save_csv:
        sub.to_csv(save_csv, index=False)
        print(f"\nSaved scatter points ({len(sub)} rows): {save_csv}")

    x = sub["proxy_cross_time_s"].to_numpy(dtype=float)
    y = sub["delay_ms"].to_numpy(dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=x, y=y, mode="markers",
                                marker=dict(size=6, opacity=0.55), name="events"))

    eq = None
    if len(sub) >= 2:
        m, b = np.polyfit(x, y, 1)
        eq = f"delay_ms = ({m:.6g}) × t + ({b:.6g})"
        print(f"\nOLS fit: {eq}")
        xl = np.array([x.min(), x.max()])
        fig.add_trace(go.Scatter(x=xl, y=m*xl+b, mode="lines",
                                  line=dict(color="red", width=2), name="OLS fit"))

    title = f"FIRE crossing time vs delay (< {max_ms} ms, n={len(sub)})"
    if eq:
        title += f"<br><sup>{eq}</sup>"

    fig.update_layout(title=title, template="plotly_white",
                      xaxis_title="proxy_cross_time_s", yaxis_title="delay_ms", height=520)

    if save_html:
        fig.write_html(save_html, include_plotlyjs="cdn")
        print(f"Saved plot: {save_html}")
    if show:
        fig.show()


# ============================================================
# MAIN
# ============================================================

def main():
    # --- Load labeled proxy CSV ---
    print(f"Loading proxy CSV: {PROXY_CSV_PATH}")
    proxy_df = pd.read_csv(PROXY_CSV_PATH, low_memory=False)

    time_col = find_col(proxy_df, ["time_s", "time", "timestamp"])
    label_col = find_col(proxy_df, ["trig_label"])

    if time_col is None:
        raise ValueError(f"No time column found. Columns: {list(proxy_df.columns)}")
    if label_col is None:
        raise ValueError(f"No trig_label column found. Run add_trig_labels.py first.")

    proxy_time = pd.to_numeric(proxy_df[time_col], errors="coerce")

    # Get FIRE rows only
    fire_mask = proxy_df[label_col].astype(str).str.strip().str.upper() == 'FIRE'
    fire_times = proxy_time[fire_mask].reset_index(drop=True)
    print(f"  FIRE events: {len(fire_times)}")

    # --- Load ADC ---
    print(f"Loading ADC: {ADC_CSV_PATH}")
    adc_df = pd.read_csv(ADC_CSV_PATH, low_memory=False)

    adc_time_col = find_col(adc_df, ["time_s", "time", "timestamp"])
    stim_col = find_col(adc_df, ["stimulationBool", "stim", "stim_on"])

    adc_time = pd.to_numeric(adc_df[adc_time_col], errors="coerce")
    stim_bool = to_bool_series(adc_df[stim_col])
    valid = adc_time.notna() & stim_bool.notna()
    adc_time = adc_time[valid].reset_index(drop=True)
    stim_bool = stim_bool[valid].reset_index(drop=True)

    # --- NO alignment — raw timestamps as-is ---

    # --- ADC rising edges ---
    adc_rises = rising_edge_times(adc_time, stim_bool)
    print(f"  ADC rising edges: {len(adc_rises)}")

    # --- Match each FIRE to next ADC rising edge ---
    adc_arr = adc_rises.to_numpy()
    results = []
    for t in fire_times:
        idx = np.searchsorted(adc_arr, t, side="left")
        if idx < len(adc_arr):
            results.append({
                "proxy_cross_time_s": t,
                "adc_rising_time_s": adc_arr[idx],
                "delay_ms": (adc_arr[idx] - t) * 1000.0,
            })
        else:
            results.append({
                "proxy_cross_time_s": t,
                "adc_rising_time_s": np.nan,
                "delay_ms": np.nan,
            })

    results_df = pd.DataFrame(results)

    # --- Summary ---
    print(f"\n==================== SUMMARY ====================")
    print(f"Proxy CSV:      {PROXY_CSV_PATH}")
    print(f"ADC CSV:        {ADC_CSV_PATH}")
    print(f"Alignment:      NONE (raw timestamps)")
    print(f"FIRE events:    {len(fire_times)}")
    print(f"ADC rises:      {len(adc_rises)}")
    print(f"Matched rows:   {len(results_df)}")

    valid_delays = results_df["delay_ms"].dropna()
    if len(valid_delays) > 0:
        print(f"\nDelay stats:")
        print(f"  Mean:   {valid_delays.mean():.2f} ms")
        print(f"  Median: {valid_delays.median():.2f} ms")
        print(f"  Min:    {valid_delays.min():.2f} ms")
        print(f"  Max:    {valid_delays.max():.2f} ms")

    print("\nAll matched events:")
    print(results_df.head(100000).to_string(index=False))

    # --- Save ---
    if SAVE_OUTPUT:
        try:
            results_df.to_csv(OUTPUT_CSV_PATH, index=False)
            print(f"\nSaved: {OUTPUT_CSV_PATH}")
        except OSError as e:
            print(f"\nCould not save ({e!r}).")

    if PLOT_DELAY_SCATTER:
        html = OUTPUT_CSV_PATH.replace(".csv", "_delay_scatter.html") if PLOT_DELAY_SAVE_HTML else None
        pts = OUTPUT_CSV_PATH.replace(".csv", "_delay_scatter_points.csv") if PLOT_DELAY_SAVE_POINTS_CSV else None
        plot_delay_scatter(results_df, DELAY_PLOT_MAX_MS, save_html=html, save_csv=pts, show=PLOT_DELAY_SHOW)


if __name__ == "__main__":
    main()