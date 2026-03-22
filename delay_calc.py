import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ============================================================
# USER SETTINGS
# ============================================================
ADC_CSV_PATH   = r"F:\s531\processed data from 531\CL testing\macro\period8\ADC2.csv"
PROXY_CSV_PATH = r"F:\s531_binary\period8\period8_bin.csv"

THRESHOLD = 95  #p8 = 95, p10 = 65
MIN_SEPARATION_S = 0.0  # optional: minimum time between proxy trigger events
# ADC time_s += (proxy_end - 2*BIN_SIZE_S - adc_end) using last cleaned samples (shifts ADC on the time axis).
BIN_SIZE_S = 512/24414.0625
# After ADC shift: optionally shift proxy so its last sample matches ADC’s last sample.
ALIGN_TIMES_BY_END = False
SAVE_OUTPUT = True
OUTPUT_CSV_PATH = r"F:\s531_binary\period8\delay_results_period8.csv"

# Optional scatter: proxy_cross_time_s vs delay_ms (only delay_ms < threshold, non-NaN)
PLOT_DELAY_SCATTER = True
DELAY_PLOT_MAX_MS = 70.0
# If True, saves HTML next to OUTPUT_CSV_PATH (*_delay_scatter.html). Requires write permission.
PLOT_DELAY_SAVE_HTML = True
PLOT_DELAY_SHOW = True


# ============================================================
# HELPERS
# ============================================================
def find_column_case_insensitive(df, target_names):
    lower_map = {col.lower(): col for col in df.columns}
    for name in target_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def to_bool_series(series):
    """
    Robust conversion of stimulation column to boolean.
    Handles True/False, 1/0, strings, etc.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int) != 0

    s = series.astype(str).str.strip().str.lower()
    true_vals = {"true", "1", "high", "yes", "on"}
    return s.isin(true_vals)


def rising_edge_times_from_bool(time_s, bool_signal):
    """
    Return times where signal goes from False -> True.
    """
    prev = bool_signal.shift(1, fill_value=False)
    rising = (~prev) & bool_signal
    return time_s[rising].reset_index(drop=True)


def threshold_crossing_times(time_s, values, threshold, min_separation_s=0.0):
    """
    Return times where values cross from below threshold to >= threshold.
    Optionally suppress crossings that happen too close together.
    """
    above = values >= threshold
    prev_above = above.shift(1, fill_value=False)
    crossings = (~prev_above) & above

    crossing_times = time_s[crossings].reset_index(drop=True)

    if min_separation_s > 0 and len(crossing_times) > 0:
        kept = [crossing_times.iloc[0]]
        for t in crossing_times.iloc[1:]:
            if t - kept[-1] >= min_separation_s:
                kept.append(t)
        crossing_times = pd.Series(kept, name="time_s")

    return crossing_times.reset_index(drop=True)


def match_proxy_to_adc(proxy_times, adc_times):
    """
    For each proxy crossing, match to the next ADC rising edge.
    Delay = adc_time - proxy_time
    """
    results = []
    adc_arr = adc_times.to_numpy()

    for proxy_t in proxy_times:
        idx = np.searchsorted(adc_arr, proxy_t, side="left")
        if idx < len(adc_arr):
            adc_t = adc_arr[idx]
            delay = adc_t - proxy_t
            results.append({
                "proxy_cross_time_s": proxy_t,
                "adc_rising_time_s": adc_t,
                "delay_ms": delay * 1000.0,
            })
        else:
            results.append({
                "proxy_cross_time_s": proxy_t,
                "adc_rising_time_s": np.nan,
                "delay_ms": np.nan,
            })

    return pd.DataFrame(results)


def plot_proxy_time_vs_delay(results_df, max_delay_ms, save_html_path=None, show=True):
    """
    Scatter proxy_cross_time_s vs delay_ms for rows with valid delay_ms < max_delay_ms.
    """
    sub = results_df.dropna(subset=["delay_ms"])
    sub = sub[sub["delay_ms"] < max_delay_ms]
    if len(sub) == 0:
        print(f"\nDelay scatter: no points with delay_ms < {max_delay_ms} (and non-NaN).")
        return

    x_arr = sub["proxy_cross_time_s"].to_numpy(dtype=float)
    y_arr = sub["delay_ms"].to_numpy(dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=x_arr,
            y=y_arr,
            mode="markers",
            marker=dict(size=6, opacity=0.55),
            name="events",
        )
    )

    eq_str = None
    if len(sub) >= 2:
        m, b = np.polyfit(x_arr, y_arr, 1)
        eq_str = f"delay_ms = ({m:.6g}) × proxy_cross_time_s + ({b:.6g})"
        print(f"\nOLS linear fit on plotted points:\n  {eq_str}")
        x_line = np.array([x_arr.min(), x_arr.max()], dtype=float)
        y_line = m * x_line + b
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color="red", width=2),
                name="OLS fit",
            )
        )
    else:
        print("\nDelay scatter: need at least 2 points for y = m·x + b fit.")

    title = f"proxy_cross_time_s vs delay_ms (delay_ms < {max_delay_ms} ms, n={len(sub)})"
    if eq_str is not None:
        title += f"<br><sup style='color:#333'>{eq_str}</sup>"

    fig.update_layout(
        title=dict(text=title),
        template="plotly_white",
        xaxis_title="proxy_cross_time_s",
        yaxis_title="delay_ms",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if save_html_path is not None:
        try:
            fig.write_html(save_html_path, include_plotlyjs="cdn")
            print(f"\nSaved delay scatter plot to:\n{save_html_path}")
        except OSError as e:
            print(f"\nDelay scatter: could not save HTML ({e!r}).")

    if show:
        fig.show()


# ============================================================
# MAIN
# ============================================================
def main():
    # ---------------- Load ADC ----------------
    adc_df = pd.read_csv(ADC_CSV_PATH, low_memory=False)

    adc_time_col = find_column_case_insensitive(adc_df, ["time_s", "time", "timestamp"])
    stim_col = find_column_case_insensitive(adc_df, ["stimulationBool", "stim", "stim_on"])

    if adc_time_col is None:
        raise ValueError(f"Could not find ADC time column. Columns found: {list(adc_df.columns)}")
    if stim_col is None:
        raise ValueError(f"Could not find ADC stimulation column. Columns found: {list(adc_df.columns)}")

    adc_time = pd.to_numeric(adc_df[adc_time_col], errors="coerce")
    stim_bool = to_bool_series(adc_df[stim_col])

    adc_valid = adc_time.notna() & stim_bool.notna()
    adc_time = adc_time[adc_valid].reset_index(drop=True)
    stim_bool = stim_bool[adc_valid].reset_index(drop=True)

    # ---------------- Load Proxy ----------------
    proxy_df = pd.read_csv(PROXY_CSV_PATH, low_memory=False)

    proxy_time_col = find_column_case_insensitive(proxy_df, ["time_s", "time", "timestamp"])
    feature_col = find_column_case_insensitive(
        proxy_df, ["feature_value", "feature", "value", "proxy_feature"]
    )

    if proxy_time_col is None:
        raise ValueError(f"Could not find proxy time column. Columns found: {list(proxy_df.columns)}")
    if feature_col is None:
        raise ValueError(f"Could not find feature value column. Columns found: {list(proxy_df.columns)}")

    proxy_time = pd.to_numeric(proxy_df[proxy_time_col], errors="coerce")
    feature_val = pd.to_numeric(proxy_df[feature_col], errors="coerce")

    proxy_valid = proxy_time.notna() & feature_val.notna()
    proxy_time = proxy_time[proxy_valid].reset_index(drop=True)
    feature_val = feature_val[proxy_valid].reset_index(drop=True)

    if len(adc_time) == 0 or len(proxy_time) == 0:
        raise ValueError("Empty ADC or proxy time series after cleaning.")

    adc_end = float(adc_time.iloc[-1])
    proxy_end = float(proxy_time.iloc[-1])
    adc_time_shift_s = proxy_end - 2.0 * BIN_SIZE_S - adc_end
    adc_time = adc_time + adc_time_shift_s

    proxy_time_shift_s = 0.0
    if ALIGN_TIMES_BY_END:
        adc_end_after = float(adc_time.iloc[-1])
        proxy_end_now = float(proxy_time.iloc[-1])
        proxy_time_shift_s = adc_end_after - proxy_end_now
        proxy_time = proxy_time + proxy_time_shift_s

    adc_rise_times = rising_edge_times_from_bool(adc_time, stim_bool)

    proxy_cross_times = threshold_crossing_times(
        proxy_time,
        feature_val,
        THRESHOLD,
        min_separation_s=MIN_SEPARATION_S
    )

    # ---------------- Match events ----------------
    results_df = match_proxy_to_adc(proxy_cross_times, adc_rise_times)

    # ---------------- Print summary ----------------
    print("\n==================== SUMMARY ====================")
    print(f"ADC file:   {ADC_CSV_PATH}")
    print(f"Proxy file: {PROXY_CSV_PATH}")
    print(f"Threshold:  {THRESHOLD}")
    print(f"BIN_SIZE_S: {BIN_SIZE_S}")
    print(f"ADC time_s shift: +{adc_time_shift_s:.9g} s  (adc += proxy_end - 2*BIN_SIZE_S - adc_end)")
    if ALIGN_TIMES_BY_END:
        print(f"End alignment: proxy time_s shifted by +{proxy_time_shift_s:.9g} s (ends match)")
    print(f"Proxy crossings found: {len(proxy_cross_times)}")
    print(f"ADC rising edges found: {len(adc_rise_times)}")
    print(f"Matched rows: {len(results_df)}")

    valid_delays = results_df["delay_ms"].dropna()
  
    print("\nAll matched events:")
    print(results_df.head(100000).to_string(index=False))

    # ---------------- Save ----------------
    if SAVE_OUTPUT:
        try:
            results_df.to_csv(OUTPUT_CSV_PATH, index=False)
            print(f"\nSaved results to:\n{OUTPUT_CSV_PATH}")
        except OSError as e:
            print(f"\nCould not save CSV ({e!r}). Close the file if it is open in another app.")

    if PLOT_DELAY_SCATTER:
        html_path = None
        if PLOT_DELAY_SAVE_HTML:
            html_path = OUTPUT_CSV_PATH.replace(".csv", "_delay_scatter.html")
        plot_proxy_time_vs_delay(
            results_df,
            DELAY_PLOT_MAX_MS,
            save_html_path=html_path,
            show=PLOT_DELAY_SHOW,
        )


if __name__ == "__main__":
    main()