import pandas as pd
import numpy as np

# ============================================================
# USER SETTINGS
# ============================================================
ADC_CSV_PATH   = r"F:\s531\processed data from 531\Mat Data\E\CL testing\period2\ADC1.csv"
PROXY_CSV_PATH = r"G:\closed loop testing\recorded binary files\proxy_feature_record2.csv"

THRESHOLD = 90   # <-- set your threshold here
MIN_SEPARATION_S = 0.0  # optional: minimum time between proxy trigger events
SAVE_OUTPUT = False

OUTPUT_CSV_PATH = r"F:\closed loop testing\recorded binary files\delay_results_period2.csv"


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
                "delay_s": delay,
                "delay_ms": delay * 1000.0
            })
        else:
            results.append({
                "proxy_cross_time_s": proxy_t,
                "adc_rising_time_s": np.nan,
                "delay_s": np.nan,
                "delay_ms": np.nan
            })

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================
def main():
    # ---------------- Load ADC ----------------
    adc_df = pd.read_csv(ADC_CSV_PATH, low_memory=False)

    adc_time_col = find_column_case_insensitive(adc_df, ["time_s", "time", "timestamp"])
    stim_col = find_column_case_insensitive(adc_df, ["stimulation", "stim", "stim_on"])

    if adc_time_col is None:
        raise ValueError(f"Could not find ADC time column. Columns found: {list(adc_df.columns)}")
    if stim_col is None:
        raise ValueError(f"Could not find ADC stimulation column. Columns found: {list(adc_df.columns)}")

    adc_time = pd.to_numeric(adc_df[adc_time_col], errors="coerce")
    stim_bool = to_bool_series(adc_df[stim_col])

    adc_valid = adc_time.notna() & stim_bool.notna()
    adc_time = adc_time[adc_valid].reset_index(drop=True)
    stim_bool = stim_bool[adc_valid].reset_index(drop=True)

    adc_rise_times = rising_edge_times_from_bool(adc_time, stim_bool)

    # ---------------- Load Proxy ----------------
    proxy_df = pd.read_csv(PROXY_CSV_PATH, low_memory=False)

    proxy_time_col = find_column_case_insensitive(proxy_df, ["time_s", "time", "timestamp"])
    feature_col = find_column_case_insensitive(proxy_df, ["feature_value", "feature", "value"])

    if proxy_time_col is None:
        raise ValueError(f"Could not find proxy time column. Columns found: {list(proxy_df.columns)}")
    if feature_col is None:
        raise ValueError(f"Could not find feature value column. Columns found: {list(proxy_df.columns)}")

    proxy_time = pd.to_numeric(proxy_df[proxy_time_col], errors="coerce")
    feature_val = pd.to_numeric(proxy_df[feature_col], errors="coerce")

    proxy_valid = proxy_time.notna() & feature_val.notna()
    proxy_time = proxy_time[proxy_valid].reset_index(drop=True)
    feature_val = feature_val[proxy_valid].reset_index(drop=True)

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
    print(f"Proxy crossings found: {len(proxy_cross_times)}")
    print(f"ADC rising edges found: {len(adc_rise_times)}")
    print(f"Matched rows: {len(results_df)}")

    valid_delays = results_df["delay_ms"].dropna()
    if len(valid_delays) > 0:
        print("\nDelay statistics (proxy crossing -> next ADC rising edge):")
        print(f"Mean delay:   {valid_delays.mean():.3f} ms")
        print(f"Median delay: {valid_delays.median():.3f} ms")
        print(f"Min delay:    {valid_delays.min():.3f} ms")
        print(f"Max delay:    {valid_delays.max():.3f} ms")
    else:
        print("\nNo valid matched delays found.")

    print("\nAll matched events:")
    print(results_df.head(100000).to_string(index=False))

    # ---------------- Save ----------------
    if SAVE_OUTPUT:
        results_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\nSaved results to:\n{OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()