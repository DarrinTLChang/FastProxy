import pandas as pd
import numpy as np
import os


# ============================================================
# USER SETTINGS
# ============================================================
FASTPROXY_CSV_PATH = r'/Volumes/D_Drive/s531_fp_output/Day4_test/p4/includeChannel=True/hemisphere_neo_binned.csv'
BURST_CSV_PATH = r'/Volumes/D_Drive/SangerLabBursts/outputs_RS_burst/day4_test/Period4/rankSurprise/separateGPi__SNR=1.2-1000__FR=0.8Hz__aClust=8%__limClust=75__aReg=5%__limReg=75__aNet=3%__limNet=75__minSpk=3__minDur=0ms__minCh=0__region__network/network_bursts_RS_left.csv'

HEMISPHERE = "L"          # "L" or "R"
THRESHOLD = 90
MIN_SEPARATION_S = 0.0
MAX_MATCH_DELAY_S = 1.0   # set None to disable max delay filtering

SAVE_OUTPUT = True
OUTPUT_CSV_PATH = r'/Volumes/D_Drive/s531_fp_output/Day4_test/p4/includeChannel=True'


# ============================================================
# HELPERS
# ============================================================
def find_column_case_insensitive(df, target_names):
    lower_map = {col.lower(): col for col in df.columns}
    for name in target_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def load_burst_start_times(burst_csv_path):
    if os.path.basename(burst_csv_path).startswith("._"):
        raise ValueError(f"Refusing to read macOS metadata file: {burst_csv_path}")

    burst_df = pd.read_csv(burst_csv_path, low_memory=False)

    burst_col = find_column_case_insensitive(burst_df, ["burst_start_ms", "burst_start"])
    if burst_col is None:
        raise ValueError(f"Could not find burst start column. Columns found: {list(burst_df.columns)}")

    burst_times = pd.to_numeric(burst_df[burst_col], errors="coerce") / 1000.0
    burst_times = burst_times.dropna().reset_index(drop=True)
    return burst_times


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


def load_fastproxy_crossing_times(fastproxy_csv_path, hemisphere, threshold, min_separation_s=0.0):
    df = pd.read_csv(fastproxy_csv_path, low_memory=False)

    time_col = find_column_case_insensitive(df, ["time_s", "time", "timestamp"])
    if time_col is None:
        raise ValueError(f"Could not find FastProxy time column. Columns found: {list(df.columns)}")

    hemisphere = hemisphere.upper()
    if hemisphere == "L":
        signal_col = find_column_case_insensitive(
            df,
            ["hemisphere_l_median_proxy", "hemisphere_l", "left_median_proxy", "left"]
        )
    elif hemisphere == "R":
        signal_col = find_column_case_insensitive(
            df,
            ["hemisphere_r_median_proxy", "hemisphere_r", "right_median_proxy", "right"]
        )
    else:
        raise ValueError("HEMISPHERE must be 'L' or 'R'")

    if signal_col is None:
        raise ValueError(f"Could not find hemisphere {hemisphere} column. Columns found: {list(df.columns)}")

    time_s = pd.to_numeric(df[time_col], errors="coerce")
    values = pd.to_numeric(df[signal_col], errors="coerce")

    valid = time_s.notna() & values.notna()
    time_s = time_s[valid].reset_index(drop=True)
    values = values[valid].reset_index(drop=True)

    crossing_times = threshold_crossing_times(
        time_s,
        values,
        threshold,
        min_separation_s=min_separation_s
    )

    return crossing_times, signal_col


def match_burst_to_events(burst_times, event_times, max_match_delay_s=None):
    """
    For each burst start, match to the next event time.
    Delay = event_time - burst_time
    """
    results = []
    event_arr = event_times.to_numpy()

    for burst_t in burst_times:
        idx = np.searchsorted(event_arr, burst_t, side="left")

        if idx < len(event_arr):
            event_t = event_arr[idx]
            delay = event_t - burst_t

            if max_match_delay_s is not None and delay > max_match_delay_s:
                results.append({
                    "burst_start_time_s": burst_t,
                    "fastproxy_cross_time_s": np.nan,
                    "delay_s": np.nan,
                    "delay_ms": np.nan,
                    "matched": False,
                    "reason": f"next_crossing_too_far_gt_{max_match_delay_s}s"
                })
            else:
                results.append({
                    "burst_start_time_s": burst_t,
                    "fastproxy_cross_time_s": event_t,
                    "delay_s": delay,
                    "delay_ms": delay * 1000.0,
                    "matched": True,
                    "reason": "matched_next_fastproxy_crossing"
                })
        else:
            results.append({
                "burst_start_time_s": burst_t,
                "fastproxy_cross_time_s": np.nan,
                "delay_s": np.nan,
                "delay_ms": np.nan,
                "matched": False,
                "reason": "no_fastproxy_crossing_after_burst"
            })

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================
def main():
    # ---------------- Load Burst Times ----------------
    burst_times = load_burst_start_times(BURST_CSV_PATH)

    # ---------------- Load FastProxy crossings ----------------
    cross_times, used_signal_col = load_fastproxy_crossing_times(
        FASTPROXY_CSV_PATH,
        hemisphere=HEMISPHERE,
        threshold=THRESHOLD,
        min_separation_s=MIN_SEPARATION_S
    )

    # ---------------- Match events ----------------
    results_df = match_burst_to_events(
        burst_times,
        cross_times,
        max_match_delay_s=MAX_MATCH_DELAY_S
    )

    # ---------------- Print summary ----------------
    print("\n==================== SUMMARY ====================")
    print(f"FastProxy file: {FASTPROXY_CSV_PATH}")
    print(f"Burst file:     {BURST_CSV_PATH}")
    print(f"Hemisphere:     {HEMISPHERE}")
    print(f"Signal column:  {used_signal_col}")
    print(f"Threshold:      {THRESHOLD}")
    print(f"Burst starts found:        {len(burst_times)}")
    print(f"FastProxy crossings found: {len(cross_times)}")
    print(f"Matched rows:              {len(results_df)}")

    valid_delays = results_df.loc[results_df["matched"], "delay_ms"].dropna()
    if len(valid_delays) > 0:
        print("\nDelay statistics (burst start -> next FastProxy threshold crossing):")
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