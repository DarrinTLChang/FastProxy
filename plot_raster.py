import os
import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.io import loadmat


# =============================================================================
# PATHS / SETTINGS
# =============================================================================
# ADC_CSV_PATH = r'/Users/darrin/Downloads/march 14 closed loop testing/ADCs/period6/ADC1.csv'
ADC_CSV_PATH = None


# Use one of these:
PROXY_CSV_PATH = None
# FASTPROXY_CSV_PATH = r"E:\s531_fp_output\Day5_baseline\p9\includeChannel=False\hemisphere_neo_binned.csv"
FASTPROXY_CSV_PATH = r"F:\s531_binary\period2_test\offline_sel\hemisphere_neo_binned.csv"

# '/Volumes/D_Drive/s531_fp_output/Day5_baseline/p9/includeChannel=True_VO/VA/GPi1/hemisphere_neo_binned_plot_full.html'
SPIKETIME_MAT_PATH = r"F:\s531\processed data from 531\Mat Data\Z\CL testing\spikes_v4_varcluster_sameClusters\micro_CommonFiltered_0_01Hz\period2\SpikeClusters_3std_wav\spikeTime.mat"

OUTPUT_FOLDER = (
    os.path.dirname(FASTPROXY_CSV_PATH) if FASTPROXY_CSV_PATH
    else (os.path.dirname(PROXY_CSV_PATH) if PROXY_CSV_PATH else os.path.join(os.path.dirname(__file__), "adc_binary_hemi_plot_output"))
)
BURST_RS_CSV_PATH = None
# (
#     '/Volumes/D_Drive/SangerLabBursts/outputs_RS_burst/day5_baseline/Period9/rankSurprise/separateGPi__SNR=1.2-1000__FR=0.8Hz__aClust=8%__limClust=75__aReg=5%__limReg=75__aNet=3%__limNet=75__minSpk=3__minDur=0ms__minCh=0__region__network/network_bursts_RS_left.csv'
# )


THRESHOLD = 90
MIN_SNR = 1.2
MAX_SNR = 10000
SIDE_TO_PLOT = "L"   # "L", "R", or None
FASTPROXY_SHIFT_S = 0.021  # shift FastProxy time to the right (seconds)

# Include-channel toggle
INCLUDE_ENABLE = False
INCLUDE_PY_PATH = os.path.join(os.path.dirname(__file__), "include_channels.py")

# Plot styling
RASTER_DOT_SIZE = 5
RASTER_OPACITY = 0.8
STIM_FILL_COLOR = "rgba(0, 200, 0, 0.18)"
BURST_FILL_COLOR = "rgba(200, 0, 0, 0.18)"


# =============================================================================
# HELPERS
# =============================================================================
def _opt_path(path: str | None) -> str | None:
    """None or string 'none'/'NONE' → None; otherwise return path as-is (if it's a path string)."""
    if path is None:
        return None
    if isinstance(path, str) and path.strip().lower() in ("none", ""):
        return None
    return path


def require_file(path: str, label: str):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found:\n{path}")


def color_for_nonburst(elec: str) -> str:
    if "_L_" in elec:
        return "royalblue"
    if "_R_" in elec:
        return "darkorange"
    return "black"


def ensure_list(x: Any) -> list:
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        if x.dtype == object:
            return list(x.ravel())
        return list(np.ravel(x))
    return [x]


def numeric_1d(x: Any) -> np.ndarray:
    if x is None:
        return np.array([], dtype=float)

    try:
        arr = np.asarray(x).squeeze()
    except Exception:
        return np.array([], dtype=float)

    if arr.size == 0:
        return np.array([], dtype=float)

    if arr.dtype != object:
        out = arr.astype(float).ravel()
        return out[np.isfinite(out)]

    vals = []
    for item in arr.ravel():
        try:
            sub = np.asarray(item).astype(float).ravel()
            sub = sub[np.isfinite(sub)]
            vals.extend(sub.tolist())
        except Exception:
            pass
    return np.asarray(vals, dtype=float)


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    s = series.astype(str).str.strip().str.lower()
    return s.isin({"true", "1", "t", "yes", "y"})


def compute_true_intervals(time_s: np.ndarray, stim_bool: np.ndarray) -> list[tuple[float, float]]:
    if time_s.size == 0 or stim_bool.size == 0 or time_s.size != stim_bool.size:
        return []

    if time_s.size == 1:
        return [(float(time_s[0]), float(time_s[0]))] if stim_bool[0] else []

    dt = np.diff(time_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    median_dt = float(np.median(dt)) if dt.size > 0 else 0.0

    stim_i = stim_bool.astype(np.int8)
    padded = np.r_[0, stim_i, 0]
    changes = np.diff(padded)

    start_idx = np.where(changes == 1)[0]
    end_idx = np.where(changes == -1)[0] - 1

    intervals = []
    for s_idx, e_idx in zip(start_idx, end_idx):
        x0 = float(time_s[s_idx])
        x1 = float(time_s[e_idx + 1]) if e_idx + 1 < len(time_s) else float(time_s[e_idx] + median_dt)
        intervals.append((x0, x1))

    return intervals


def find_column_case_insensitive(df: pd.DataFrame, names: list[str]):
    lower_map = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


# =============================================================================
# INCLUDE-CHANNEL SUPPORT
# =============================================================================
def load_include_channels_py(path: str) -> dict[str, set[int]]:
    if not path or not os.path.isfile(path):
        return {}

    import importlib.util

    spec = importlib.util.spec_from_file_location("include_channels", path)
    if spec is None or spec.loader is None:
        return {}

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    raw = getattr(mod, "INCLUDE_CHANNELS", None)
    if not isinstance(raw, dict):
        return {}

    include_map: dict[str, set[int]] = {}
    for key, chans in raw.items():
        if not key or not chans:
            continue
        s = set()
        for ch in chans:
            try:
                s.add(int(ch))
            except Exception:
                continue
        if s:
            include_map[str(key)] = s

    return include_map


def parse_electrode_name(elec: str):
    if elec is None:
        return None

    elec = str(elec).strip()
    m = re.match(r"^micro([A-Za-z0-9]+)_([LR])_(\d+)(?:_CommonFiltered)?$", elec)
    if not m:
        return None

    region, side, channel = m.groups()
    return region, side, int(channel)


# =============================================================================
# RASTER LOADING
# =============================================================================
def load_valid_raster_units(
    spiketime_mat_path: str,
    min_snr: float,
    max_snr: float = np.inf,
    side_to_plot: str | None = None,
    include_enable: bool = False,
    include_py_path: str | None = None,
) -> list[dict[str, Any]]:
    mat = loadmat(spiketime_mat_path, simplify_cells=True)

    if "spikeTime" not in mat:
        raise KeyError(f"'spikeTime' not found in {spiketime_mat_path}")

    spike_time = mat["spikeTime"]

    if isinstance(spike_time, dict):
        channels = [spike_time]
    elif isinstance(spike_time, list):
        channels = spike_time
    elif isinstance(spike_time, np.ndarray):
        channels = list(spike_time.ravel())
    else:
        raise TypeError(f"Unexpected spikeTime type: {type(spike_time)}")

    include_map = load_include_channels_py(include_py_path) if include_enable else {}
    units = []

    for ch_idx, ch in enumerate(channels):
        if not isinstance(ch, dict):
            continue

        elec = str(ch.get("electrode", f"channel_{ch_idx + 1}"))

        parsed = parse_electrode_name(elec)
        if parsed is None:
            if include_enable:
                continue
            region = None
            side = None
            channel_num = None
        else:
            region, side, channel_num = parsed

        if side_to_plot is not None:
            if parsed is None or side != side_to_plot:
                continue

        if include_enable:
            if parsed is None:
                continue

            include_key = f"{region}_{side}"
            include_set = include_map.get(include_key)

            if not include_set:
                continue

            if channel_num not in include_set:
                continue

        times_list = ensure_list(ch.get("time", []))
        snr_vals = numeric_1d(ch.get("snr", []))

        n_clusters = max(len(times_list), len(snr_vals))
        if n_clusters == 0:
            continue

        for cl_idx in range(n_clusters):
            snr = snr_vals[cl_idx] if cl_idx < len(snr_vals) else np.nan
            if not np.isfinite(snr) or snr < min_snr or snr > max_snr:
                continue
            if cl_idx >= len(times_list):
                continue

            times = numeric_1d(times_list[cl_idx])
            if times.size == 0:
                continue

            units.append({
                "electrode": elec,
                "cluster": cl_idx + 1,
                "times_ms": times,
            })

    return units


def add_all_rasters_overlay(fig: go.Figure, units: list[dict[str, Any]]):
    x_all = []
    y_all = []
    text_all = []
    color_all = []

    for unit in units:
        x_s = unit["times_ms"] / 1000.0
        label = f'{unit["electrode"]} C{unit["cluster"]}'
        color = color_for_nonburst(unit["electrode"])

        x_all.extend(x_s.tolist())
        y_all.extend([unit["y_plot"]] * len(x_s))
        text_all.extend([label] * len(x_s))
        color_all.extend([color] * len(x_s))

    if not x_all:
        return

    fig.add_trace(
        go.Scattergl(
            x=np.asarray(x_all),
            y=np.asarray(y_all),
            mode="markers",
            text=text_all,
            hovertemplate="%{text}<br>t=%{x:.3f}s<extra></extra>",
            marker=dict(
                symbol="line-ns-open",
                size=RASTER_DOT_SIZE,
                line=dict(width=1),
                color=color_all,
            ),
            opacity=RASTER_OPACITY,
            showlegend=False,
            name="raster",
        )
    )


# =============================================================================
# PROXY / FASTPROXY LOADING
# =============================================================================
def load_signal_data(side: str | None):
    """
    Returns a dict:
      {
        "source_name": "...",
        "signals": [{"side": "L"/"R"/None, "time": np.ndarray, "value": np.ndarray, "label": str}, ...]
      }
    """
    if FASTPROXY_CSV_PATH:
        require_file(FASTPROXY_CSV_PATH, "FastProxy CSV")
        df = pd.read_csv(FASTPROXY_CSV_PATH, low_memory=False)

        time_col = find_column_case_insensitive(df, ["time_s", "time", "timestamp"])
        if time_col is None:
            raise ValueError(f"FastProxy CSV missing time column. Found: {list(df.columns)}")

        left_col = find_column_case_insensitive(
            df,
            ["hemisphere_l_median_proxy", "hemisphere_l", "left_median_proxy", "left"]
        )
        right_col = find_column_case_insensitive(
            df,
            ["hemisphere_r_median_proxy", "hemisphere_r", "right_median_proxy", "right"]
        )

        if left_col is None and right_col is None:
            raise ValueError(
                f"FastProxy CSV missing hemisphere columns. Found: {list(df.columns)}"
            )

        time_s = pd.to_numeric(df[time_col], errors="coerce")
        signals = []

        if side in (None, "L") and left_col is not None:
            val = pd.to_numeric(df[left_col], errors="coerce")
            mask = time_s.notna() & val.notna()
            if mask.any():
                signals.append({
                    "side": "L",
                    "time": time_s[mask].to_numpy() + FASTPROXY_SHIFT_S,
                    "value": val[mask].to_numpy(),
                    "label": left_col,
                })

        if side in (None, "R") and right_col is not None:
            val = pd.to_numeric(df[right_col], errors="coerce")
            mask = time_s.notna() & val.notna()
            if mask.any():
                signals.append({
                    "side": "R",
                    "time": time_s[mask].to_numpy() + FASTPROXY_SHIFT_S,
                    "value": val[mask].to_numpy(),
                    "label": right_col,
                })

        return {"source_name": os.path.basename(FASTPROXY_CSV_PATH), "signals": signals}

    if PROXY_CSV_PATH:
        require_file(PROXY_CSV_PATH, "Proxy CSV")
        df = pd.read_csv(PROXY_CSV_PATH, low_memory=False)

        time_col = find_column_case_insensitive(df, ["time_s", "time", "timestamp"])
        feat_col = find_column_case_insensitive(df, ["feature_value", "feature", "value"])

        if time_col is None or feat_col is None:
            raise ValueError(
                f"Proxy CSV missing time/value columns. Found: {list(df.columns)}"
            )

        time_s = pd.to_numeric(df[time_col], errors="coerce")
        val = pd.to_numeric(df[feat_col], errors="coerce")
        mask = time_s.notna() & val.notna()

        signals = []
        if mask.any():
            signals.append({
                "side": side,
                "time": time_s[mask].to_numpy(),
                "value": val[mask].to_numpy(),
                "label": feat_col,
            })

        return {"source_name": os.path.basename(PROXY_CSV_PATH), "signals": signals}

    raise ValueError("Set either PROXY_CSV_PATH or FASTPROXY_CSV_PATH.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    adc_path = _opt_path(ADC_CSV_PATH)
    proxy_path = _opt_path(PROXY_CSV_PATH)
    fastproxy_path = _opt_path(FASTPROXY_CSV_PATH)
    spiketime_path = _opt_path(SPIKETIME_MAT_PATH)
    burst_path = _opt_path(BURST_RS_CSV_PATH)

    if not proxy_path and not fastproxy_path:
        raise ValueError("Set at least one of PROXY_CSV_PATH or FASTPROXY_CSV_PATH (or use 'none' for others).")
    if proxy_path and fastproxy_path:
        raise ValueError("Set only one of PROXY_CSV_PATH or FASTPROXY_CSV_PATH, not both.")

    if fastproxy_path:
        require_file(fastproxy_path, "FastProxy CSV")
    if proxy_path:
        require_file(proxy_path, "Proxy CSV")
    if spiketime_path:
        require_file(spiketime_path, "SpikeTime MAT")
    if adc_path:
        require_file(adc_path, "ADC CSV")
    if burst_path:
        require_file(burst_path, "Burst RS CSV")

    if INCLUDE_ENABLE:
        print(f"Include-channel filtering enabled: {INCLUDE_PY_PATH}")
    else:
        print("Include-channel filtering disabled")

    # ── ADC (optional) ─────────────────────────────────────────────────────
    stim_intervals = []
    if adc_path:
        adc_df = pd.read_csv(adc_path, low_memory=False)
        if "time_s" in adc_df.columns:
            adc_time = pd.to_numeric(adc_df["time_s"], errors="coerce")
        else:
            adc_time = pd.to_numeric(adc_df.iloc[:, 0], errors="coerce")
        stim_col = find_column_case_insensitive(adc_df, ["stimulation", "stim", "stim_on"])
        if stim_col is None:
            raise ValueError(f'ADC CSV must contain a stimulation column. Found: {list(adc_df.columns)}')
        stim_bool = parse_bool_series(adc_df[stim_col])
        adc_mask = adc_time.notna() & stim_bool.notna()
        adc_time = adc_time[adc_mask].to_numpy()
        stim_bool = stim_bool[adc_mask].to_numpy(dtype=bool)
        stim_intervals = compute_true_intervals(adc_time, stim_bool)

    # ── Burst RS intervals (optional) ──────────────────────────────────────
    burst_intervals = []
    if burst_path:
        burst_df = pd.read_csv(burst_path, low_memory=False)
        if "burst_start_ms" not in burst_df.columns or "burst_end_ms" not in burst_df.columns:
            raise ValueError(
                f"Burst CSV must contain 'burst_start_ms' and 'burst_end_ms'. Found: {list(burst_df.columns)}"
            )
        burst_start_s = pd.to_numeric(burst_df["burst_start_ms"], errors="coerce") / 1000.0
        burst_end_s = pd.to_numeric(burst_df["burst_end_ms"], errors="coerce") / 1000.0
        burst_mask = burst_start_s.notna() & burst_end_s.notna() & (burst_start_s < burst_end_s)
        burst_intervals = list(zip(burst_start_s[burst_mask].to_numpy(), burst_end_s[burst_mask].to_numpy()))

    # ── Signal source: proxy or fastproxy ─────────────────────────────────
    signal_pack = load_signal_data(SIDE_TO_PLOT)
    signals = signal_pack["signals"]

    if len(signals) == 0:
        raise ValueError("No valid signal columns found to plot.")

    # ── Plot each available side separately ───────────────────────────────
    for sig in signals:
        side = sig["side"]
        proxy_time = sig["time"]
        proxy_val = sig["value"]
        signal_label = sig["label"]

        units = []
        if spiketime_path:
            raster_side = side if side in ("L", "R") else SIDE_TO_PLOT
            units = load_valid_raster_units(
                spiketime_path,
                MIN_SNR,
                max_snr=MAX_SNR,
                side_to_plot=raster_side,
                include_enable=INCLUDE_ENABLE,
                include_py_path=INCLUDE_PY_PATH,
            )
            if len(units) == 0 and spiketime_path:
                print(f"Skipping side {side}: no valid raster units found.")
                continue

        proxy_lo = float(np.nanpercentile(proxy_val, 1))
        proxy_hi = float(np.nanpercentile(proxy_val, 99))
        proxy_span = proxy_hi - proxy_lo
        if proxy_span <= 0:
            proxy_span = max(abs(proxy_hi), 1.0)

        if units:
            raster_band_bottom = proxy_lo - 0.60 * proxy_span
            raster_band_top = proxy_lo - 0.10 * proxy_span
            if len(units) == 1:
                units[0]["y_plot"] = 0.5 * (raster_band_bottom + raster_band_top)
            else:
                y_positions = np.linspace(raster_band_bottom, raster_band_top, len(units))
                for unit, y in zip(units, y_positions):
                    unit["y_plot"] = float(y)
            y_min = raster_band_bottom - 0.08 * proxy_span
            y_max = proxy_hi + 2.20 * proxy_span
        else:
            y_min = proxy_lo - 0.15 * proxy_span
            y_max = proxy_hi + 2.20 * proxy_span

        fig = go.Figure()

        for x0, x1 in stim_intervals:
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=STIM_FILL_COLOR,
                line_width=0,
                layer="below",
            )

        for x0, x1 in burst_intervals:
            fig.add_vrect(
                x0=float(x0),
                x1=float(x1),
                fillcolor=BURST_FILL_COLOR,
                line_width=0,
                layer="below",
            )

        fig.add_trace(
            go.Scattergl(
                x=proxy_time,
                y=proxy_val,
                mode="lines",
                line=dict(width=1.5, color="steelblue"),
                name=signal_label,
                hoverinfo="skip",
            )
        )

        if units:
            add_all_rasters_overlay(fig, units)

        if THRESHOLD is not None:
            fig.add_hline(
                y=float(THRESHOLD),
                line=dict(color="black", width=1, dash="dash"),
            )

        if units:
            raster_band_top = proxy_lo - 0.10 * proxy_span
            fig.add_hline(
                y=raster_band_top,
                line=dict(color="rgba(100,100,100,0.4)", width=1, dash="dot"),
            )

        title_side = f" side={side}" if side in ("L", "R") else ""
        fig.update_layout(
            template="plotly_white",
            height=500,
            title=f"Signal with stimulation shading and raster{title_side}",
            xaxis=dict(
                title="Time (s)",
                # range=[0, 150],
            ),
            yaxis=dict(
                title=dict(text=signal_label),
                range=[y_min, y_max],
            ),
            hovermode="closest",
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor="rgba(255,255,255,0.7)",
            ),
        )

        out_dir = os.path.join(OUTPUT_FOLDER)
        os.makedirs(out_dir, exist_ok=True)

        side_suffix = f"_{side}" if side in ("L", "R") else ""
        html_path = os.path.join(out_dir, f"signal_stimulation_shading_raster{side_suffix}.html")
        fig.write_html(html_path, include_plotlyjs="cdn")
        print(f"Saved {html_path}")

        fig.show()


if __name__ == "__main__":
    main()