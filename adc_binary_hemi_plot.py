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
# ADC_CSV_PATH = r"D:\s531\processed data from 531\Mat Data\E\CL testing\period2\ADC1.csv"
ADC_CSV_PATH = r'/Volumes/D_Drive/s531_data/Day1/ClosedLoopTesting/recorded binary files/Mat Data/E/CL testing/period2/ADC1.csv'


# PROXY_CSV_PATH = r"D:\closed loop testing\recorded binary files\proxy_feature_record2.csv"
PROXY_CSV_PATH = r'/Volumes/D_Drive/s531_data/Day1/ClosedLoopTesting/recorded binary files/proxy_feature_record2.csv'

# SPIKETIME_MAT_PATH = r"D:\s531\processed data from 531\Mat Data\Z\CL testing\spikes_v4_varcluster_sameClusters\micro_CommonFiltered_0_01Hz\period2\SpikeClusters_3std_wav\spikeTime.mat"
SPIKETIME_MAT_PATH = r'/Volumes/D_Drive/s531_data/Day1/ClosedLoopTesting/recorded binary files/Mat Data/Z/CL testing/spikes_v4_varcluster_sameClusters/micro_CommonFiltered_0_01Hz/period2/SpikeClusters_3std_wav/spikeTime.mat'


BURST_RS_CSV_PATH = (
    '/Volumes/D_Drive/SangerLabBursts/outputs_RS_burst/day1_test/Period2/rankSurprise/'
    'separateGPi__SNR=1.2-25.0__FR=0.8Hz__aClust=8%__limClust=75__aReg=5%__limReg=75__aNet=3%__limNet=75__minSpk=3__minDur=0ms__minCh=0__region__network/'
    'network_bursts_RS_left.csv'
)

OUTPUT_FOLDER = r'/Volumes/D_Drive/s531_output/Day1/closed_test_p1-3_output/p2/plot'

THRESHOLD = 90
MIN_SNR = 1.2
MAX_SNR = 10000
SIDE_TO_PLOT = "L"   # "L", "R", or None

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
def require_file(path: str, label: str):
    if not os.path.isfile(path):
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


# =============================================================================
# INCLUDE-CHANNEL SUPPORT
# =============================================================================
def load_include_channels_py(path: str) -> dict[str, set[int]]:
    """
    Load include channels from a python file that defines:
      INCLUDE_CHANNELS = { "GPi1_L": [1,2,3], ... }

    Returns:
      dict like {"GPi1_L": {1,2,3}, ...}
    """
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
    """
    Parse electrode strings like:
      microGPi1_L_1
      microGPi1_L_1_CommonFiltered

    Returns:
      (region, side, channel) or None
    """
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

        # Side filter
        if side_to_plot is not None:
            if parsed is None or side != side_to_plot:
                continue

        # Include filter
        if include_enable:
            if parsed is None:
                continue

            include_key = f"{region}_{side}"
            include_set = include_map.get(include_key)

            # Mimic your fastProxy behavior:
            # if a region/side is not listed at all, skip it entirely
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
        # print(f"Loaded {len(units)} valid raster units")
    return units


def add_all_rasters_overlay(fig: go.Figure, units: list[dict[str, Any]]):
    x_all = []
    y_all = []
    text_all = []
    color_all = []

    for unit in units:
        x_s = unit["times_ms"] / 1000.0   # assumes spikeTime is in ms
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
# MAIN
# =============================================================================
def main():
    require_file(ADC_CSV_PATH, "ADC CSV")
    require_file(PROXY_CSV_PATH, "Proxy CSV")
    require_file(SPIKETIME_MAT_PATH, "SpikeTime MAT")
    require_file(BURST_RS_CSV_PATH, "Burst RS CSV")

    if INCLUDE_ENABLE:
        print(f"Include-channel filtering enabled: {INCLUDE_PY_PATH}")
    else:
        print("Include-channel filtering disabled")

    # ── ADC: stimulation only ────────────────────────────────────────────────
    adc_df = pd.read_csv(ADC_CSV_PATH, low_memory=False)
    # if "stimulation" not in adc_df.columns:
    #     raise ValueError(f'ADC CSV must contain "stimulation". Found: {list(adc_df.columns)}')

    if "time_s" in adc_df.columns:
        adc_time = pd.to_numeric(adc_df["time_s"], errors="coerce")
    else:
        adc_time = pd.to_numeric(adc_df.iloc[:, 0], errors="coerce")

    stim_bool = parse_bool_series(adc_df["stimulation"])

    adc_mask = adc_time.notna() & stim_bool.notna()
    adc_time = adc_time[adc_mask].to_numpy()
    stim_bool = stim_bool[adc_mask].to_numpy(dtype=bool)

    stim_intervals = compute_true_intervals(adc_time, stim_bool)

    # ── Burst RS intervals (burst_start_ms, burst_end_ms) ────────────────────────
    burst_df = pd.read_csv(BURST_RS_CSV_PATH, low_memory=False)
    if "burst_start_ms" not in burst_df.columns or "burst_end_ms" not in burst_df.columns:
        raise ValueError(
            f"Burst CSV must contain 'burst_start_ms' and 'burst_end_ms'. Found: {list(burst_df.columns)}"
        )
    burst_start_s = pd.to_numeric(burst_df["burst_start_ms"], errors="coerce") / 1000.0
    burst_end_s = pd.to_numeric(burst_df["burst_end_ms"], errors="coerce") / 1000.0
    burst_mask = burst_start_s.notna() & burst_end_s.notna() & (burst_start_s < burst_end_s)
    burst_intervals = list(zip(burst_start_s[burst_mask].to_numpy(), burst_end_s[burst_mask].to_numpy()))

    # ── Proxy ────────────────────────────────────────────────────────────────
    proxy_df = pd.read_csv(PROXY_CSV_PATH, low_memory=False)
    if "time_s" not in proxy_df.columns or "feature_value" not in proxy_df.columns:
        raise ValueError(
            f"Proxy CSV missing 'time_s' or 'feature_value'. Found: {list(proxy_df.columns)}"
        )

    proxy_time = pd.to_numeric(proxy_df["time_s"], errors="coerce")
    proxy_val = pd.to_numeric(proxy_df["feature_value"], errors="coerce")

    proxy_mask = proxy_time.notna() & proxy_val.notna()
    proxy_time = proxy_time[proxy_mask].to_numpy()
    proxy_val = proxy_val[proxy_mask].to_numpy()

    if proxy_time.size == 0 or proxy_val.size == 0:
        raise ValueError("Proxy CSV has no valid numeric data.")

    # ── Raster ───────────────────────────────────────────────────────────────
    units = load_valid_raster_units(
        SPIKETIME_MAT_PATH,
        MIN_SNR,
        max_snr=MAX_SNR,
        side_to_plot=SIDE_TO_PLOT,
        include_enable=INCLUDE_ENABLE,
        include_py_path=INCLUDE_PY_PATH,
    )
    if len(units) == 0:
        raise ValueError("No valid raster units found.")

    # Robust proxy range so big peaks do not squash the raster
    proxy_lo = float(np.nanpercentile(proxy_val, 1))
    proxy_hi = float(np.nanpercentile(proxy_val, 99))
    proxy_span = proxy_hi - proxy_lo
    if proxy_span <= 0:
        proxy_span = max(abs(proxy_hi), 1.0)

    raster_band_bottom = proxy_lo - 0.60 * proxy_span
    raster_band_top = proxy_lo - 0.10 * proxy_span

    if len(units) == 1:
        units[0]["y_plot"] = 0.5 * (raster_band_bottom + raster_band_top)
    else:
        y_positions = np.linspace(raster_band_bottom, raster_band_top, len(units))
        for unit, y in zip(units, y_positions):
            unit["y_plot"] = float(y)

    y_min = raster_band_bottom - 0.08 * proxy_span
    y_max = proxy_hi + 0.15 * proxy_span

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig = go.Figure()

    # Green shading where stimulation == TRUE
    for x0, x1 in stim_intervals:
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=STIM_FILL_COLOR,
            line_width=0,
            layer="below",
        )

    # Red shading for burst RS intervals
    for x0, x1 in burst_intervals:
        fig.add_vrect(
            x0=float(x0),
            x1=float(x1),
            fillcolor=BURST_FILL_COLOR,
            line_width=0,
            layer="below",
        )

    # Proxy line
    fig.add_trace(
        go.Scattergl(
            x=proxy_time,
            y=proxy_val,
            mode="lines",
            line=dict(width=1.5, color="steelblue"),
            name="proxy feature",
            hoverinfo="skip",
        )
    )

    # Raster overlay
    add_all_rasters_overlay(fig, units)

    # Threshold
    if THRESHOLD is not None:
        fig.add_hline(
            y=float(THRESHOLD),
            line=dict(color="black", width=1, dash="dash"),
        )

    # Separator above raster band
    fig.add_hline(
        y=raster_band_top,
        line=dict(color="rgba(100,100,100,0.4)", width=1, dash="dot"),
    )

    fig.update_layout(
        template="plotly_white",
        height=700,
        title="Proxy feature with stimulation shading and raster",
        xaxis_title="Time (s)",
        yaxis=dict(
            title=dict(text="Proxy feature"),
            range=[y_min, y_max],
        ),
        hovermode="closest",
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.7)",
        ),
    )

    stimchan_subdir = f"StimChan={INCLUDE_ENABLE}"
    out_dir = os.path.join(OUTPUT_FOLDER, stimchan_subdir)
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, "proxy_stimulation_shading_raster.html")
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"Saved {html_path}")

    fig.show()


if __name__ == "__main__":
    main()