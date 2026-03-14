import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# Paths (edit if needed)
ADC_CSV_PATH = r"F:\s531\processed data from 531\Mat Data\E\CL testing\period2\ADC1.csv"
PROXY_CSV_PATH = r"G:\closed loop testing\recorded binary files\proxy_feature_record2.csv"
HEMI_CSV_PATH = r"C:\Users\Maral\Desktop\Darrin\FastProxy\s531_output\V7_day1Channels_531_Period2_validation\hemisphere_neo_binned.csv"

# Set to a number (e.g. 1e7) to draw the line, or None to disable.
THRESHOLD = 90


def main():
    # ── Load ADC ground-truth CSV ─────────────────────────────────────────────
    adc_df = pd.read_csv(ADC_CSV_PATH, low_memory=False)
    if adc_df.shape[1] < 2:
        raise ValueError(f"ADC CSV must have at least 2 columns. Got: {list(adc_df.columns)}")

    # First column is time_s
    adc_time = pd.to_numeric(adc_df.iloc[:, 0], errors="coerce")
    # Prefer explicit 'value' column if present; otherwise take second column
    if "value" in adc_df.columns:
        adc_val_raw = adc_df["value"]
    else:
        adc_val_raw = adc_df.iloc[:, 1]
    adc_val = pd.to_numeric(adc_val_raw, errors="coerce")

    adc_mask = adc_time.notna() & adc_val.notna()
    adc_time = adc_time[adc_mask].to_numpy()
    adc_val = adc_val[adc_mask].to_numpy()

    # ── Load proxy feature CSV ───────────────────────────────────────────────
    proxy_df = pd.read_csv(PROXY_CSV_PATH, low_memory=False)
    if "time_s" not in proxy_df.columns or "feature_value" not in proxy_df.columns:
        raise ValueError(
            f"Proxy CSV missing 'time_s' and 'feature_value'. Found: {list(proxy_df.columns)}"
        )

    proxy_time = pd.to_numeric(proxy_df["time_s"], errors="coerce")
    proxy_val = pd.to_numeric(proxy_df["feature_value"], errors="coerce")
    proxy_mask = proxy_time.notna() & proxy_val.notna()
    proxy_time = proxy_time[proxy_mask].to_numpy()
    proxy_val = proxy_val[proxy_mask].to_numpy()

    # ── Load hemisphere CSV (will be overlaid with proxy) ─────────────────────
    hemi_df = pd.read_csv(HEMI_CSV_PATH, low_memory=False)
    if "time_s" not in hemi_df.columns:
        raise ValueError(f"Hemisphere CSV missing 'time_s'. Columns: {list(hemi_df.columns)}")
    # Prefer explicit hemisphere_L_median_proxy column if present; else use last column
    if "hemisphere_L_median_proxy" in hemi_df.columns:
        hemi_y_raw = hemi_df["hemisphere_L_median_proxy"]
        hemi_label = "hemisphere median proxy (L)"
    else:
        hemi_y_raw = hemi_df.iloc[:, -1]
        hemi_label = str(hemi_df.columns[-1])

    hemi_t = pd.to_numeric(hemi_df["time_s"], errors="coerce")
    hemi_y = pd.to_numeric(hemi_y_raw, errors="coerce")
    hemi_mask = hemi_t.notna() & hemi_y.notna()
    hemi_t = hemi_t[hemi_mask].to_numpy()
    hemi_y = hemi_y[hemi_mask].to_numpy()

    # ── Single figure: proxy + hemisphere on primary axis, ADC on secondary ───
    fig = go.Figure()

    # Proxy on primary y-axis
    fig.add_trace(
        go.Scattergl(
            x=proxy_time,
            y=proxy_val,
            mode="lines",
            line=dict(width=1.5, color="steelblue"),
            name="proxy feature (binary)",
        ),
    )

    # Hemisphere median proxy on primary y-axis
    fig.add_trace(
        go.Scattergl(
            x=hemi_t,
            y=hemi_y,
            mode="lines",
            line=dict(width=1.2, color="darkgreen"),
            name=hemi_label,
        ),
    )

    # ADC value on secondary y-axis
    fig.add_trace(
        go.Scattergl(
            x=adc_time,
            y=adc_val,
            mode="lines",
            line=dict(width=1.0, color="crimson"),
            name="ADC value",
            yaxis="y2",
        ),
    )

    # Optional horizontal threshold line on proxy axis
    if THRESHOLD is not None:
        fig.add_hline(
            y=float(THRESHOLD),
            line=dict(color="black", width=1, dash="dash"),
        )

    # Layout and axes
    fig.update_layout(
        template="plotly_white",
        height=650,
        title=f"Proxy, hemisphere, and ADC overlay\n({os.path.basename(PROXY_CSV_PATH)})",
        xaxis=dict(title="Time (s)"),
        yaxis=dict(title="Proxy / hemisphere"),
        yaxis2=dict(
            title="ADC value",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
        ),
        hovermode="x unified",
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.7)",
        ),
    )

    # Save HTML next to proxy CSV
    base = os.path.splitext(PROXY_CSV_PATH)[0]
    html_path = base + "_adc_overlay_plot.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"Saved {html_path}")
    fig.show()


if __name__ == "__main__":
    main()

