import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# Paths (edit if needed)
ADC_CSV_PATH = r"F:\s531\processed data from 531\Mat Data\E\CL testing\period2\ADC1.csv"
PROXY_CSV_PATH = r"F:\closed loop testing\recorded binary files\proxy_feature_record2.csv"

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

    fig = go.Figure()

    # Proxy feature on primary y-axis
    fig.add_trace(
        go.Scattergl(
            x=proxy_time,
            y=proxy_val,
            mode="lines",
            line=dict(width=1.5, color="steelblue"),
            name="proxy feature",
            yaxis="y1",
        )
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
        )
    )

    # Optional horizontal threshold line on proxy axis
    if THRESHOLD is not None:
        fig.add_hline(
            y=float(THRESHOLD),
            line=dict(color="black", width=1, dash="dash"),
        )

    fig.update_layout(
        template="plotly_white",
        height=650,
        title="Proxy feature with ADC value overlay",
        xaxis_title="Time (s)",
        yaxis_title="Proxy feature",
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

