import pandas as pd
import plotly.graph_objects as go

ADC_CSV = r"F:\s531\processed data from 531\CL testing\macro\period9\ADC2.csv"
BINARY_CSV = r"F:\s531_binary\period9\period9_test_bin.csv"

# ADC_CSV_PATH   = r"F:\s531\processed data from 531\CL testing\macro\period8\ADC2.csv"
# PROXY_CSV_PATH = r"F:\s531_binary\period8\proxy_with_labels.csv"

# Seconds to shift the binary trace left (subtracted from binary time_s). Increase if binary is delayed vs ADC.
# Use a negative value here to shift the binary trace to the right instead.
BINARY_SHIFT_LEFT_S = 0

# Horizontal threshold line (black, dotted). On proxy scale (left axis, yref="y").
THRESHOLD = 95.0
THRESHOLD_YREF = "y"

def _load_series(path: str, x_col: str, y_col: str) -> tuple:
    df = pd.read_csv(path, usecols=[x_col, y_col], low_memory=False)
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna()
    return df[x_col].to_numpy(), df[y_col].to_numpy()


adc_x, adc_y = _load_series(ADC_CSV, "time_s", "adc_value")
bin_x, bin_y = _load_series(BINARY_CSV, "time_s", "proxy_feature")
bin_x = bin_x - BINARY_SHIFT_LEFT_S

fig = go.Figure()
# proxy_feature first (legend / layer order), then adc_value
fig.add_trace(
    go.Scattergl(
        x=bin_x,
        y=bin_y,
        mode="lines",
        line=dict(width=1, color="#ff7f0e"),
        name="proxy_feature",
        hovertemplate="time_s=%{x:.6f}<br>proxy_feature=%{y:.4f}<extra></extra>",
    )
)
fig.add_trace(
    go.Scattergl(
        x=adc_x,
        y=adc_y,
        mode="lines",
        line=dict(width=1, color="#1f77b4"),
        name="adc_value",
        yaxis="y2",
        hoverinfo="skip",
    )
)

fig.add_hline(
    y=THRESHOLD,
    yref=THRESHOLD_YREF,
    line_dash="dot",
    line_color="black",
    line_width=1.5,
    annotation_text="threshold",
    annotation_position="top left",
)

fig.update_layout(
    hovermode="closest",
    title="ADC + binary proxy (proxy axis visible; ADC axis hidden)",
    template="plotly_white",
    xaxis_title="time_s",
    yaxis=dict(
        title=dict(text="proxy_feature", font=dict(color="black")),
        tickfont=dict(color="black"),
        linecolor="black",
        tickcolor="black",
        exponentformat="none",
        tickformat=".0f",
        range=[85, 150],
    ),
    yaxis2=dict(
        overlaying="y",
        side="right",
        showgrid=False,
        visible=False,
        showticklabels=False,
        showline=False,
    ),
    height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

html_path = ADC_CSV.replace(".csv", "_binary_overlay_plot.html")
fig.write_html(html_path, include_plotlyjs="cdn")
print(f"Saved {html_path}")

fig.show()
