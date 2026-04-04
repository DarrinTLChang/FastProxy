import pandas as pd
import plotly.graph_objects as go

from plotly_resampler import FigureResampler  # pyright: ignore[reportMissingImports]

# path = r"F:\s531\processed data from 531\CL testing\macro\period8\ADC2.csv"
path = r"F:\s531_binary\period8_test\offline\hemisphere_neo_binned.csv"
# Max points drawn per view while panning/zooming (full series kept as hf_x/hf_y)
MAX_N_SAMPLES = 8000

# Read time_s and adc_value; coerce to numeric (handles headers/mixed types)
df = pd.read_csv(path, usecols=["time_s", "adc_value"], low_memory=False)
df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
df["adc_value"] = pd.to_numeric(df["adc_value"], errors="coerce")
df = df.dropna()

x = df["time_s"].to_numpy()
y = df["adc_value"].to_numpy()
x_label = "time_s"
y_label = "adc_value"

fig = FigureResampler(go.Figure())
fig.add_trace(
    go.Scattergl(
        mode="lines",
        line=dict(width=1),
        name=y_label,
        hoverinfo="skip",
    ),
    hf_x=x,
    hf_y=y,
    max_n_samples=MAX_N_SAMPLES,
)
fig.update_layout(hovermode=False)
fig.update_layout(
    title="ADC2.csv",
    template="plotly_white",
    xaxis_title=x_label,
    yaxis_title=y_label,
    height=450,
)

if __name__ == "__main__":
    # Opens a local Dash app; zoom/pan resamples from the full arrays above.
    # Static HTML cannot do this—use this viewer for interactive work.
    print("Starting Dash (Ctrl+C to stop). Open the URL shown in the console.")
    fig.show_dash(debug=False)
