import pandas as pd
import plotly.graph_objects as go

path = r"F:\s531_binary\period8\period8_bin.csv"
# Read time_s and proxy_feature; coerce to numeric (handles headers/mixed types)
df = pd.read_csv(path, usecols=["time_s", "proxy_feature"], low_memory=False)
df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
df["proxy_feature"] = pd.to_numeric(df["proxy_feature"], errors="coerce")
df = df.dropna()

x = df["time_s"].to_numpy()
y = df["proxy_feature"].to_numpy()
x_label = "time_s"
y_label = "proxy_feature"

fig = go.Figure()
fig.add_trace(go.Scattergl(x=x, y=y, mode="lines", line=dict(width=1), name=y_label))
fig.update_layout(
    title="ADC2.csv",
    template="plotly_white",
    xaxis_title=x_label,
    yaxis_title=y_label,
    height=450,
)

html_path = path.replace(".csv", "_plot.html")
fig.write_html(html_path, include_plotlyjs="cdn")
print(f"Saved {html_path}")

fig.show()
