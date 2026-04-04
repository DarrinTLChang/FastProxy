import pandas as pd
import plotly.graph_objects as go

path = r"F:\s531_binary\period8\period8_test_bin.csv"
# path = r"F:\s531_binary\period8_test\offline\hemisphere_neo_binned.csv"

Y_MIN = 50
Y_MAX = 350

df = pd.read_csv(path, usecols=["time_s", "proxy_feature"], low_memory=False)
df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
df["proxy_feature"] = pd.to_numeric(df["proxy_feature"], errors="coerce")
df = df.dropna()

x = df["time_s"].to_numpy()
y = df["proxy_feature"].to_numpy()

print(f"Rows: {len(df)}")
print(f"Min: {y.min():.4f}, Max: {y.max():.4f}, Median: {pd.Series(y).median():.4f}")

fig = go.Figure()
fig.add_trace(go.Scattergl(x=x, y=y, mode="lines", line=dict(width=1), name="proxy_feature"))
fig.update_layout(
    # title=path.split("\\")[-1],
    title="Period 8 Online FastProxy",

    template="plotly_white",
    xaxis_title="time_s",
    yaxis_title="proxy_feature",
    yaxis=dict(range=[Y_MIN, Y_MAX]),
    height=450,
)

html_path = path.replace(".csv", "_plot.html")
fig.write_html(html_path, include_plotlyjs="cdn")
print(f"Saved {html_path}")
fig.show()