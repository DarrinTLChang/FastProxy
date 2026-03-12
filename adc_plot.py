import pandas as pd
import plotly.graph_objects as go

path = r"D:\s531\processed data from 531\Mat Data\E\CL testing\period1\ADC1.csv"
# Read first two columns and coerce to numeric (handles headers/mixed types)
df = pd.read_csv(path, usecols=[0, 1], low_memory=False)
df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
df = df.dropna()

x = df.iloc[:, 0].to_numpy()
y = df.iloc[:, 1].to_numpy()
x_label = str(df.columns[0])
y_label = str(df.columns[1])

fig = go.Figure()
fig.add_trace(go.Scattergl(x=x, y=y, mode="lines", line=dict(width=1), name=y_label))
fig.update_layout(
    title="ADC1.csv",
    template="plotly_white",
    xaxis_title=x_label,
    yaxis_title=y_label,
    height=450,
)

html_path = path.replace(".csv", "_plot.html")
fig.write_html(html_path, include_plotlyjs="cdn")
print(f"Saved {html_path}")

fig.show()
