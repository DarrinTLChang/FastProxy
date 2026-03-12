import os

import pandas as pd
import plotly.graph_objects as go


CSV_PATH = r"E:\closed loop testing\recorded binary files\proxy_feature_record3.csv"

# Columns to plot
X_COL = "time_s"
Y_COL = "feature_value"


def main():
    df = pd.read_csv(CSV_PATH, low_memory=False)

    if X_COL not in df.columns or Y_COL not in df.columns:
        raise ValueError(
            f"Missing expected columns. Wanted x='{X_COL}', y='{Y_COL}'. "
            f"Found: {list(df.columns)}"
        )

    x = pd.to_numeric(df[X_COL], errors="coerce")
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    mask = x.notna() & y.notna()

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=x[mask],
            y=y[mask],
            mode="lines",
            line=dict(width=1),
            name=Y_COL,
        )
    )
    fig.update_layout(
        title=os.path.basename(CSV_PATH),
        template="plotly_white",
        xaxis_title=X_COL,
        yaxis_title=Y_COL,
        height=450,
    )

    html_path = os.path.splitext(CSV_PATH)[0] + "_plot.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"Saved {html_path}")
    fig.show()


if __name__ == "__main__":
    main()

