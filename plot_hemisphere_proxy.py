"""Plot time_s vs hemisphere_L_median_proxy and time_s vs hemisphere_R_median_proxy."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CSV_PATH = r"C:\Users\darri\OneDrive\Documents\GitHub\FastProxy\s531_output\p1-3_output\p1_output\hemisphere_neo_binned.csv"
def main():
    data = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)
    time_s = data[:, 0]
    L_proxy = data[:, 1]
    # R_proxy = data[:, 2]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Left hemisphere", "Right hemisphere"),
    )

    fig.add_trace(
        go.Scattergl(x=time_s, y=L_proxy, mode="lines", line=dict(color="steelblue", width=1), name="hemisphere_L_median_proxy"),
        row=1,
        col=1,
    )
    # fig.add_trace(
    #     go.Scattergl(x=time_s, y=R_proxy, mode="lines", line=dict(color="coral", width=1), name="hemisphere_R_median_proxy"),
    #     row=2,
    #     col=1,
    # )

    fig.update_yaxes(title_text="hemisphere_L_median_proxy", row=1, col=1)
    fig.update_yaxes(title_text="hemisphere_R_median_proxy", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)

    fig.update_layout(
        title="Hemisphere median proxy vs time",
        template="plotly_white",
        height=650,
        showlegend=False,
    )

    html_path = CSV_PATH.replace(".csv", "_plots.html")
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"Saved {html_path}")

    # Optional: also save a PNG if kaleido is installed.
    # pip install -U kaleido
    try:
        png_path = CSV_PATH.replace(".csv", "_plots.png")
        fig.write_image(png_path, scale=2)
        print(f"Saved {png_path}")
    except Exception:
        pass

    fig.show()

if __name__ == "__main__":
    main()
