"""Plot time_s vs hemisphere_L_median_proxy and time_s vs hemisphere_R_median_proxy."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# CSV_PATH = r'/Volumes/D_Drive/s531_fp_output/Day1/Baseline/fastProxy/Period3/hemisphere_neo_binned.csv'

# CSV_PATH = r'/Volumes/D_Drive/s531_fp_output/Day2/Baseline/fastProxy/Period2/hemisphere_neo_binned.csv'
CSV_PATH = r'/Volumes/D_Drive/s531_fp_output/Day4_test/p4/includeChannel=True/hemisphere_neo_binned.csv'
plot_R = False
# Horizontal threshold line (same units as proxy). Set to None to disable.
THRESHOLD = None  # e.g. 1e7
def main():
    data = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)
    time_s = data[:, 0]
    L_proxy = data[:, 1]

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
    if plot_R:
        R_proxy = data[:, 2]

        fig.add_trace(
            go.Scattergl(x=time_s, y=R_proxy, mode="lines", line=dict(color="coral", width=1), name="hemisphere_R_median_proxy"),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="hemisphere_R_median_proxy", row=2, col=1)


    fig.update_yaxes(title_text="hemisphere_L_median_proxy", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)

    fig.update_layout(
        title="Hemisphere median proxy vs time",
        template="plotly_white",
        height=650,
        showlegend=False,
    )

    # Optional horizontal threshold line on both subplots
    if THRESHOLD is not None:
        fig.add_hline(
            y=THRESHOLD,
            line=dict(color="black", width=1, dash="dash"),
            row=1,
            col=1,
        )
        fig.add_hline(
            y=THRESHOLD,
            line=dict(color="black", width=1, dash="dash"),
            row=2,
            col=1,
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
