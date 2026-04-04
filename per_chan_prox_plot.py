import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FOLDER = r"F:\s531_binary\period8_test_LP\offline_all"
SKIP = "hemisphere_neo_binned.csv"


def main():
    for fname in sorted(os.listdir(FOLDER)):
        fname_lower = fname.lower()

        if fname.startswith("._"):
            continue
        if fname_lower.endswith(".html"):
            continue
        if not fname_lower.endswith(".csv"):
            continue
        if fname == SKIP:
            continue

        path = os.path.join(FOLDER, fname)

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Skipping {fname}: could not read CSV ({e})")
            continue

        if "time_s" not in df.columns:
            print(f"Skipping {fname}: no time_s column")
            continue

        time_s = pd.to_numeric(df["time_s"], errors="coerce")
        channel_cols = [f"ch{i}" for i in range(1, 11) if f"ch{i}" in df.columns]

        if not channel_cols:
            print(f"Skipping {fname}: no channel columns found")
            continue

        base = fname.replace(".csv", "")
        out_dir = os.path.join(FOLDER, "per_chan_plots")
        os.makedirs(out_dir, exist_ok=True)

        valid_cols = []
        for col in channel_cols:
            y = pd.to_numeric(df[col], errors="coerce")
            valid = time_s.notna() & y.notna()
            if valid.any():
                valid_cols.append(col)

        if not valid_cols:
            print(f"Skipping {fname}: no valid channel data")
            continue

        # --- individual per-channel HTMLs ---
        for col in valid_cols:
            y = pd.to_numeric(df[col], errors="coerce")
            valid = time_s.notna() & y.notna()

            fig = go.Figure()
            fig.add_trace(
                go.Scattergl(
                    x=time_s[valid],
                    y=y[valid],
                    mode="lines",
                    name=col,
                    line=dict(width=0.8, color="#1f77b4"),
                )
            )
            fig.update_layout(
                title=f"{base} — {col}",
                xaxis_title="Time (s)",
                yaxis_title=col,
                yaxis=dict(range=[0,20]),
                template="plotly_white",
                height=500,
            )
            fig.write_html(os.path.join(out_dir, f"{base}_{col}_plot.html"))

        # --- combined scrollable HTML (all channels stacked) ---
        combined = make_subplots(
            rows=len(valid_cols), cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=[f"{col}" for col in valid_cols],
        )
        for i, col in enumerate(valid_cols, start=1):
            y = pd.to_numeric(df[col], errors="coerce")
            valid = time_s.notna() & y.notna()
            combined.add_trace(
                go.Scattergl(
                    x=time_s[valid],
                    y=y[valid],
                    mode="lines",
                    name=col,
                    line=dict(width=0.8, color="#1f77b4"),
                ),
                row=i, col=1,
            )
            combined.update_yaxes(title_text=col, range=[0, 20], row=i, col=1)

        combined.update_layout(
            title=f"{base} — all channels",
            template="plotly_white",
            height=400 * len(valid_cols),
            showlegend=False,
        )
        combined.update_xaxes(title_text="Time (s)", row=len(valid_cols), col=1)
        combined.write_html(os.path.join(out_dir, f"{base}_all_channels.html"))

        print(f"  {fname} -> {len(valid_cols)} individual + 1 combined HTML written")

    print("Done.")


if __name__ == "__main__":
    main()