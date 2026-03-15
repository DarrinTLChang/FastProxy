import os
import pandas as pd
import plotly.graph_objects as go

FOLDER = r'/Volumes/D_Drive/s531_fp_output/Day5_baseline/p8/includeChannel=False'
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
        made_any = False

        for col in channel_cols:
            y = pd.to_numeric(df[col], errors="coerce")
            valid = time_s.notna() & y.notna()
            if not valid.any():
                continue

            fig = go.Figure()
            fig.add_trace(
                go.Scattergl(
                    x=time_s[valid],
                    y=y[valid],
                    mode="lines",
                    name=col,
                    line=dict(width=0.8),
                )
            )

            fig.update_layout(
                title=f"{base} — {col}",
                xaxis_title="Time (s)",
                yaxis_title=col,
                template="plotly_white",
                height=500,
            )

            out_path = os.path.join(FOLDER, f"{base}_{col}_plot.html")
            fig.write_html(out_path)
            made_any = True

        if made_any:
            print(f"  {fname} -> per-channel HTMLs written")
        else:
            print(f"Skipping {fname}: no valid channel data")

    print("Done.")


if __name__ == "__main__":
    main()