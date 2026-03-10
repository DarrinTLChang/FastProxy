"""Plot time_s vs last column for every CSV in test523p1 except hemisphere_neo_binned.csv."""
import os
import numpy as np
import matplotlib.pyplot as plt

FOLDER = r"C:\Users\Maral\Desktop\Darrin\FastProxy\s523_test_outputs\test523p1"
SKIP = "hemisphere_neo_binned.csv"


def main():
    for fname in sorted(os.listdir(FOLDER)):
        if not fname.lower().endswith(".csv") or fname == SKIP:
            continue
        path = os.path.join(FOLDER, fname)
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        time_s = data[:, 0]
        last_col = data[:, -1]

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(time_s, last_col, color="steelblue", linewidth=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Proxy")
        ax.set_title(fname.replace(".csv", ""))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = path.replace(".csv", "_plot.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  {fname} -> {os.path.basename(out_path)}")

    print("Done.")


if __name__ == "__main__":
    main()
