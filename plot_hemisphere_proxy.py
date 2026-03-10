"""Plot time_s vs hemisphere_L_median_proxy and time_s vs hemisphere_R_median_proxy."""
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = r"C:\Users\Maral\Desktop\Darrin\FastProxy\s523_test_outputs\test523p1\hemisphere_neo_binned.csv"

def main():
    data = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)
    time_s = data[:, 0]
    L_proxy = data[:, 1]
    R_proxy = data[:, 2]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(time_s, L_proxy, color="steelblue", linewidth=0.5)
    ax1.set_ylabel("hemisphere_L_median_proxy")
    ax1.set_title("Left hemisphere")
    ax1.grid(True, alpha=0.3)

    ax2.plot(time_s, R_proxy, color="coral", linewidth=0.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("hemisphere_R_median_proxy")
    ax2.set_title("Right hemisphere")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CSV_PATH.replace(".csv", "_plots.png"), dpi=150)
    print(f"Saved {CSV_PATH.replace('.csv', '_plots.png')}")
    plt.show()

if __name__ == "__main__":
    main()
