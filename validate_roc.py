import numpy as np
import os
import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════
BURST_PAD_BEFORE = 0.020   # 20 ms before burst_start
BURST_PAD_AFTER  = 0.000   # 0 ms after burst_end


def load_proxy(csv_path):
    df = pd.read_csv(csv_path)
    result = {'time_s': df['time_s'].values}
    for col in df.columns:
        if 'hemisphere_L' in col:
            result['L'] = df[col].values
        elif 'hemisphere_R' in col:
            result['R'] = df[col].values
    return result


def load_bursts(csv_path):
    df = pd.read_csv(csv_path)
    return df['burst_start_ms'].values / 1000.0, df['burst_end_ms'].values / 1000.0


def create_burst_labels(time_s, burst_starts, burst_ends):
    labels = np.zeros(len(time_s), dtype=int)
    for start, end in zip(burst_starts, burst_ends):
        mask = (time_s >= start - BURST_PAD_BEFORE) & (time_s <= end + BURST_PAD_AFTER)
        labels[mask] = 1
    return labels


def find_youden_j_threshold(fpr, tpr, thresholds):
    """
    Find threshold that maximizes Youden's J:
        J = sensitivity + specificity - 1 = TPR - FPR
    """
    j_scores = tpr - fpr

    # Avoid choosing the initial inf threshold if present
    finite_mask = np.isfinite(thresholds)
    if np.any(finite_mask):
        valid_indices = np.where(finite_mask)[0]
        best_local_idx = np.argmax(j_scores[finite_mask])
        idx = valid_indices[best_local_idx]
    else:
        idx = np.argmax(j_scores)

    return {
        'threshold': thresholds[idx],
        'sensitivity': tpr[idx],
        'specificity': 1 - fpr[idx],
        'youden_j': j_scores[idx],
        'fpr': fpr[idx],
        'tpr': tpr[idx],
    }


def plot_roc(fpr, tpr, auc_score, optimal, side_label, output_path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, color='steelblue', linewidth=2, label=f'ROC (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Random')
    ax.plot(
        optimal['fpr'],
        optimal['tpr'],
        'b^',
        markersize=10,
        label=(
            f"Youden's J = {optimal['youden_j']:.2f} "
            f"(sens={optimal['sensitivity']:.2f}, spec={optimal['specificity']:.2f})"
        )
    )
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC — {side_label}\nAUC = {auc_score:.4f}', fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"    -> {output_path}")


def analyze_side(time_s, proxy, burst_csv, side_key, output_dir):
    side_label = 'Left' if side_key == 'L' else 'Right'
    if not burst_csv or not os.path.isfile(burst_csv):
        print(f"\n  {side_label}: No burst CSV, skipping.")
        return None

    burst_starts, burst_ends = load_bursts(burst_csv)
    labels = create_burst_labels(time_s, burst_starts, burst_ends)
    n_pos = int(np.sum(labels))
    n_neg = len(labels) - n_pos

    print(f"\n  {side_label}: {len(burst_starts)} bursts, {n_pos} burst bins ({100*n_pos/len(labels):.1f}%)")

    if n_pos == 0 or n_neg == 0:
        print("    Cannot compute ROC.")
        return None

    fpr, tpr, thresholds = roc_curve(labels, proxy)
    auc_score = auc(fpr, tpr)
    opt = find_youden_j_threshold(fpr, tpr, thresholds)

    print(f"    AUC:         {auc_score:.4f}")
    print(f"    J thresh:    {opt['threshold']:.4e}")
    print(f"    Sensitivity: {opt['sensitivity']:.4f}")
    print(f"    Specificity: {opt['specificity']:.4f}")
    print(f"    Youden's J:  {opt['youden_j']:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    plot_roc(
        fpr,
        tpr,
        auc_score,
        opt,
        side_label,
        os.path.join(output_dir, f'roc_{side_key}.png')
    )

    return {
        'side': side_label,
        'auc': round(auc_score, 4),
        'threshold_youden_j': f'{opt["threshold"]:.4e}',
        'sensitivity': round(opt['sensitivity'], 4),
        'specificity': round(opt['specificity'], 4),
        'youden_j': round(opt['youden_j'], 4),
        'n_burst_bins': n_pos,
        'n_nonburst_bins': n_neg,
        'pct_burst': round(100.0 * n_pos / len(labels), 2),
    }


def main():
    if len(sys.argv) == 2 and os.path.isfile(sys.argv[1]):
        with open(sys.argv[1], 'r') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        if len(lines) < 1:
            print("Args file must contain at least proxy_csv path (one path per line).")
            sys.exit(1)
        proxy_csv = lines[0]
        burst_L = None if len(lines) <= 1 or lines[1].lower() == 'none' else lines[1]
        burst_R = None if len(lines) <= 2 or lines[2].lower() == 'none' else lines[2]
        out_dir = lines[3] if len(lines) > 3 else './validation_output'
    elif len(sys.argv) < 4:
        print("Usage: python validate_roc.py <proxy_csv> <burst_left> <burst_right> [output_dir]")
        print("  Use 'none' if a side has no burst data.")
        print()
        print("  Or:  python validate_roc.py <args_file>")
        print("  Args file: one path per line — proxy_csv, burst_left, burst_right, output_dir")
        print("  Use 'none' on a line for a missing burst file.")
        sys.exit(1)
    else:
        proxy_csv = sys.argv[1]
        burst_L = sys.argv[2] if sys.argv[2].lower() != 'none' else None
        burst_R = sys.argv[3] if sys.argv[3].lower() != 'none' else None
        out_dir = sys.argv[4] if len(sys.argv) > 4 else './validation_output'

    print(f"Proxy: {proxy_csv}")
    print(f"Padding: -{BURST_PAD_BEFORE*1000:.0f}ms / +{BURST_PAD_AFTER*1000:.0f}ms")

    proxy_data = load_proxy(proxy_csv)
    time_s = proxy_data['time_s']
    print(f"  {len(time_s)} bins loaded")

    results = []

    if 'L' in proxy_data and burst_L:
        r = analyze_side(time_s, proxy_data['L'], burst_L, 'L', out_dir)
        if r:
            results.append(r)

    if 'R' in proxy_data and burst_R:
        r = analyze_side(time_s, proxy_data['R'], burst_R, 'R', out_dir)
        if r:
            results.append(r)

    if results:
        fields = list(results[0].keys())
        csv_path = os.path.join(out_dir, 'results.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(results)
        print(f"\n  -> {csv_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()