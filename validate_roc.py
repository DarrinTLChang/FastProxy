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
# Paths to ground truth burst CSVs (update these)
BURST_CSV_LEFT  = r""   # path to network_bursts_RS_left.csv
BURST_CSV_RIGHT = r""   # path to network_bursts_RS_right.csv

# Path to fast proxy hemisphere CSV (update this)
PROXY_CSV       = r""   # path to hemisphere_neo_binned.csv

# Burst window padding (in seconds)
BURST_PAD_BEFORE = 0.020   # 20 ms before burst_start
BURST_PAD_AFTER  = 0.100   # 100 ms after burst_end


# ──────────────────────────────────────────────
# Load proxy data
# ──────────────────────────────────────────────

def load_proxy(csv_path):
    """
    Load the hemisphere CSV.
    Returns dict with 'time_s' and proxy arrays for each side found.
    """
    df = pd.read_csv(csv_path)
    result = {'time_s': df['time_s'].values}

    for col in df.columns:
        if 'hemisphere_L' in col:
            result['L'] = df[col].values
        elif 'hemisphere_R' in col:
            result['R'] = df[col].values

    return result


# ──────────────────────────────────────────────
# Load ground truth bursts
# ──────────────────────────────────────────────

def load_bursts(csv_path):
    """
    Load burst start/end times from CSV.
    Expects columns 'burst_start_ms' and 'burst_end_ms'.
    Returns arrays in seconds.
    """
    df = pd.read_csv(csv_path)
    starts_s = df['burst_start_ms'].values / 1000.0
    ends_s = df['burst_end_ms'].values / 1000.0
    return starts_s, ends_s


# ──────────────────────────────────────────────
# Create binary burst label for each time bin
# ──────────────────────────────────────────────

def create_burst_labels(time_s, burst_starts, burst_ends,
                        pad_before=BURST_PAD_BEFORE, pad_after=BURST_PAD_AFTER):
    """
    For each time bin, label 1 if it falls within any padded burst window, 0 otherwise.

    pad_before: seconds to extend before burst_start (default 20 ms)
    pad_after:  seconds to extend after burst_end (default 100 ms)
    """
    labels = np.zeros(len(time_s), dtype=int)

    for start, end in zip(burst_starts, burst_ends):
        mask = (time_s >= start - pad_before) & (time_s <= end + pad_after)
        labels[mask] = 1

    return labels


# ──────────────────────────────────────────────
# ROC / AUC analysis
# ──────────────────────────────────────────────

def compute_roc_auc(proxy_values, burst_labels, side_label):
    """
    Compute ROC curve and AUC.

    proxy_values: 1D array of proxy values (the "score")
    burst_labels: 1D binary array (1 = burst, 0 = no burst)

    Returns (fpr, tpr, thresholds, auc_score)
    """
    n_burst = np.sum(burst_labels)
    n_total = len(burst_labels)
    pct_burst = 100.0 * n_burst / n_total

    print(f"\n  {side_label}:")
    print(f"    Total bins:  {n_total}")
    print(f"    Burst bins:  {n_burst} ({pct_burst:.1f}%)")
    print(f"    Non-burst:   {n_total - n_burst} ({100 - pct_burst:.1f}%)")

    if n_burst == 0:
        print(f"    WARNING: No burst bins found — cannot compute ROC.")
        return None, None, None, None
    if n_burst == n_total:
        print(f"    WARNING: All bins are burst — cannot compute ROC.")
        return None, None, None, None

    fpr, tpr, thresholds = roc_curve(burst_labels, proxy_values)
    auc_score = auc(fpr, tpr)

    print(f"    AUC: {auc_score:.4f}")

    # Interpretation
    if auc_score >= 0.9:
        print(f"    Interpretation: Excellent — proxy strongly distinguishes bursts")
    elif auc_score >= 0.8:
        print(f"    Interpretation: Good — proxy reliably detects bursts")
    elif auc_score >= 0.7:
        print(f"    Interpretation: Fair — proxy has moderate discriminative power")
    elif auc_score >= 0.6:
        print(f"    Interpretation: Poor — proxy weakly distinguishes bursts")
    else:
        print(f"    Interpretation: No better than random")

    return fpr, tpr, thresholds, auc_score


# ──────────────────────────────────────────────
# Find optimal threshold from ROC
# ──────────────────────────────────────────────

def find_optimal_threshold(fpr, tpr, thresholds, burst_labels):
    """
    Find optimal thresholds by two criteria:
      1. Youden's J (TPR - FPR) — best balance of sensitivity/specificity
      2. Highest accuracy — most total correct predictions (accounts for class imbalance)
    """
    # Youden's J
    j_scores = tpr - fpr
    j_idx = np.argmax(j_scores)

    # Accuracy: acc = TPR * prevalence + (1-FPR) * (1-prevalence)
    n_total = len(burst_labels)
    n_pos = np.sum(burst_labels)
    prevalence = n_pos / n_total
    accuracy = tpr * prevalence + (1 - fpr) * (1 - prevalence)
    acc_idx = np.argmax(accuracy)

    return {
        'threshold_youden': thresholds[j_idx],
        'tpr_youden': tpr[j_idx],
        'fpr_youden': fpr[j_idx],
        'sensitivity_youden': tpr[j_idx],
        'specificity_youden': 1 - fpr[j_idx],
        'youden_j': j_scores[j_idx],
        'threshold_acc': thresholds[acc_idx],
        'tpr_acc': tpr[acc_idx],
        'fpr_acc': fpr[acc_idx],
        'sensitivity_acc': tpr[acc_idx],
        'specificity_acc': 1 - fpr[acc_idx],
        'best_accuracy': accuracy[acc_idx],
    }


# ──────────────────────────────────────────────
# Burst vs Non-Burst stats (quick sanity check)
# ──────────────────────────────────────────────

def burst_vs_nonburst_stats(proxy_values, burst_labels, side_label):
    """Compare proxy amplitude during burst vs non-burst."""
    burst_vals = proxy_values[burst_labels == 1]
    nonburst_vals = proxy_values[burst_labels == 0]

    if len(burst_vals) == 0 or len(nonburst_vals) == 0:
        return

    burst_mean = np.mean(burst_vals)
    nonburst_mean = np.mean(nonburst_vals)
    ratio = burst_mean / nonburst_mean if nonburst_mean > 0 else float('inf')

    print(f"\n    Burst vs Non-Burst ({side_label}):")
    print(f"      Burst mean:     {burst_mean:.4e}")
    print(f"      Non-burst mean: {nonburst_mean:.4e}")
    print(f"      Ratio:          {ratio:.2f}x")


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def plot_roc(fpr, tpr, auc_score, optimal, side_label, output_path):
    """Plot the ROC curve with AUC and both optimal thresholds marked."""
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(fpr, tpr, color='steelblue', linewidth=2,
            label=f'ROC (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1,
            label='Random (AUC = 0.5)')

    # Mark Youden's J optimal
    ax.plot(optimal['fpr_youden'], optimal['tpr_youden'], 'ro', markersize=10,
            label=f'Youden (sens={optimal["sensitivity_youden"]:.2f}, '
                  f'spec={optimal["specificity_youden"]:.2f})')

    # Mark accuracy optimal
    ax.plot(optimal['fpr_acc'], optimal['tpr_acc'], 'gs', markersize=10,
            label=f'Accuracy (sens={optimal["sensitivity_acc"]:.2f}, '
                  f'spec={optimal["specificity_acc"]:.2f}, '
                  f'acc={optimal["best_accuracy"]:.2f})')

    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title(f'ROC Curve — {side_label} Hemisphere\nAUC = {auc_score:.4f}',
                 fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"    -> ROC plot: {output_path}")


def plot_proxy_with_bursts(time_s, proxy_values, burst_starts, burst_ends,
                            side_label, output_path,
                            threshold_acc=None, threshold_youden=None):
    """Interactive plotly plot of proxy trace with burst windows shaded."""
    import plotly.graph_objects as go

    fig = go.Figure()

    # Proxy trace
    fig.add_trace(go.Scattergl(
        x=time_s, y=proxy_values,
        mode='lines',
        line=dict(color='steelblue', width=0.5),
        name='Proxy',
    ))

    # Burst windows as shaded regions
    for start, end in zip(burst_starts, burst_ends):
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor='red', opacity=0.15,
            line_width=0,
        )

    # Youden's J threshold line
    if threshold_youden is not None and np.isfinite(threshold_youden):
        fig.add_hline(
            y=threshold_youden,
            line_dash='dash',
            line_color='orange',
            line_width=2,
            annotation_text=f'Youden threshold: {threshold_youden:.3e}',
            annotation_position='top left',
            annotation_font_color='orange',
        )

    # Accuracy-optimal threshold line
    if threshold_acc is not None and np.isfinite(threshold_acc):
        fig.add_hline(
            y=threshold_acc,
            line_dash='dash',
            line_color='green',
            line_width=2,
            annotation_text=f'Accuracy threshold: {threshold_acc:.3e}',
            annotation_position='top left',
            annotation_font_color='green',
        )

    fig.update_layout(
        title=f'{side_label} Hemisphere — Proxy with Burst Windows',
        xaxis_title='Time (s)',
        yaxis_title='Proxy Value',
        yaxis=dict(exponentformat='e'),
        template='plotly_white',
        height=500,
    )

    # Save as interactive HTML
    html_path = output_path.replace('.png', '.html')
    fig.write_html(html_path)
    print(f"    -> Interactive plot: {html_path}")


# ──────────────────────────────────────────────
# Write results CSV
# ──────────────────────────────────────────────

def write_results_csv(results, output_path):
    """Write a summary CSV of ROC/AUC results."""
    fieldnames = [
        'side', 'auc',
        'threshold_youden', 'sensitivity_youden', 'specificity_youden', 'youden_j',
        'threshold_acc', 'sensitivity_acc', 'specificity_acc', 'best_accuracy',
        'burst_mean', 'nonburst_mean', 'ratio',
        'n_burst_bins', 'n_nonburst_bins', 'pct_burst',
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n  -> Results CSV: {output_path}")


# ──────────────────────────────────────────────
# Run analysis for one side
# ──────────────────────────────────────────────

def analyze_side(time_s, proxy_values, burst_csv_path, side_key, output_dir):
    """
    Full ROC/AUC analysis for one hemisphere side.
    Returns results dict or None if no data.
    """
    side_label = 'Left' if side_key == 'L' else 'Right'

    if not burst_csv_path or not os.path.isfile(burst_csv_path):
        print(f"\n  {side_label}: No burst CSV found, skipping.")
        return None

    # Load bursts
    burst_starts, burst_ends = load_bursts(burst_csv_path)
    print(f"\n  {side_label}: {len(burst_starts)} burst events loaded")

    # Create binary labels
    labels = create_burst_labels(time_s, burst_starts, burst_ends)

    # ROC / AUC
    fpr, tpr, thresholds, auc_score = compute_roc_auc(proxy_values, labels, side_label)

    if auc_score is None:
        return None

    # Optimal threshold
    optimal = find_optimal_threshold(fpr, tpr, thresholds, labels)
    print(f"\n    --- Youden's J criterion ---")
    print(f"    Threshold:   {optimal['threshold_youden']:.4e}")
    print(f"    Sensitivity: {optimal['sensitivity_youden']:.4f}")
    print(f"    Specificity: {optimal['specificity_youden']:.4f}")
    print(f"    Youden's J:  {optimal['youden_j']:.4f}")
    print(f"\n    --- Highest accuracy criterion ---")
    print(f"    Threshold:   {optimal['threshold_acc']:.4e}")
    print(f"    Sensitivity: {optimal['sensitivity_acc']:.4f}")
    print(f"    Specificity: {optimal['specificity_acc']:.4f}")
    print(f"    Accuracy:    {optimal['best_accuracy']:.4f}")

    # Burst vs non-burst stats
    burst_vs_nonburst_stats(proxy_values, labels, side_label)

    burst_vals = proxy_values[labels == 1]
    nonburst_vals = proxy_values[labels == 0]
    burst_mean = np.mean(burst_vals) if len(burst_vals) > 0 else 0
    nonburst_mean = np.mean(nonburst_vals) if len(nonburst_vals) > 0 else 0
    ratio = burst_mean / nonburst_mean if nonburst_mean > 0 else 0

    # Plots
    os.makedirs(output_dir, exist_ok=True)

    plot_roc(fpr, tpr, auc_score, optimal, side_label,
             os.path.join(output_dir, f'roc_{side_key}.png'))

    plot_proxy_with_bursts(time_s, proxy_values, burst_starts, burst_ends,
                           side_label,
                           os.path.join(output_dir, f'proxy_bursts_{side_key}.html'),
                           threshold_acc=optimal['threshold_acc'],
                           threshold_youden=optimal['threshold_youden'])

    n_burst = int(np.sum(labels))
    n_total = len(labels)

    return {
        'side': side_label,
        'auc': round(auc_score, 4),
        'threshold_youden': f'{optimal["threshold_youden"]:.4e}',
        'sensitivity_youden': round(optimal['sensitivity_youden'], 4),
        'specificity_youden': round(optimal['specificity_youden'], 4),
        'youden_j': round(optimal['youden_j'], 4),
        'threshold_acc': f'{optimal["threshold_acc"]:.4e}',
        'sensitivity_acc': round(optimal['sensitivity_acc'], 4),
        'specificity_acc': round(optimal['specificity_acc'], 4),
        'best_accuracy': round(optimal['best_accuracy'], 4),
        'burst_mean': f'{burst_mean:.4e}',
        'nonburst_mean': f'{nonburst_mean:.4e}',
        'ratio': round(ratio, 2),
        'n_burst_bins': n_burst,
        'n_nonburst_bins': n_total - n_burst,
        'pct_burst': round(100.0 * n_burst / n_total, 2),
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    if len(sys.argv) < 4:
        print("Usage: python validate_roc.py <proxy_csv> <burst_csv_left> <burst_csv_right> [output_folder]")
        print()
        print("  proxy_csv       : hemisphere_neo_binned.csv from the pipeline")
        print("  burst_csv_left  : network_bursts_RS_left.csv (ground truth)")
        print("  burst_csv_right : network_bursts_RS_right.csv (ground truth)")
        print("  output_folder   : where to save results (default: ./validation_output)")
        print()
        print("  Use 'none' for a burst CSV if that side has no data.")
        sys.exit(1)

    proxy_csv = sys.argv[1]
    burst_csv_left = sys.argv[2] if sys.argv[2].lower() != 'none' else None
    burst_csv_right = sys.argv[3] if sys.argv[3].lower() != 'none' else None
    output_dir = sys.argv[4] if len(sys.argv) > 4 else './validation_output'

    if not os.path.isfile(proxy_csv):
        print(f"Error: proxy CSV not found: {proxy_csv}")
        sys.exit(1)

    print(f"Loading proxy data from: {proxy_csv}")
    print(f"Burst padding: -{BURST_PAD_BEFORE*1000:.0f} ms before, +{BURST_PAD_AFTER*1000:.0f} ms after")
    proxy_data = load_proxy(proxy_csv)
    time_s = proxy_data['time_s']
    print(f"  {len(time_s)} time bins loaded")

    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    # Analyze left
    if 'L' in proxy_data and burst_csv_left:
        result = analyze_side(time_s, proxy_data['L'], burst_csv_left, 'L', output_dir)
        if result:
            all_results.append(result)

    # Analyze right
    if 'R' in proxy_data and burst_csv_right:
        result = analyze_side(time_s, proxy_data['R'], burst_csv_right, 'R', output_dir)
        if result:
            all_results.append(result)

    # Write summary CSV
    if all_results:
        write_results_csv(all_results, os.path.join(output_dir, 'roc_auc_results.csv'))

    print("\nDone!")


if __name__ == '__main__':
    main()