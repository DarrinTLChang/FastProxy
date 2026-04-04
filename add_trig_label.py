"""
Add trigger labels from the closed-loop log to the proxy CSV.

Parses [TRIG] FIRE and [TRIG] SKIP events from the log,
matches each to the proxy CSV by proxy_feat value (exact 6-decimal match),
adds a 'trig_label' column (FIRE, SKIP, or empty).

Call by :  
python add_trig_labels.py <proxy_csv> <log_file> <output_csv>
"""

import pandas as pd
import numpy as np
import re
import sys
import os


def parse_trig_events(log_path):
    events = []
    trig_re = re.compile(
        r'\[TRIG\] (FIRE|SKIP) proxy_feat=([\d.]+)')

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = trig_re.search(line)
            if m:
                events.append({
                    'type': m.group(1),
                    'proxy_feat_str': m.group(2),  # keep as string for exact matching
                })
    return events


def main():
    if len(sys.argv) < 4:
        print("Usage: python add_trig_labels.py <proxy_csv> <log_file> <output_csv>")
        sys.exit(1)

    proxy_csv = sys.argv[1]
    log_file = sys.argv[2]
    output_csv = sys.argv[3]

    # Load proxy CSV
    df = pd.read_csv(proxy_csv)

    # Find the feature column
    feat_col = None
    for name in ['proxy_feature', 'feature_value', 'feature', 'value']:
        if name in df.columns:
            feat_col = name
            break
        for col in df.columns:
            if col.lower() == name.lower():
                feat_col = col
                break
        if feat_col:
            break

    if feat_col is None:
        print(f"Error: no feature column found. Columns: {list(df.columns)}")
        sys.exit(1)

    # Round CSV values to 6 decimals and convert to string for exact matching
    proxy_values = df[feat_col].values.astype(float)
    proxy_str = [f"{v:.6f}" for v in proxy_values]

    # Build lookup: rounded string -> list of indices (in case of duplicates)
    lookup = {}
    for i, s in enumerate(proxy_str):
        if s not in lookup:
            lookup[s] = []
        lookup[s].append(i)

    # Parse log
    print(f"Parsing log: {log_file}")
    events = parse_trig_events(log_file)
    print(f"  {len(events)} TRIG events found")

    # Add empty column
    df['trig_label'] = ''

    # Match each event by exact 6-decimal string
    matched = 0
    unmatched = 0
    for ev in events:
        target_str = f"{float(ev['proxy_feat_str']):.6f}"

        if target_str in lookup and len(lookup[target_str]) > 0:
            # Take the first unused match
            idx = lookup[target_str].pop(0)
            df.at[idx, 'trig_label'] = ev['type']
            matched += 1
        else:
            unmatched += 1
            print(f"  WARNING: no exact match for {ev['type']} proxy_feat={ev['proxy_feat_str']}")

    n_fire = (df['trig_label'] == 'FIRE').sum()
    n_skip = (df['trig_label'] == 'SKIP').sum()
    print(f"  Matched: {matched}/{len(events)}")
    if unmatched > 0:
        print(f"  Unmatched: {unmatched}")
    print(f"  FIRE: {n_fire}, SKIP: {n_skip}")

    # Save
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")


if __name__ == '__main__':
    main()