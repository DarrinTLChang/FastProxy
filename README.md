# fastProxyV2

A **real-time-style fast proxy pipeline** for recordings. It treats the signal as a stream of **20 ms packets**, runs a **causal** highpass → NEO → mean on each packet, and outputs one proxy value per packet. Outputs are written as one CSV per (region, side) with per-channel and median proxy columns.

## What it does

- **Input:** A folder of `.mat` files (v7.3 HDF5). Filenames must match `micro{region}_{L|R}_{channel}.mat` (e.g. `microVIM_L_1.mat` = VIM, Left, channel 1).
- **Processing:** For each file, the signal is split into **20 ms bins**. For each bin:
  1. **Causal highpass** (3rd-order Butterworth, 350 Hz). Filter state is **carried across bins** so the full recording is filtered as one continuous stream.
  2. **NEO** (Nonlinear Energy Operator, k=1) on that bin only.
  3. **Mean** of the NEO output → one scalar per bin.
- **Grouping:** Files are grouped by (region, side). For each group, all channels are processed and trimmed to the shortest length.
- **Output:** One CSV per group: `time_s`, `ch1`, `ch2`, …, `{region}_{side}_median_proxy`. The last column is the median across channels at each time bin.  
  Filename: `micro{region}_{side}_neo_binned.csv`.

Optional **plotting:** with `--plot`, the script also plots the median proxy over 0–150 s and saves a PNG next to each CSV.

## Usage

```bash
python fastProxyV2.py <input_folder> [output_folder] [--plot]
```

- **input_folder** — Path to the folder containing the `.mat` files.
- **output_folder** — (Optional) Where to write the CSVs. Defaults to `input_folder`.
- **--plot** — Also plot the median proxy (0–150 s) for each group and save PNGs.

## Requirements

- Python 3.x
- `numpy`, `scipy`, `h5py`, `matplotlib`

Install with:

```bash
pip install numpy scipy h5py matplotlib
```

## Example

```bash
python fastProxyV2.py ./patient_data_raw ./output --plot
```

This reads all `micro*_L_*.mat` and `micro*_R_*.mat` from `patient_data_raw`, writes `micro{region}_{side}_neo_binned.csv` into `output`, and generates plot PNGs when `--plot` is used.
