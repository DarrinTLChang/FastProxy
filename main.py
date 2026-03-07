"""
Fast proxy: raw → highpass → NEO → 20 ms block average → CSV.
"""

from __future__ import annotations

import re
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

try:
    import h5py
except ImportError:
    h5py = None

# ── Settings ──
DATA_ROOT = Path(__file__).resolve().parent / "patient_data_raw"
OUTPUT_DIR = Path(__file__).resolve().parent / "patient_data_proxy"
FS = 24414.0625
HIGHPASS_HZ: float | None = 300.0  # None = skip highpass
HIGHPASS_ORDER = 4
WINDOW_MS = 20.0

_MICRO_PATTERN = re.compile(r"^micro(.+)_(L|R)_(\d+)(?:_CommonFiltered)?\.mat$", re.IGNORECASE)


# ── Pipeline ──
def highpass_filter(data: np.ndarray) -> np.ndarray:
    sos = butter(HIGHPASS_ORDER, HIGHPASS_HZ, btype="high", fs=FS, output="sos")
    return sosfiltfilt(sos, data)


def neo(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    out[1:-1] = np.sqrt(np.abs(x[1:-1] ** 2 - x[:-2] * x[2:]))
    return out


def block_average(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_per = max(1, int(round(WINDOW_MS * 1e-3 * FS)))
    n_win = len(signal) // n_per
    trimmed = signal[: n_win * n_per].reshape(n_win, n_per)
    t_s = np.arange(n_win) * (WINDOW_MS * 1e-3)
    return t_s, np.mean(trimmed, axis=1)


def raw_to_proxy(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = highpass_filter(raw) if HIGHPASS_HZ else raw.astype(float)
    return block_average(neo(x))


# ── File I/O ──
def load_mat(path: Path) -> np.ndarray:
    try:
        mat = loadmat(str(path), squeeze_me=True)
    except Exception as e:
        if "v7.3" in str(e) or "HDF" in str(e).upper():
            if h5py is None:
                raise ImportError("v7.3 .mat requires: pip install h5py")
            with h5py.File(path, "r") as f:
                return np.asarray(f["data"]).astype(float).ravel()
        raise
    return np.asarray(mat["data"], dtype=float).ravel()


def save_csv(path: Path, t_s: np.ndarray, proxy: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        np.savetxt(f, np.column_stack([t_s, proxy]), delimiter=",",
                   header="time(s),proxy value", comments="")


def _parse_name(path: Path) -> tuple[str, str, int] | None:
    m = _MICRO_PATTERN.match(path.name)
    return (m.group(1), m.group(2).upper(), int(m.group(3))) if m else None


def _group_by_region_side(mat_files: list[Path], root: Path) -> dict:
    groups = defaultdict(list)
    for p in mat_files:
        parsed = _parse_name(p)
        if not parsed:
            continue
        region, side, ch = parsed
        groups[(p.relative_to(root).parent, region, side)].append(p)
    for k in groups:
        groups[k].sort(key=lambda p: p.name)
    return dict(groups)


# ── Main ──
def process_all():
    mat_files = sorted(DATA_ROOT.rglob("*.mat"))
    if not mat_files:
        return print("No .mat files found.")
    saved = []

    # Per-channel
    for p in mat_files:
        try:
            t_s, proxy = raw_to_proxy(load_mat(p))
            out = OUTPUT_DIR / p.relative_to(DATA_ROOT).parent / f"{p.stem}_proxy.csv"
            save_csv(out, t_s, proxy)
            saved.append(out)
        except Exception as e:
            print(f"SKIP {p.relative_to(DATA_ROOT)}: {e}")

    # Per-region (median across channels, L and R separate)
    for (rel_dir, region, side), paths in _group_by_region_side(mat_files, DATA_ROOT).items():
        try:
            all_data = np.array([load_mat(p) for p in paths])
            t_s, proxy = raw_to_proxy(np.median(all_data, axis=0))
            out = OUTPUT_DIR / rel_dir / f"micro{region}_{side}_common_proxy.csv"
            save_csv(out, t_s, proxy)
            saved.append(out)
        except Exception as e:
            print(f"SKIP region {region} {side}: {e}")

    print(f"Saved {len(saved)} CSV(s).")


if __name__ == "__main__":
    process_all()
