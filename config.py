"""Shared configuration for the Aquabot water-quality pipeline.

Purpose
    One place for everything the numbered scripts share: the random seed,
    folder layout, which dataset is active, the Penaeus vannamei water-quality
    thresholds, the labelling rule, colours, plot style and small helpers.
    Every script imports from here, so a threshold or colour is changed once.

Paper link
    None directly. The thresholds are the project brief's domain rules for
    P. vannamei grow-out ponds; the pond label is the worst single parameter
    (Liebig's law of the minimum).

Outputs
    Nothing when imported, except creating figures/<dataset>/,
    models/<dataset>/ and reports/<dataset>/. Run `python config.py` to print
    the active settings and a few hand-checkable labelling examples.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

# ---------------------------------------------------------------- paths ----
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SYNTHETIC_CSV = DATA_DIR / "prawn_pond_synthetic.csv"
REAL_CSV = DATA_DIR / "pond_dataset.csv"            # Dry Bar after 00_prepare
REAL_SITE2_CSV = DATA_DIR / "pond_dataset_site2.csv"  # Cat Point (cross-site)


def _pick_dataset() -> str:
    """AQUABOT_DATASET = synthetic | real. Unset: real if its CSV exists."""
    choice = os.environ.get("AQUABOT_DATASET", "").strip().lower()
    if not choice:
        return "real" if REAL_CSV.exists() else "synthetic"
    if choice not in ("synthetic", "real"):
        raise ValueError(f"AQUABOT_DATASET must be 'synthetic' or 'real', got {choice!r}")
    return choice


DATASET = _pick_dataset()
DATA_CSV = REAL_CSV if DATASET == "real" else SYNTHETIC_CSV
# Every real-data result must carry this caveat (it is an estuary, not a pond).
DATASET_NOTE = "estuarine monitoring data" if DATASET == "real" else "synthetic pond"

FIG_DIR = ROOT / "figures" / DATASET
MODEL_DIR = ROOT / "models" / DATASET
REPORT_DIR = ROOT / "reports" / DATASET
METRICS_JSON = REPORT_DIR / "metrics.json"
for _d in (DATA_DIR, RAW_DIR, FIG_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------- data schema ----
TIME_COL = "timestamp"
LABEL_COL = "quality_label"
FEATURES = ["ph", "temperature_c", "do_mg_l", "salinity_ppt"]
PRETTY = {"ph": "pH", "temperature_c": "Temperature (°C)",
          "do_mg_l": "DO (mg/L)", "salinity_ppt": "Salinity (ppt)"}

# ------------------------------------------- domain rules (P. vannamei) ----
# Inclusive (low, high) ranges. Inside "good" -> Good; else inside
# "moderate" -> Moderate; otherwise Critical.
INF = float("inf")
BANDS = {
    "ph":            {"good": (7.5, 8.5),  "moderate": (7.0, 9.0)},
    "temperature_c": {"good": (28.0, 32.0), "moderate": (26.0, 34.0)},
    "do_mg_l":       {"good": (5.0, INF),  "moderate": (3.0, INF)},
    "salinity_ppt":  {"good": (10.0, 25.0), "moderate": (5.0, 35.0)},
}
CLASSES = ["Good", "Moderate", "Critical"]          # index = severity 0, 1, 2

# Oxygen states for 06: Anoxia < 2, Hypoxia 2-4, Normoxia >= 4 mg/L.
DO_ANOXIA, DO_HYPOXIA = 2.0, 4.0
DO_STATES = ["Anoxia", "Hypoxia", "Normoxia"]

# 05, 06, 08: skip any model window that touches a sensor gap longer than this.
MAX_GAP_H = 6


def severity(values, param: str) -> np.ndarray:
    """Severity of one parameter: 0 Good, 1 Moderate, 2 Critical, NaN if missing."""
    v = np.asarray(values, dtype=float)
    (g_lo, g_hi), (m_lo, m_hi) = BANDS[param]["good"], BANDS[param]["moderate"]
    sev = np.where((v >= g_lo) & (v <= g_hi), 0.0,
                   np.where((v >= m_lo) & (v <= m_hi), 1.0, 2.0))
    return np.where(np.isnan(v), np.nan, sev)


def label_frame(df: pd.DataFrame) -> pd.Series:
    """Pond label per row = the worst parameter. NaN if any reading is missing."""
    sev = np.column_stack([severity(df[c], c) for c in FEATURES])
    worst = sev.max(axis=1)                     # NaN propagates: no partial labels
    names = np.array(CLASSES, dtype=object)
    out = np.full(len(df), np.nan, dtype=object)
    ok = ~np.isnan(worst)
    out[ok] = names[worst[ok].astype(int)]
    return pd.Series(out, index=df.index, name=LABEL_COL)


def label_reading(ph: float, temperature_c: float, do_mg_l: float, salinity_ppt: float) -> str:
    """Label one reading (used by recommend.py and the dashboard)."""
    row = pd.DataFrame([[ph, temperature_c, do_mg_l, salinity_ppt]], columns=FEATURES)
    return label_frame(row).iloc[0]


def oxygen_state(do_mg_l) -> np.ndarray:
    """0 Anoxia, 1 Hypoxia, 2 Normoxia; NaN where DO is missing."""
    v = np.asarray(do_mg_l, dtype=float)
    state = np.where(v < DO_ANOXIA, 0.0, np.where(v < DO_HYPOXIA, 1.0, 2.0))
    return np.where(np.isnan(v), np.nan, state)


# ------------------------------------------------------- colours / plots ----
STATUS_COLOURS = {"Good": "#2a9d4a", "Moderate": "#e8a33d", "Critical": "#c0392b"}
SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]
FIG_DPI = 300


def apply_style() -> None:
    """Same look for every figure. Agg backend: saves PNGs, never opens a window."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cycler import cycler
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": FIG_DPI, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
        "axes.prop_cycle": cycler(color=SERIES_COLOURS),
    })


def save_fig(fig, name: str) -> Path:
    """Save to figures/<dataset>/<name>.png at 300 DPI and print where it went."""
    import matplotlib.pyplot as plt
    path = FIG_DIR / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT).as_posix()}")
    return path


# ---------------------------------------------------------------- helpers ----
def section(title: str) -> None:
    """Console divider so each script's output reads in clear sections."""
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def _plain(obj):
    """Make numpy scalars/arrays and Paths JSON-friendly."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"cannot serialise {type(obj).__name__}")


def record_metric(name: str, values: dict) -> None:
    """Write one block of headline numbers to reports/<dataset>/metrics.json."""
    metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8")) if METRICS_JSON.exists() else {}
    metrics[name] = values
    METRICS_JSON.write_text(json.dumps(metrics, indent=2, default=_plain), encoding="utf-8")


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Read a pipeline-format CSV (default: the active dataset), sorted by time."""
    path = Path(path) if path else DATA_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 01_generate_dataset.py (synthetic) or "
            "tools/fetch_nerrs.py + 00_prepare_dataset.py (real) first.")
    df = pd.read_csv(path, parse_dates=[TIME_COL])
    return df.sort_values(TIME_COL).reset_index(drop=True)


def interpolate_short_gaps(df: pd.DataFrame, max_gap_h: int) -> pd.DataFrame:
    """Linearly fill runs of at most max_gap_h missing hours; longer gaps stay NaN.

    Expects a regular hourly grid. Used by 00 (3 h, for labelling) and by
    05/06/08 (MAX_GAP_H, before windows that touch a longer gap are skipped).
    """
    out = df.copy()
    for col in out.columns:
        s = out[col].astype(float)
        gap = s.isna()
        run_len = gap.groupby((~gap).cumsum()).transform("sum")   # length of each NaN run
        filled = s.interpolate(limit_area="inside")               # never extrapolates
        out[col] = s.mask(gap & (run_len <= max_gap_h), filled)
    return out


if __name__ == "__main__":
    section("Active settings")
    print(f"  dataset      : {DATASET} ({DATASET_NOTE})")
    print(f"  data file    : {DATA_CSV.relative_to(ROOT).as_posix()} (exists: {DATA_CSV.exists()})")
    print(f"  figures  ->  : {FIG_DIR.relative_to(ROOT).as_posix()}/")
    print(f"  models   ->  : {MODEL_DIR.relative_to(ROOT).as_posix()}/")
    print(f"  reports  ->  : {REPORT_DIR.relative_to(ROOT).as_posix()}/")
    print(f"  seed {SEED}, MAX_GAP_H {MAX_GAP_H}")

    section("Labelling rule self-check (worst parameter wins)")
    cases = [  # ph, temp, DO, salinity, expected label, why
        (8.0, 30.0, 6.0, 15.0, "Good", "everything in the Good band"),
        (8.0, 30.0, 4.0, 15.0, "Moderate", "DO between 3 and 5"),
        (8.0, 30.0, 2.5, 15.0, "Critical", "DO below 3"),
        (9.2, 30.0, 6.0, 15.0, "Critical", "pH above 9"),
        (8.0, 33.0, 6.0, 15.0, "Moderate", "temperature 32-34"),
        (8.0, 30.0, 6.0, 4.0, "Critical", "salinity below 5"),
        (7.5, 28.0, 5.0, 10.0, "Good", "band edges are inclusive"),
        (7.2, 30.0, 2.5, 15.0, "Critical", "Moderate pH + Critical DO -> worst"),
    ]
    for ph, t, do, s, want, why in cases:
        got = label_reading(ph, t, do, s)
        assert got == want, f"{(ph, t, do, s)} -> {got}, expected {want}"
        print(f"  pH {ph:<4} T {t:<5} DO {do:<4} S {s:<5} -> {got:<9} ({why})")
    assert pd.isna(label_reading(8.0, 30.0, float("nan"), 15.0)), "missing reading must give no label"
    print("  missing DO -> no label (NaN)")
    assert list(oxygen_state([1.9, 2.0, 3.9, 4.0])) == [0, 1, 1, 2]
    print("  oxygen states at 1.9 / 2.0 / 3.9 / 4.0 mg/L -> Anoxia / Hypoxia / Hypoxia / Normoxia")
    print(f"\nAll {len(cases) + 2} checks passed.")
