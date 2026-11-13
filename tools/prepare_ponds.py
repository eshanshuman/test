"""Prepare the real fish-pond dataset (Udanor et al. 2022) for the DO tasks.

Purpose
    The synthetic pond is designed and the NERRS data is an estuary; this is
    the only dataset here from real STOCKED ponds: "Sensor Based Aquaponics
    Fish Pond Datasets" (Udanor et al. 2022, Data in Brief 43:108400; Kaggle
    doi:10.34740/KAGGLE/DSV/3748790; CC BY 4.0). African catfish (Clarias
    gariepinus) in indoor tarpaulin ponds, University of Nigeria Nsukka,
    June-December 2021, one reading every few seconds to minutes. Freshwater,
    no salinity sensor, so it is used for dissolved-oxygen tasks only.

    Steps
      1. Verify each raw file's SHA-256 against the sums published with the
         leakage-aware re-analysis (github.com/MohamedElminur1/
         DL_Models_Fishweight, data/raw/SHA256SUMS). Stop on any mismatch.
      2. Timestamps: first 19 characters of the first column. "CET" and
         "+01:00" stamps are already local (West Africa Time, UTC+1); "UTC"
         stamps are shifted by +1 h.
      3. QC on raw readings: DO 0.1 < x <= 20 mg/L; pH 4-11; temperature
         15-40 °C (the probe writes -127 when unplugged); any run of >= 30
         identical consecutive DO or pH values is a stuck sensor and removed.
      4. Hourly median of the valid readings (an hour needs >= 3 of them).
      5. Decide which ponds are usable, printing the evidence: pond 2 repeats
         pond 1's readings, pond 8 has too few valid hours, ponds 10-12 have
         almost no valid data.

Outputs
    data/ponds/pond_XX.csv for the usable ponds (timestamp, ph, temperature_c,
    do_mg_l, salinity_ppt = 0.0) and data/ponds/qc_summary.csv (all 11 files).
    The raw files in data/raw/udanor/ are git-ignored.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config as cfg

SHA256 = {
    1: ("IoTpond1.csv", "063ea4f0c9fcd24f016fbfc52e5422b655dfb4f3542d40476d99d899655dda76"),
    2: ("IoTPond2.csv", "be6b8d9113d2c32d82a6fe8fa51538bcd1085a1034681e8b7f2d67275436eb63"),
    3: ("IoTPond3.csv", "a1980e80250bfd610274acb3e281ebf93ba2b1e5449c61982bf97d262f8c01be"),
    4: ("IoTPond4.csv", "ff3822f2e28b191e126d9b92fc6ea6a7be36b1fc5924ac97af1a83f7ec598634"),
    6: ("IoTPond6.csv", "5da60c205dc569f147ec9f20b285c869884171741316a3b9af052850b68193ed"),
    7: ("IoTPond7.csv", "8fc3c26241da6ad9bd748c116e06cac00664c0a0d7cfa424fc8453ed5e03bd52"),
    8: ("IoTPond8.csv", "1260a962f979876a1d451a55208bcd9e7fef077e3d13ccfee98863f6afb128a7"),
    9: ("IoTPond9.csv", "9e3aea85c8f2b68504548e3ef273a8cb032a2817118d82da4a6b5758e615bf54"),
    10: ("IoTPond10.csv", "7cd9f8b98a6edf6f2111046f35b19a1304f840c4eaef0b182b5b8922cd601690"),
    11: ("IoTPond11.csv", "70146266d05a99a41bf90f3902b231c81c01af551b216053bc76ffea5299be78"),
    12: ("IoTPond12.csv", "9699eeaf9f5054bc4df034985c4880820ec192c5bd5be0d8247b486eb3e1b5d5"),
}
COLUMN_PATTERNS = {"do_mg_l": r"dis+olved|oxygen", "ph": r"^\s*ph\s*$", "temperature_c": r"temp"}
RANGES = {"do_mg_l": (0.1, 20.0), "ph": (4.0, 11.0), "temperature_c": (15.0, 40.0)}   # DO lower bound exclusive
STUCK_RUN = 30               # >= this many identical consecutive values = stuck sensor (DO, pH)
MIN_PER_HOUR = 3             # valid readings needed for an hourly value
MIN_DO_HOURS = 1000          # a pond needs at least this many hours of valid DO
DUPLICATE_SHARE = 0.5        # a pond repeating more than this share of another pond is not independent


def find_column(columns, pattern: str) -> str:
    hits = [c for c in columns if re.search(pattern, str(c), flags=re.IGNORECASE)]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one column matching {pattern!r}, found {hits}")
    return hits[0]


def stuck_mask(values: pd.Series) -> pd.Series:
    """True where a value belongs to a run of >= STUCK_RUN identical consecutive readings."""
    run_id = (values != values.shift()).cumsum()
    run_len = values.groupby(run_id).transform("size")
    return (run_len >= STUCK_RUN) & values.notna()


def load_raw(pond: int) -> pd.DataFrame:
    """Raw readings in time order: timestamp (local), do_mg_l, ph, temperature_c."""
    raw = pd.read_csv(cfg.PONDS_RAW_DIR / SHA256[pond][0], dtype=str, low_memory=False)
    stamp = raw.iloc[:, 0].astype(str)                       # timestamp is always the FIRST column
    time = pd.to_datetime(stamp.str[:19].str.replace("T", " "), format="%Y-%m-%d %H:%M:%S", errors="coerce")
    time = time + pd.to_timedelta(np.where(stamp.str.contains("UTC"), 1, 0), unit="h")
    out = pd.DataFrame({cfg.TIME_COL: time})
    for name, pattern in COLUMN_PATTERNS.items():
        out[name] = pd.to_numeric(raw[find_column(raw.columns, pattern)], errors="coerce")
    return out.dropna(subset=[cfg.TIME_COL]).sort_values(cfg.TIME_COL, kind="stable").reset_index(drop=True)


def quality_control(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Blank invalid readings; return the cleaned frame and what was removed."""
    clean, removed = raw.copy(), {}
    for col, (lo, hi) in RANGES.items():
        in_range = (raw[col] > lo if col == "do_mg_l" else raw[col] >= lo) & (raw[col] <= hi)
        stuck = stuck_mask(raw[col]) if col in ("do_mg_l", "ph") else pd.Series(False, index=raw.index)
        clean[col] = raw[col].where(in_range & ~stuck)
        removed[col] = {"out_of_range": int((raw[col].notna() & ~in_range).sum()),
                        "stuck": int((stuck & in_range).sum())}
    return clean, removed


def hourly(clean: pd.DataFrame) -> pd.DataFrame:
    """Hourly median of valid readings on a complete hourly grid; < MIN_PER_HOUR readings -> NaN."""
    grouped = clean.set_index(cfg.TIME_COL)[list(RANGES)].resample("1h")
    med = grouped.median().where(grouped.count() >= MIN_PER_HOUR)
    med = med.loc[med.notna().any(axis=1).idxmax():med.notna().any(axis=1)[::-1].idxmax()]
    med["salinity_ppt"] = 0.0                                # freshwater: no salinity sensor
    return med[cfg.FEATURES].round(4)


def main() -> None:
    cfg.section(f"Prepare ponds: Udanor et al. 2022 ({cfg.PONDS_NOTE})")
    if not cfg.PONDS_RAW_DIR.exists():
        raise SystemExit(f"{cfg.PONDS_RAW_DIR} not found. Download the Kaggle dataset "
                         "(doi:10.34740/KAGGLE/DSV/3748790) and unzip its 11 CSV files there.")

    cfg.section("1. SHA-256 check against the published sums")
    for pond, (fname, expected) in SHA256.items():
        path = cfg.PONDS_RAW_DIR / fname
        if not path.exists():
            raise SystemExit(f"missing raw file: {path}")
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"  {fname:<14} {'OK' if got == expected else 'MISMATCH'}  {got[:16]}...")
        if got != expected:
            raise SystemExit(f"SHA-256 mismatch for {fname}: got {got}, expected {expected}. Stopping.")
    print(f"  all {len(SHA256)} files match")

    cfg.section("2-4. Timestamps, quality control, hourly medians")
    raws, hours, rows = {}, {}, []
    print(f"  {'pond':>4} {'raw rows':>9} {'DO kept':>8} {'DO range':>9} {'DO stuck':>9} {'pH kept':>8} "
          f"{'T kept':>7} {'DO hours':>9} {'first hour':>17} {'last hour':>17}")
    for pond in SHA256:
        raw = load_raw(pond)
        clean, removed = quality_control(raw)
        h = hourly(clean)
        raws[pond], hours[pond] = raw, h
        do = h["do_mg_l"].dropna()
        kept = {c: 100 * clean[c].notna().sum() / max(len(raw), 1) for c in RANGES}
        rows.append({"pond": pond, "file": SHA256[pond][0], "raw_rows": len(raw),
                     "do_kept_pct": round(kept["do_mg_l"], 2), "ph_kept_pct": round(kept["ph"], 2),
                     "temp_kept_pct": round(kept["temperature_c"], 2),
                     "do_removed_out_of_range": removed["do_mg_l"]["out_of_range"],
                     "do_removed_stuck": removed["do_mg_l"]["stuck"],
                     "hours_with_do": len(do),
                     "do_p5": round(float(do.quantile(0.05)), 3) if len(do) else np.nan,
                     "do_median": round(float(do.median()), 3) if len(do) else np.nan,
                     "do_p95": round(float(do.quantile(0.95)), 3) if len(do) else np.nan,
                     "hours_below_4": int((do < cfg.DO_HYPOXIA).sum()),
                     "hours_below_2": int((do < cfg.DO_ANOXIA).sum())})
        print(f"  {pond:>4} {len(raw):>9,} {kept['do_mg_l']:>7.1f}% {removed['do_mg_l']['out_of_range']:>9,} "
              f"{removed['do_mg_l']['stuck']:>9,} {kept['ph']:>7.1f}% {kept['temperature_c']:>6.1f}% "
              f"{len(do):>9,} {str(h.index[0])[:16]:>17} {str(h.index[-1])[:16]:>17}")

    cfg.section("5. Which ponds are usable (evidence)")
    summary = pd.DataFrame(rows).set_index("pond")
    reasons = {}
    pairs = {p: set(zip(r[cfg.TIME_COL], r["do_mg_l"])) for p, r in raws.items()}
    print("  Share of each pond's (timestamp, DO) pairs that reappear in another pond's file (> 1 % shown):")
    for a in SHA256:
        for b in SHA256:
            if a < b and pairs[a] and pairs[b]:
                share_a = len(pairs[a] & pairs[b]) / len(pairs[a])
                if share_a > 0.01:
                    print(f"    {100 * share_a:5.1f} % of pond {a}'s pairs are in pond {b}'s file "
                          f"({len(pairs[a] & pairs[b]):,} of {len(pairs[a]):,}; pond {b} has {len(pairs[b]):,})")
                if share_a > DUPLICATE_SHARE:                # the later, larger file is the copy
                    reasons[b] = f"repeats {100 * share_a:.1f} % of pond {a}'s (timestamp, DO) pairs: not independent"
    for pond in SHA256:
        n = int(summary.loc[pond, "hours_with_do"])
        if pond not in reasons and n < MIN_DO_HOURS:
            reasons[pond] = f"only {n:,} hours of valid DO (needs {MIN_DO_HOURS:,})"
    usable = [p for p in SHA256 if p not in reasons]
    for pond in SHA256:
        print(f"  pond {pond:>2}: " + (f"EXCLUDED, {reasons[pond]}" if pond in reasons
                                       else f"used ({int(summary.loc[pond, 'hours_with_do']):,} h of valid DO)"))
    summary["used"] = [p in usable for p in summary.index]
    summary["exclusion_reason"] = [reasons.get(p, "") for p in summary.index]
    if tuple(usable) != cfg.POND_IDS:
        raise SystemExit(f"Usable ponds {usable} differ from config.POND_IDS {cfg.POND_IDS}: "
                         "the data does not support the planned exclusions. Stopping.")
    print(f"  -> usable ponds {usable} match config.POND_IDS")

    cfg.section("Hourly DO in the usable ponds")
    cfg.PONDS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  {'pond':>4} {'grid hours':>11} {'DO hours':>9} {'p5':>6} {'median':>7} {'p95':>6} "
          f"{'< 4 mg/L':>9} {'< 2 mg/L':>9}")
    for pond in usable:
        hours[pond].reset_index().rename(columns={"index": cfg.TIME_COL}).to_csv(
            cfg.PONDS_DIR / f"pond_{pond:02d}.csv", index=False)
        s = summary.loc[pond]
        print(f"  {pond:>4} {len(hours[pond]):>11,} {int(s['hours_with_do']):>9,} {s['do_p5']:>6.2f} "
              f"{s['do_median']:>7.2f} {s['do_p95']:>6.2f} {int(s['hours_below_4']):>9,} {int(s['hours_below_2']):>9,}")
    summary.reset_index().to_csv(cfg.PONDS_DIR / "qc_summary.csv", index=False)
    print(f"\n  wrote data/ponds/pond_XX.csv for {len(usable)} ponds and data/ponds/qc_summary.csv")


if __name__ == "__main__":
    main()
