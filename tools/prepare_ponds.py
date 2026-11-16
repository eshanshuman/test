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
      5. Stuck sensor at the HOURLY level (the raw-level run check misses a
         stuck sensor whose runs are broken by gaps):
           a. any run of >= 6 consecutive valid hours with an identical DO
              (or identical pH) value is removed; "consecutive" means
              consecutive among the valid hours, so gaps do not break a run;
           b. any exact value that recurs in more than 2 % of a pond's valid
              hours (and at least 6 times) is removed. For DO this applies wherever the value sits
              (floor, ceiling or mid-range): DO is logged to 0.001 mg/L and an
              hourly median does not repeat that often unless the sensor
              outputs a default. pH is coarsely quantised by the probe, so
              repeats are normal there and only a recurring floor or ceiling
              value (at or beyond the pond's 5th / 95th percentile) is removed.
         Removed hours are blanked and marked in do_flag / ph_flag.
      6. Decide which ponds are usable, printing the evidence, on the CLEANED
         hours: a pond that repeats another pond's readings is not
         independent; a pond needs at least 1,000 hours of valid DO.
      7. Drift check: monthly DO mean / sd / min / max per pond after cleaning.

Outputs
    data/ponds/pond_XX.csv for the usable ponds (timestamp, ph, temperature_c,
    do_mg_l, salinity_ppt = 0.0, do_flag, ph_flag), data/ponds/qc_summary.csv
    (all 11 files, with the hours each rule removed) and
    data/ponds/monthly_do.csv.
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
MIN_DO_HOURS = 1000          # a pond needs at least this many hours of valid DO (after cleaning)
HOURLY_RUN = 6               # >= this many consecutive valid hours with one identical value = stuck
RECUR_SHARE = 0.02           # an exact value in more than this share of a pond's valid hours = default output
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


def hourly_qc(h: pd.DataFrame) -> tuple[pd.DataFrame, dict, list[str]]:
    """Hourly stuck-sensor rules (docstring step 5). Returns cleaned frame, counts, evidence lines."""
    out, counts, notes = h.copy(), {}, []
    for col in ("do_mg_l", "ph"):
        v = h[col].dropna()
        run_len = v.groupby((v != v.shift()).cumsum()).transform("size") if len(v) else v
        in_run = run_len >= HOURLY_RUN
        freq = v.value_counts()
        # at least HOURLY_RUN occurrences too, or one reading in a near-empty pond would count as "recurring"
        recurring = freq[(freq > RECUR_SHARE * len(v)) & (freq >= HOURLY_RUN)]
        p5, p95 = (v.quantile(0.05), v.quantile(0.95)) if len(v) else (np.nan, np.nan)
        drop_values = []
        for value, n in recurring.items():
            kind = "floor" if value <= p5 else "ceiling" if value >= p95 else "mid-range"
            removed = col == "do_mg_l" or kind != "mid-range"
            if removed:
                drop_values.append(value)
            if col == "do_mg_l" or kind != "mid-range":       # pH mid-range repeats are quantisation: not listed
                notes.append(f"{cfg.PRETTY[col].split(' ')[0]} value {value:g} in {n:,} of {len(v):,} valid hours "
                             f"({100 * n / len(v):.1f} %), {kind}: removed")
        is_recurring = v.isin(drop_values)
        flag = pd.Series("", index=h.index, dtype=object)
        flag[v.index[in_run & ~is_recurring]] = "stuck_run"
        flag[v.index[~in_run & is_recurring]] = "recurring_value"
        flag[v.index[in_run & is_recurring]] = "stuck_run+recurring_value"
        out[col] = h[col].where(flag == "")
        out[col.split("_")[0] + "_flag"] = flag
        counts[col] = {"before": len(v), "stuck_run": int(in_run.sum()), "recurring": int(is_recurring.sum()),
                       "total": int((flag != "").sum())}
    return out, counts, notes


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
    raws, hours, hourly_notes, rows = {}, {}, {}, []
    print(f"  {'pond':>4} {'raw rows':>9} {'DO kept':>8} {'DO range':>9} {'DO stuck':>9} {'pH kept':>8} "
          f"{'T kept':>7} {'DO hours':>9} {'first hour':>17} {'last hour':>17}")
    for pond in SHA256:
        raw = load_raw(pond)
        clean, removed = quality_control(raw)
        h, hq, notes = hourly_qc(hourly(clean))
        raws[pond], hours[pond], hourly_notes[pond] = raw, h, notes
        do = h["do_mg_l"].dropna()
        kept = {c: 100 * clean[c].notna().sum() / max(len(raw), 1) for c in RANGES}
        rows.append({"pond": pond, "file": SHA256[pond][0], "raw_rows": len(raw),
                     "do_kept_pct": round(kept["do_mg_l"], 2), "ph_kept_pct": round(kept["ph"], 2),
                     "temp_kept_pct": round(kept["temperature_c"], 2),
                     "do_removed_out_of_range": removed["do_mg_l"]["out_of_range"],
                     "do_removed_stuck": removed["do_mg_l"]["stuck"],
                     "do_hours_before_hourly_qc": hq["do_mg_l"]["before"],
                     "do_removed_stuck_run": hq["do_mg_l"]["stuck_run"],
                     "do_removed_recurring": hq["do_mg_l"]["recurring"],
                     "do_removed_hourly_total": hq["do_mg_l"]["total"],
                     "ph_hours_before_hourly_qc": hq["ph"]["before"],
                     "ph_removed_stuck_run": hq["ph"]["stuck_run"],
                     "ph_removed_recurring": hq["ph"]["recurring"],
                     "ph_removed_hourly_total": hq["ph"]["total"],
                     "hours_with_do": len(do),
                     "do_p5": round(float(do.quantile(0.05)), 3) if len(do) else np.nan,
                     "do_median": round(float(do.median()), 3) if len(do) else np.nan,
                     "do_p95": round(float(do.quantile(0.95)), 3) if len(do) else np.nan,
                     "hours_below_4": int((do < cfg.DO_HYPOXIA).sum()),
                     "hours_below_2": int((do < cfg.DO_ANOXIA).sum())})
        print(f"  {pond:>4} {len(raw):>9,} {kept['do_mg_l']:>7.1f}% {removed['do_mg_l']['out_of_range']:>9,} "
              f"{removed['do_mg_l']['stuck']:>9,} {kept['ph']:>7.1f}% {kept['temperature_c']:>6.1f}% "
              f"{len(do):>9,} {str(h.index[0])[:16]:>17} {str(h.index[-1])[:16]:>17}")

    cfg.section(f"5. Hourly stuck-sensor rules (runs >= {HOURLY_RUN} h; values in > {100 * RECUR_SHARE:g} % of hours)")
    summary = pd.DataFrame(rows).set_index("pond")
    print(f"  {'pond':>4} {'DO h before':>12} {'run':>6} {'recurring':>10} {'removed':>8} {'DO h after':>11}   "
          f"{'pH h before':>12} {'run':>6} {'recurring':>10} {'removed':>8}")
    for pond, r in summary.iterrows():
        print(f"  {pond:>4} {r['do_hours_before_hourly_qc']:>12,} {r['do_removed_stuck_run']:>6,} "
              f"{r['do_removed_recurring']:>10,} {r['do_removed_hourly_total']:>8,} {r['hours_with_do']:>11,}   "
              f"{r['ph_hours_before_hourly_qc']:>12,} {r['ph_removed_stuck_run']:>6,} "
              f"{r['ph_removed_recurring']:>10,} {r['ph_removed_hourly_total']:>8,}")
    print("  ('run' and 'recurring' overlap: an hour can break both rules; 'removed' counts each hour once)")
    for pond, notes in hourly_notes.items():
        for note in notes:
            print(f"  pond {pond}: {note}")

    cfg.section("6. Which ponds are usable (evidence, on cleaned hours)")
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
            cut = int(summary.loc[pond, "do_removed_hourly_total"])
            reasons[pond] = (f"only {n:,} hours of valid DO (needs {MIN_DO_HOURS:,})"
                             + (f" after {cut:,} stuck-sensor hours were removed" if cut else ""))
    usable = [p for p in SHA256 if p not in reasons]
    for pond in SHA256:
        print(f"  pond {pond:>2}: " + (f"EXCLUDED, {reasons[pond]}" if pond in reasons
                                       else f"used ({int(summary.loc[pond, 'hours_with_do']):,} h of valid DO)"))
    summary["used"] = [p in usable for p in summary.index]
    summary["exclusion_reason"] = [reasons.get(p, "") for p in summary.index]
    if tuple(usable) != cfg.POND_IDS:
        raise SystemExit(f"Usable ponds {usable} differ from config.POND_IDS {cfg.POND_IDS}: "
                         "update config.POND_IDS to match the evidence above, then re-run. Stopping.")
    print(f"  -> usable ponds {usable} match config.POND_IDS")

    cfg.section("Hourly DO in the usable ponds")
    cfg.PONDS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  {'pond':>4} {'grid hours':>11} {'DO hours':>9} {'p5':>6} {'median':>7} {'p95':>6} "
          f"{'< 4 mg/L':>9} {'< 2 mg/L':>9}")
    for stale in cfg.PONDS_DIR.glob("pond_*.csv"):            # no file may survive for a pond that is not usable
        stale.unlink()
    for pond in usable:
        hours[pond].reset_index().rename(columns={"index": cfg.TIME_COL}).to_csv(
            cfg.PONDS_DIR / f"pond_{pond:02d}.csv", index=False)
        s = summary.loc[pond]
        print(f"  {pond:>4} {len(hours[pond]):>11,} {int(s['hours_with_do']):>9,} {s['do_p5']:>6.2f} "
              f"{s['do_median']:>7.2f} {s['do_p95']:>6.2f} {int(s['hours_below_4']):>9,} {int(s['hours_below_2']):>9,}")
    summary.reset_index().to_csv(cfg.PONDS_DIR / "qc_summary.csv", index=False)

    cfg.section("7. Drift check: monthly DO after cleaning (mg/L)")
    monthly = []
    for pond in usable:
        do = hours[pond]["do_mg_l"].dropna()
        print(f"  pond {pond}")
        for month, g in do.groupby(do.index.to_period("M")):
            monthly.append({"pond": pond, "month": str(month), "hours": len(g), "mean": round(float(g.mean()), 3),
                            "sd": round(float(g.std()), 3), "min": round(float(g.min()), 3),
                            "max": round(float(g.max()), 3)})
            m = monthly[-1]
            print(f"    {m['month']}  hours {m['hours']:>4}   mean {m['mean']:>6.2f}   sd {m['sd']:>5.2f}   "
                  f"min {m['min']:>5.2f}   max {m['max']:>6.2f}")
    pd.DataFrame(monthly).to_csv(cfg.PONDS_DIR / "monthly_do.csv", index=False)
    print(f"\n  wrote data/ponds/pond_XX.csv for {len(usable)} ponds, qc_summary.csv and monthly_do.csv")


if __name__ == "__main__":
    main()
