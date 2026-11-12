"""00 - Data adapter: turn any pre-collected CSV/Excel file into pipeline format.

Purpose
    Field data never arrives in the pipeline's format. This script reads a
    CSV or Excel file, works out which column is which, converts units,
    removes physically impossible values, averages to an hourly grid, labels
    each hour and writes data/pond_dataset.csv. After that every numbered
    script runs on the new data unchanged (AQUABOT_DATASET=real).

    Steps: guess columns by name -> convert units (°F, DO % saturation,
    conductivity) -> physical QC bounds -> hourly means on a complete hourly
    grid -> label. Gaps of up to 3 h are interpolated FOR LABELLING ONLY; the
    saved readings are never filled in, and hours inside longer gaps stay in
    the file with an empty label rather than being dropped (dropping them
    would silently glue the two sides of an outage together).

Paper link
    None. It is the plug-in point for a real pond logger (see README, "how to
    plug in a new dataset").

Outputs
    data/pond_dataset.csv (or --output): timestamp, ph, temperature_c,
    do_mg_l, salinity_ppt, quality_label. Round-trip test:
    tools/test_prepare_roundtrip.py.

Example
    python 00_prepare_dataset.py data/raw/nerrs_apadb_2013_jun_sep.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
import water_chem as wc

# Boundary-aware patterns: "DO_mgl" and "D.O." are DO, "door" and "dose" are not.
# (?<![a-z]) / (?![a-z]) mean "no letter directly before / after".
PATTERNS = {
    "time": r"date|time",
    "ph": r"(?<![a-z])ph(?![a-z])",
    "temp": r"temp|(?<![a-z])deg(?![a-z])",
    "do": r"(?<![a-z])d\.?o\.?(?![a-z])|oxygen|dissolved",
    "salinity": r"sal|cond|(?<![a-z])ec(?![a-z])",
}
# When several columns match, prefer the one whose name fits the stated unit.
UNIT_HINTS = {"pct": r"pct|%|sat|perc", "mgl": r"mg", "ppt": r"sal|ppt|psu",
              "ec": r"cond|(?<![a-z])ec(?![a-z])"}
# Physical QC bounds: outside these a reading is a sensor fault, not water.
QC_BOUNDS = {"ph": (0.0, 14.0), "temperature_c": (-2.0, 45.0),
             "do_mg_l": (0.0, 25.0), "salinity_ppt": (0.0, 60.0)}
LABEL_GAP_H = 3


def guess_column(columns, role: str, hint: str | None = None) -> str | None:
    hits = [c for c in columns if re.search(PATTERNS[role], str(c).lower())]
    if hint and len(hits) > 1:
        preferred = [c for c in hits if re.search(UNIT_HINTS[hint], str(c).lower())]
        hits = preferred or hits
    return hits[0] if hits else None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert a CSV/Excel file to the Aquabot pipeline format.")
    ap.add_argument("input", type=Path)
    ap.add_argument("--output", type=Path, default=cfg.REAL_CSV)
    for role in ("time", "ph", "temp", "do", "salinity"):
        ap.add_argument(f"--{role}", help=f"name of the {role} column (overrides the guess)")
    ap.add_argument("--temp-unit", choices=["C", "F"], default="C")
    ap.add_argument("--do-unit", choices=["mgl", "pct"], default="mgl")
    ap.add_argument("--salinity-unit", default="ppt",
                    choices=["ppt", "ec_us", "ec_ms", "ec25_us", "ec25_ms"],
                    help="ec = conductivity at water temperature, ec25 = specific "
                         "conductance at 25 °C; us = µS/cm, ms = mS/cm")
    ap.add_argument("--assume-salinity", type=float,
                    help="constant salinity (ppt) when the file has no salinity column")
    ap.add_argument("--dayfirst", action="store_true", help="dates are DD/MM/YYYY")
    ap.add_argument("--label-col", help="column holding expert grades")
    ap.add_argument("--label-map", help='e.g. "A:Good,B:Moderate,C:Critical"')
    ap.add_argument("--sheet", default=0, help="Excel sheet name or index")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg.section(f"00 Prepare dataset: {args.input.name}")
    if args.input.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(args.input, sheet_name=args.sheet)
    else:
        raw = pd.read_csv(args.input)
    print(f"  read {len(raw):,} rows, {len(raw.columns)} columns")

    # ---- 1. which column is which --------------------------------------
    hints = {"do": args.do_unit, "salinity": "ppt" if args.salinity_unit == "ppt" else "ec"}
    cols = {r: getattr(args, r) or guess_column(raw.columns, r, hints.get(r))
            for r in ("time", "ph", "temp", "do", "salinity")}
    cfg.section("Column mapping")
    for role, col in cols.items():
        how = "given" if getattr(args, role) else "guessed"
        print(f"  {role:<9} <- {col!r} ({how})" if col else f"  {role:<9} <- NOT FOUND")
    if cols["salinity"] is None and args.assume_salinity is not None:
        print(f"  salinity  := constant {args.assume_salinity} ppt (--assume-salinity)")
    missing = [r for r, c in cols.items()
               if c is None and not (r == "salinity" and args.assume_salinity is not None)]
    if missing:
        raise SystemExit(f"Could not find column(s) for: {', '.join(missing)}. "
                         "Name them with --time/--ph/--temp/--do/--salinity.")
    bad = [c for c in cols.values() if c is not None and c not in raw.columns]
    if bad:
        raise SystemExit(f"Column(s) not in the file: {bad}. Available: {list(raw.columns)}")

    # ---- 2. units (done on the raw readings: the conversions are not linear)
    cfg.section("Unit conversion")
    num = lambda c: pd.to_numeric(raw[c], errors="coerce").to_numpy(float)
    df = pd.DataFrame({cfg.TIME_COL: pd.to_datetime(raw[cols["time"]], dayfirst=args.dayfirst,
                                                    errors="coerce")})
    df["ph"] = num(cols["ph"])
    temp = num(cols["temp"])
    df["temperature_c"] = wc.fahrenheit_to_celsius(temp) if args.temp_unit == "F" else temp
    print(f"  temperature: {'°F -> °C' if args.temp_unit == 'F' else '°C, unchanged'}")
    if cols["salinity"] is None:
        df["salinity_ppt"] = args.assume_salinity
    else:
        sal, unit = num(cols["salinity"]), args.salinity_unit
        if unit != "ppt":
            ms_cm = sal / 1000.0 if unit.endswith("_us") else sal
            sal = (wc.salinity_from_specific_conductance(ms_cm) if unit.startswith("ec25")
                   else wc.salinity_from_conductivity(ms_cm, df["temperature_c"].to_numpy()))
        df["salinity_ppt"] = sal
        print(f"  salinity   : {'ppt, unchanged' if unit == 'ppt' else unit + ' -> PSS-78 salinity'}")
    do = num(cols["do"])
    if args.do_unit == "pct":
        do = wc.do_percent_to_mgl(do, df["temperature_c"].to_numpy(), df["salinity_ppt"].to_numpy())
    df["do_mg_l"] = do
    print(f"  DO         : {'% saturation -> mg/L (Benson & Krause)' if args.do_unit == 'pct' else 'mg/L, unchanged'}")
    n_bad_time = int(df[cfg.TIME_COL].isna().sum())
    df = df.dropna(subset=[cfg.TIME_COL])
    print(f"  timestamps : {n_bad_time} unreadable rows dropped; "
          f"{df[cfg.TIME_COL].min()} to {df[cfg.TIME_COL].max()}")

    # ---- 3. physical QC bounds -----------------------------------------
    cfg.section("Physical QC bounds")
    for col, (lo, hi) in QC_BOUNDS.items():
        out = (df[col] < lo) | (df[col] > hi)
        df.loc[out, col] = np.nan
        print(f"  {cfg.PRETTY[col]:<18} allowed {lo:g} to {hi:g}: {int(out.sum())} removed")

    # ---- 4. hourly means on a complete hourly grid -----------------------
    hourly = df.set_index(cfg.TIME_COL).sort_index()[cfg.FEATURES].resample("1h").mean()
    cfg.section("Hourly grid")
    print(f"  {len(hourly):,} hourly rows, {hourly.index[0]} to {hourly.index[-1]}")
    print("  hours with no reading: " + ", ".join(
        f"{cfg.PRETTY[c]} {int(hourly[c].isna().sum())}" for c in cfg.FEATURES))

    # ---- 5. labels ---------------------------------------------------------
    cfg.section("Labels")
    if args.label_col:
        if not args.label_map:
            raise SystemExit("--label-col needs --label-map, e.g. \"A:Good,B:Moderate,C:Critical\"")
        mapping = dict(pair.split(":") for pair in args.label_map.split(","))
        wrong = set(mapping.values()) - set(cfg.CLASSES)
        if wrong:
            raise SystemExit(f"--label-map targets must be {cfg.CLASSES}, got {sorted(wrong)}")
        grades = raw.loc[df.index, args.label_col].astype(str).str.strip().map(mapping)
        grades.index = df[cfg.TIME_COL]
        # worst grade seen within the hour, so a short Critical spell is not averaged away
        sev = grades.map({c: i for i, c in enumerate(cfg.CLASSES)}).resample("1h").max()
        labels = sev.map(dict(enumerate(cfg.CLASSES))).reindex(hourly.index)
        print(f"  expert grades from {args.label_col!r} via {mapping}")
    else:
        filled = cfg.interpolate_short_gaps(hourly, LABEL_GAP_H)
        labels = cfg.label_frame(filled)
        print(f"  threshold rule on readings with gaps <= {LABEL_GAP_H} h interpolated "
              "(saved readings are NOT filled)")
    counts = labels.value_counts().reindex(cfg.CLASSES, fill_value=0)
    n_lab = int(labels.notna().sum())
    for name, c in counts.items():
        print(f"  {name:<9}{c:>6,}  {100 * c / max(n_lab, 1):5.1f} % of labelled hours")
    print(f"  unlabelled hours (inside gaps > {LABEL_GAP_H} h): {len(hourly) - n_lab}")

    out = hourly.round(4).reset_index()
    out[cfg.LABEL_COL] = labels.to_numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    cfg.section("Done")
    print(f"  wrote {len(out):,} rows -> {args.output}")


if __name__ == "__main__":
    main()
