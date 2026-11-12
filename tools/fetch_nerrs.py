"""Download real water-quality data (NOAA NERRS SWMP, Apalachicola Bay).

Purpose
    No open prawn-pond dataset has pH, temperature, DO and salinity together,
    so the "real" dataset is NOAA's National Estuarine Research Reserve System
    (NERRS) System-Wide Monitoring Program record from Apalachicola Bay,
    Florida, as shipped with the SWMPr R package (Beck 2016):
      apadbwq = Dry Bar   (main dataset)
      apacpwq = Cat Point (second site, for the cross-site test in 08)
    The script keeps a value only when its NERRS QC flag starts with "<0>"
    (passed QC), cuts June-September 2013 and writes plain CSVs.

    IMPORTANT: this is an ESTUARY, not a pond: tides, no aerators, no
    feeding. Every result from it is labelled "estuarine monitoring data".

Paper link
    None reproduced. It gives the pipeline measured sensor data with real
    noise, drift and gaps, which the synthetic pond cannot.

Outputs
    data/raw/nerrs_apadb_2013_jun_sep.csv, data/raw/nerrs_apacp_2013_jun_sep.csv
    (15-minute rows: datetimestamp, temp, sal, do_mgl, ph) and
    data/raw/README.md (provenance, citations). The downloaded .RData files
    stay in data/raw/ but are git-ignored.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pyreadr
import requests

import config as cfg

BASE_URL = "https://raw.githubusercontent.com/fawda123/SWMPr/master/data/"
SITES = {"apadbwq": ("Dry Bar", "nerrs_apadb_2013_jun_sep.csv"),
         "apacpwq": ("Cat Point", "nerrs_apacp_2013_jun_sep.csv")}
VALUE_COLS = ["temp", "sal", "do_mgl", "ph"]
WINDOW = ("2013-06-01 00:00", "2013-09-30 23:45")
# SWMPr stores the timestamps as Eastern Standard Time with no daylight saving
# (R time zone "America/Jamaica"); pyreadr hands them back in UTC. Shifting by
# the fixed offset restores the local clock, so "hour of day" means local hour.
UTC_TO_LOCAL = pd.Timedelta(hours=-5)


def fetch(name: str) -> Path:
    path = cfg.RAW_DIR / f"{name}.RData"
    if path.exists():
        print(f"  {path.name}: already downloaded")
        return path
    resp = requests.get(BASE_URL + path.name, timeout=120)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    print(f"  {path.name}: downloaded {len(resp.content):,} bytes")
    return path


def clean(name: str, path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pyreadr.read_r(str(path))[name]
    raw["datetimestamp"] = pd.to_datetime(raw["datetimestamp"]) + UTC_TO_LOCAL
    lo, hi = pd.Timestamp(WINDOW[0]), pd.Timestamp(WINDOW[1])
    df = raw[(raw["datetimestamp"] >= lo) & (raw["datetimestamp"] <= hi)].copy()
    stats = {"rows": len(df)}
    for col in VALUE_COLS:
        passed = df[f"f_{col}"].astype(str).str.startswith("<0>")
        had_value = df[col].notna()
        df[col] = df[col].where(passed)          # anything not flagged <0> becomes NaN
        stats[col] = {"kept": int(df[col].notna().sum()),
                      "rejected_by_qc": int((had_value & ~passed).sum())}
    return df[["datetimestamp"] + VALUE_COLS].reset_index(drop=True), stats


def write_readme(all_stats: dict) -> None:
    lines = [
        "# data/raw: provenance", "",
        "Written by `tools/fetch_nerrs.py`. Do not edit by hand; re-run the script.", "",
        "## Source", "",
        "NOAA National Estuarine Research Reserve System (NERRS), System-Wide",
        "Monitoring Program (SWMP), Apalachicola Bay, Florida, water-quality",
        "sondes, as redistributed in the SWMPr R package:", "",
        f"- `{BASE_URL}apadbwq.RData` (Dry Bar, main dataset)",
        f"- `{BASE_URL}apacpwq.RData` (Cat Point, cross-site test)", "",
        "## What was done", "",
        "- Columns used: `datetimestamp`, `temp` (°C), `sal` (ppt/psu), `do_mgl` (mg/L), `ph`.",
        "- A value is kept only if its QC flag (`f_temp`, `f_sal`, `f_do_mgl`, `f_ph`)",
        "  starts with `<0>` (passed NERRS QC); every other value is blank.",
        "- Timestamps shifted from UTC to local standard time (UTC-5, no daylight saving).",
        f"- Window kept: {WINDOW[0]} to {WINDOW[1]}, 15-minute readings.", "",
        "| File | Site | Rows | temp kept | sal kept | do_mgl kept | ph kept |",
        "|---|---|---|---|---|---|---|"]
    for name, (site, fname) in SITES.items():
        st = all_stats[name]
        lines.append(f"| `{fname}` | {site} | {st['rows']:,} | "
                     + " | ".join(f"{st[c]['kept']:,}" for c in VALUE_COLS) + " |")
    lines += [
        "", "## Caveat", "",
        "This is **estuarine monitoring data**, not a prawn pond: the water is",
        "tidal, and there are no aerators, feeding or stocking. It tests the",
        "pipeline on real sensor behaviour; it does not describe a vannamei pond.", "",
        "## Citations", "",
        "- NOAA National Estuarine Research Reserve System (NERRS). System-wide",
        "  Monitoring Program. Data accessed from the NOAA NERRS Centralized Data",
        "  Management Office website: http://www.nerrsdata.org",
        "- Beck, M. W. (2016). SWMPr: An R package for retrieving, organizing, and",
        "  analyzing environmental data for estuaries. The R Journal, 8(1), 219-232.", ""]
    (cfg.RAW_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("  wrote data/raw/README.md")


def main() -> None:
    cfg.section("Fetch NOAA NERRS SWMP data (estuarine monitoring data)")
    all_stats = {}
    for name, (site, fname) in SITES.items():
        df, stats = clean(name, fetch(name))
        df.to_csv(cfg.RAW_DIR / fname, index=False)
        all_stats[name] = stats
        print(f"\n  {site} ({name}): {stats['rows']:,} rows, "
              f"{df['datetimestamp'].min()} to {df['datetimestamp'].max()}")
        for col in VALUE_COLS:
            print(f"    {col:<7} kept {stats[col]['kept']:>6,}   "
                  f"rejected by QC flag {stats[col]['rejected_by_qc']:>5,}")
        by_hour = df.groupby(df["datetimestamp"].dt.hour)["do_mgl"].mean()
        print(f"    local-time check: mean DO lowest at {by_hour.idxmin():02d}:00, "
              f"highest at {by_hour.idxmax():02d}:00")
        print(f"    wrote data/raw/{fname}")
    cfg.section("Provenance")
    write_readme(all_stats)


if __name__ == "__main__":
    main()
