"""Round-trip test for 00_prepare_dataset.py.

Purpose
    Proves the data adapter is correct, not just that it runs. It takes the
    synthetic pond (whose values we know), deliberately makes a mess of it the
    way a field logger export would be:
      * 30-minute rows instead of hourly, in shuffled order
      * dates written DD/MM/YYYY HH:MM
      * temperature in °F, DO in % saturation, conductivity in µS/cm
      * awkward column names ("DO_%sat", "Water Temp (F)", ...)
    then runs the adapter on that file through its command line and checks the
    result equals the original: same hours, same gaps, same values (to within
    the rounding of the messy file) and the same threshold labels.

Outputs
    data/raw/_roundtrip_messy.csv and data/_roundtrip_out.csv (test artefacts),
    and a PASS/FAIL report. Exit code 1 on failure.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config as cfg
import water_chem as wc

MESSY = cfg.RAW_DIR / "_roundtrip_messy.csv"
OUT = cfg.DATA_DIR / "_roundtrip_out.csv"
TOLERANCE = {"ph": 1e-6, "temperature_c": 0.01, "do_mg_l": 0.01, "salinity_ppt": 0.01}


def make_messy(clean: pd.DataFrame) -> pd.DataFrame:
    # each hourly reading appears twice (hh:00 and hh:30), so the hourly mean is the original
    half = pd.concat([clean, clean.assign(**{cfg.TIME_COL: clean[cfg.TIME_COL] + pd.Timedelta(minutes=30)})])
    t_c, sal = half["temperature_c"].to_numpy(), half["salinity_ppt"].to_numpy()
    cond_us = wc.conductivity_from_salinity(np.nan_to_num(sal, nan=15.0), np.nan_to_num(t_c, nan=30.0)) * 1000
    cond_us[np.isnan(sal) | np.isnan(t_c)] = np.nan
    messy = pd.DataFrame({
        "Date Time": half[cfg.TIME_COL].dt.strftime("%d/%m/%Y %H:%M"),
        "Water Temp (F)": (t_c * 9 / 5 + 32).round(3),
        "pH value": half["ph"].to_numpy(),
        "DO_%sat": wc.do_mgl_to_percent(half["do_mg_l"].to_numpy(), t_c, sal).round(3),
        "EC_uS_cm": cond_us.round(1),
        "operator notes": "ok",
    })
    return messy.sample(frac=1.0, random_state=cfg.SEED)          # shuffled row order


def main() -> None:
    cfg.section("Round-trip test of 00_prepare_dataset.py")
    clean = pd.read_csv(cfg.SYNTHETIC_CSV, parse_dates=[cfg.TIME_COL])
    messy = make_messy(clean)
    messy.to_csv(MESSY, index=False)
    print(f"  messy file: {len(messy):,} rows -> {MESSY.relative_to(cfg.ROOT).as_posix()}")
    print(messy.head(3).to_string(index=False))

    cmd = [sys.executable, str(cfg.ROOT / "00_prepare_dataset.py"), str(MESSY), "--output", str(OUT),
           "--dayfirst", "--temp-unit", "F", "--do-unit", "pct", "--salinity-unit", "ec_us"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    run = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
    print(run.stdout)
    if run.returncode != 0:
        print(run.stderr)
        raise SystemExit("FAIL: the adapter crashed")

    cfg.section("Comparison with the original synthetic data")
    back = pd.read_csv(OUT, parse_dates=[cfg.TIME_COL])
    ok = True

    def check(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    check("row count", len(back) == len(clean), f"{len(back):,} vs {len(clean):,}")
    check("timestamps", back[cfg.TIME_COL].equals(clean[cfg.TIME_COL]), "identical hourly grid, in order")
    # A DO % or conductivity value cannot be written when temperature (or salinity)
    # is missing in the same row, so those cells are legitimately lost in the mess.
    t_na, s_na = clean["temperature_c"].isna(), clean["salinity_ppt"].isna()
    expect_na = {"ph": clean["ph"].isna(), "temperature_c": t_na,
                 "salinity_ppt": s_na | t_na, "do_mg_l": clean["do_mg_l"].isna() | t_na | s_na}
    for col in cfg.FEATURES:
        same_gaps = bool((back[col].isna() == expect_na[col]).all())
        err = float((back[col] - clean[col]).abs().max())
        check(cfg.PRETTY[col], same_gaps and err <= TOLERANCE[col],
              f"gaps as expected: {same_gaps}; max abs error {err:.2e} (limit {TOLERANCE[col]:g})")

    # Labels: the adapter must give exactly what the threshold rule gives on the
    # same hourly values (short gaps interpolated).
    expect = cfg.label_frame(cfg.interpolate_short_gaps(back[cfg.FEATURES], 3))
    same = (back[cfg.LABEL_COL].fillna("-") == expect.fillna("-"))
    check("threshold labels", bool(same.all()), f"{int(same.sum()):,} of {len(same):,} match the rule")
    both = back[cfg.LABEL_COL].notna()
    agree = float((back.loc[both, cfg.LABEL_COL] == clean.loc[both, cfg.LABEL_COL]).mean())
    print(f"  [info] agreement with the generator's noise-free labels: {100 * agree:.2f} % "
          "(below 100 because sensor noise moves readings across thresholds)")

    print("\nROUND-TRIP " + ("PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
