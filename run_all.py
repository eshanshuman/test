"""run_all.py - Run the whole pipeline from scratch and time every step.

Purpose
    One command that regenerates every figure, model and number in the
    README, so the results can be reproduced and nothing in the reports is
    left over from an older run. Order:
      1. 01 generates the synthetic pond.
      2. For each dataset (synthetic, then real): clear that dataset's
         metrics.json, then run 02, 03, 04, 05 (--epochs 40, and once more
         with --target level for the comparison), 06 three times (+6 h; +1 h;
         +6 h --split time), 07 and recommend.py, with AQUABOT_DATASET set.
      3. 08 (cross-site, real data only).
    Every step is started with the same Python that runs this file
    (sys.executable) and PYTHONIOENCODING=utf-8, so it behaves the same in
    Windows PowerShell as on Linux. A failing step is reported and the run
    carries on. The real-data steps are skipped with a note if
    data/pond_dataset.csv is missing (run tools/fetch_nerrs.py and
    00_prepare_dataset.py first).

Outputs
    Everything the numbered scripts write, plus a summary table with minutes
    per step. Exit code 1 if any step failed.

Run
    python run_all.py            (add --only synthetic or --only real)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PER_DATASET = [
    ("02 EDA", ["02_eda.py"]),
    ("03 Random Forest", ["03_random_forest.py"]),
    ("04 LightGBM", ["04_lightgbm.py"]),
    ("05 LSTM forecast", ["05_lstm_do_forecast.py", "--epochs", "40"]),
    ("05 LSTM forecast (level target)", ["05_lstm_do_forecast.py", "--epochs", "40", "--target", "level"]),
    ("06 Oxygen state +6 h", ["06_do_state_classifier.py", "--horizon", "6"]),
    ("06 Oxygen state +1 h", ["06_do_state_classifier.py", "--horizon", "1"]),
    ("06 Oxygen state +6 h, time split", ["06_do_state_classifier.py", "--horizon", "6", "--split", "time"]),
    ("07 LR vs RF scaling", ["07_lr_vs_rf_scaling.py"]),
    ("Recommendation engine", ["recommend.py"]),
]


def run_step(name: str, args: list[str], dataset: str | None) -> dict:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if dataset:
        env["AQUABOT_DATASET"] = dataset
    label = f"{name} [{dataset}]" if dataset else name
    print(f"\n{'#' * 72}\n# {label}\n# python {' '.join(args)}\n{'#' * 72}", flush=True)
    start = time.time()
    try:
        code = subprocess.run([sys.executable, *args], cwd=ROOT, env=env).returncode
    except Exception as exc:                              # e.g. the script file is missing
        print(f"could not start: {exc}")
        code = -1
    return {"step": label, "ok": code == 0, "minutes": (time.time() - start) / 60}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the whole Aquabot pipeline")
    ap.add_argument("--only", choices=["synthetic", "real"], help="run one dataset only")
    args = ap.parse_args()
    datasets = [args.only] if args.only else ["synthetic", "real"]
    results, t0 = [], time.time()

    if "synthetic" in datasets:
        metrics = ROOT / "reports" / "synthetic" / "metrics.json"
        metrics.unlink(missing_ok=True)                   # 01 writes the first block of a fresh file
        results.append(run_step("01 Generate synthetic pond", ["01_generate_dataset.py"], None))

    for dataset in datasets:
        if dataset == "real":
            if not (ROOT / "data" / "pond_dataset.csv").exists():
                print("\nSKIPPING real data: data/pond_dataset.csv not found. Run "
                      "tools/fetch_nerrs.py and 00_prepare_dataset.py first.")
                results.append({"step": "real dataset (missing)", "ok": False, "minutes": 0.0})
                continue
            (ROOT / "reports" / "real" / "metrics.json").unlink(missing_ok=True)
        for name, step_args in PER_DATASET:
            results.append(run_step(name, step_args, dataset))

    if "real" in datasets and (ROOT / "data" / "pond_dataset_site2.csv").exists():
        results.append(run_step("08 Cross-site", ["08_cross_site.py"], "real"))

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    print(f"  {'step':<52}{'result':>8}{'minutes':>10}")
    for r in results:
        print(f"  {r['step']:<52}{'ok' if r['ok'] else 'FAILED':>8}{r['minutes']:>10.1f}")
    failed = [r["step"] for r in results if not r["ok"]]
    print(f"\n  {len(results) - len(failed)} of {len(results)} steps succeeded "
          f"in {(time.time() - t0) / 60:.1f} minutes.")
    if failed:
        print("  FAILED: " + "; ".join(failed))
    print("  Next: python tools/summarize_results.py")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
