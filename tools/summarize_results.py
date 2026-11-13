"""Print every headline number from the last run as Markdown tables.

Purpose
    The README's results tables must come from a run, never from memory. This
    script reads reports/<dataset>/metrics.json and the CSV reports written by
    the numbered scripts and prints them as Markdown tables, one section per
    dataset (synthetic, real, and the real catfish ponds). It computes nothing new: it only formats what was recorded.

Outputs
    Markdown on the console; with --out FILE the same text is also written to
    that file (default suggestion: reports/results_summary.md).

Run
    python tools/summarize_results.py --out reports/results_summary.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTES = {"synthetic": "synthetic pond", "real": "estuarine monitoring data"}
LINES: list[str] = []


def out(text: str = "") -> None:
    LINES.append(text)


def table(headers: list[str], rows: list[list]) -> None:
    out("| " + " | ".join(headers) + " |")
    out("|" + "---|" * len(headers))
    for row in rows:
        out("| " + " | ".join(str(v) for v in row) + " |")
    out()


def f(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def pm(mean, sd, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def summarize(dataset: str) -> None:
    rep = ROOT / "reports" / dataset
    if not (rep / "metrics.json").exists():
        out(f"## {dataset}: no metrics.json (run run_all.py)\n")
        return
    m = json.loads((rep / "metrics.json").read_text(encoding="utf-8"))
    out(f"## Dataset: {dataset} ({NOTES[dataset]})\n")

    if "02_eda" in m:
        e = m["02_eda"]
        out("### Data (02)\n")
        counts, total = e["class_counts"], sum(e["class_counts"].values())
        table(["Hourly rows", "Labelled", "Good", "Moderate", "Critical", "DO min hour", "DO max hour"],
              [[f"{e['rows']:,}", f"{e['labelled_rows']:,}",
                *[f"{counts[c]:,} ({100 * counts[c] / total:.1f} %)" for c in ("Good", "Moderate", "Critical")],
                f"{e['do_min_hour']:02d}:00", f"{e['do_max_hour']:02d}:00"]])
        table(["Mutual information with label (nats)"] + list(e["mutual_information_nats"]),
              [[""] + [f(v) for v in e["mutual_information_nats"].values()]])
    if "01_generate_dataset" in m:
        g = m["01_generate_dataset"]
        out(f"Generator: rule on noisy readings matches the true label on "
            f"{100 * g['rule_on_noisy_readings_accuracy']:.2f} % of complete rows; "
            f"{g['bloom_crashes']} bloom crashes; mean DO by crop third "
            f"{' / '.join(f(v, 2) for v in g['mean_do_by_crop_third'])} mg/L.\n")

    rows = []
    for block, name in (("03_random_forest", "Random Forest"), ("04_lightgbm", "LightGBM")):
        if block in m:
            rows.append([name, f(m[block]["test"]["accuracy"], 4), f(m[block]["test"]["macro_f1"], 4),
                         *[f(m[block]["per_class_f1"][c], 3) for c in ("Good", "Moderate", "Critical")]])
    if "03_random_forest" in m:
        r = m["03_random_forest"]
        rows.append(["Labelling rule on same readings", f(r["rule_reference"]["accuracy"], 4),
                     f(r["rule_reference"]["macro_f1"], 4), "", "", ""])
        rows.append(["Always the majority class", f(r["majority_baseline"]["accuracy"], 4),
                     f(r["majority_baseline"]["macro_f1"], 4), "", "", ""])
        out(f"### Status classification (03, 04), test rows: {r['n_test']}\n")
        table(["Method", "Accuracy", "Macro F1", "F1 Good", "F1 Moderate", "F1 Critical"], rows)
        out("Random Forest permutation importance (drop in macro F1): "
            + ", ".join(f"{k} {f(v)}" for k, v in r["permutation_importance"].items()) + ".\n")

    for block, label in (("05_lstm_do_forecast", "change target (main model)"),
                         ("05_lstm_do_forecast_level", "level target (comparison)")):
        if block in m:
            b = m[block]
            out(f"### DO forecast +{b['horizon_h']} h (05), {label}, test hours: {b['n_test']}\n")
            table(["Method", "RMSE (mg/L)", "MAE (mg/L)", "R²"],
                  [[n, f(b[k]["rmse"]), f(b[k]["mae"]), f(b[k]["r2"])] for n, k in
                   (("LSTM", "lstm"), ("Persistence", "persistence"), ("Same hour yesterday", "same_hour_yesterday"))])
            out(f"Skill vs persistence (1 - MSE ratio): {b['skill_vs_persistence']:+.3f}; "
                f"stopped after {b['epochs_run']} epochs.\n")
    if (rep / "05_horizon_sweep.csv").exists():
        s = pd.read_csv(rep / "05_horizon_sweep.csv")
        out("### DO forecast horizon sweep (05), test RMSE in mg/L\n")
        table(["Horizon", "LSTM (mean ± sd)", "LSTM R²", "Persistence", "Same hour yesterday", "LSTM ahead of both"],
              [[f"+{int(r.horizon)} h", pm(r.lstm_rmse_mean, r.lstm_rmse_sd), f(r.lstm_r2_mean),
                f(r.persistence_rmse), f(r.yesterday_rmse), "yes" if r.lstm_beats_best_baseline else "no"]
               for r in s.itertuples()])

    rows = []
    for tag in ("h6_random", "h1_random", "h6_time"):
        b = m.get(f"06_do_state_{tag}")
        if b:
            rows.append([f"+{b['horizon_h']} h, {b['split']}",
                         *[pm(b[k]["macro_f1_mean"], b[k]["macro_f1_sd"]) for k in ("lstm", "persistence", "yesterday")],
                         f"{b['lstm_wins_vs_persistence']} / {b['lstm_wins_vs_yesterday']} of {b['runs']}",
                         "yes" if b["lstm_beats_both"] else "no"])
    if rows:
        out("### Oxygen-state classifier (06), macro F1 over the states present\n")
        table(["Horizon, split", "LSTM", "Persistence", "Same hour yesterday",
               "LSTM wins vs persistence / yesterday", "Beats both"], rows)

    if "07_lr_vs_rf_scaling" in m:
        b = m["07_lr_vs_rf_scaling"]
        out("### Linear Regression vs Random Forest (07), test RMSE in mg/L, mean ± 95 % CI\n")
        table(["Training rows", "Linear Regression", "Random Forest", "LR R²", "RF R²"],
              [[c["n_train"], pm(c["lr_rmse_mean"], c["lr_rmse_ci95"]), pm(c["rf_rmse_mean"], c["rf_rmse_ci95"]),
                f(c["lr_r2_mean"]), f(c["rf_r2_mean"])] for c in b["curve"]])
        out(f"Forest stays ahead from n = {b['forest_ahead_from_n']}; with non-overlapping CIs from "
            f"n = {b.get('forest_clearly_ahead_from_n')}. Always predicting the mean: RMSE "
            f"{f(b['mean_baseline_rmse'])}.\n")

    if "08_cross_site" in m:
        out("### Cross-site test at Cat Point (08), test RMSE in mg/L, mean ± sd\n")
        table(["Horizon", "Trained at Dry Bar", "Trained at Cat Point", "Persistence",
               "Transfer penalty", "R² transfer / local"],
              [[f"+{r['horizon']} h", pm(r["transfer_rmse_mean"], r["transfer_rmse_sd"]),
                pm(r["local_rmse_mean"], r["local_rmse_sd"]), f(r["persistence_rmse"]),
                f"{r['transfer_penalty_pct']:+.1f} %", f"{f(r['transfer_r2_mean'])} / {f(r['local_r2_mean'])}"]
               for r in m["08_cross_site"]["results"]])

    if "recommend" in m:
        b = m["recommend"]
        out(f"### Recommendation engine, model: {b['model']}\n")
        table(["Scenario", "pH", "°C", "DO", "ppt", "Rules", "Model (confidence)", "Agree", "Final status", "Immediate"],
              [[s["scenario"], s["ph"], s["temperature_c"], s["do_mg_l"], s["salinity_ppt"], s["rule_status"],
                f"{s['model_status']} ({100 * s['model_confidence']:.0f} %)", "yes" if s["agreement"] else "no",
                s["status"], s["n_immediate"]] for s in b["scenarios"]])
        if b.get("surrogate_fidelity") is not None:
            out(f"Depth-3 surrogate tree agrees with the model on {100 * b['surrogate_fidelity']:.1f} % "
                "of test readings.\n")


def summarize_ponds() -> None:
    """Real catfish ponds (tools/prepare_ponds.py and 09_real_ponds.py)."""
    qc_path, m_path = ROOT / "data" / "ponds" / "qc_summary.csv", ROOT / "reports" / "ponds" / "metrics.json"
    if not qc_path.exists() and not m_path.exists():
        out("## ponds: nothing to report (run tools/prepare_ponds.py and 09_real_ponds.py)\n")
        return
    out("## Dataset: ponds (real catfish ponds, freshwater; Udanor et al. 2022)\n")
    if qc_path.exists():
        qc = pd.read_csv(qc_path).fillna("")
        out("### Raw files and quality control (prepare_ponds)\n")
        table(["Pond", "Raw rows", "DO kept", "Hours with DO", "DO p5 / median / p95 (mg/L)",
               "Hours < 4 mg/L", "Hours < 2 mg/L", "Used", "Why excluded"],
              [[r.pond, f"{r.raw_rows:,}", f"{r.do_kept_pct:.1f} %", f"{r.hours_with_do:,}",
                f"{r.do_p5:.2f} / {r.do_median:.2f} / {r.do_p95:.2f}", r.hours_below_4, r.hours_below_2,
                "yes" if r.used else "no", r.exclusion_reason] for r in qc.itertuples()])
    if not m_path.exists():
        return
    b = json.loads(m_path.read_text(encoding="utf-8"))["09_real_ponds"]
    out(f"### DO forecast per pond (09), test RMSE in mg/L, {b['window_h']}-h window, {b['seeds']} seeds\n")
    table(["Pond", "Horizon", "Test hours", "Local LSTM", "LSTM from other 5 ponds", "Persistence",
           "Same hour yesterday"],
          [[r["pond"], f"+{r['horizon']} h", r["n_test"], pm(r["local_rmse"], r["local_rmse_sd"]),
            pm(r["transfer_rmse"], r["transfer_rmse_sd"]), f(r["persistence_rmse"]), f(r["yesterday_rmse"])]
           for r in b["forecast"]])
    out(f"### Oxygen state +6 h per pond (09), macro F1 over the states present\n")
    table(["Pond", "Test hours", "States in test", "Local LSTM", "LSTM from other 5 ponds", "Persistence",
           "Same hour yesterday"],
          [[r["pond"], r["n_test"], r["states_in_test"], pm(r["local_f1"], r["local_f1_sd"]),
            pm(r["transfer_f1"], r["transfer_f1_sd"]), f(r["persistence_f1"]), f(r["yesterday_f1"])]
           for r in b["state"]])
    out("### Mean over the six ponds (09)\n")
    rows = []
    for name, v in b["summary"].items():
        key = "rmse" if name.startswith("forecast") else "f1"
        label = name.replace("forecast_h", "DO forecast RMSE, +").replace("state_h", "Oxygen state macro F1, +") + " h"
        rows.append([label, *[f(v[f"mean_{k}_{key}"]) for k in ("local", "transfer", "persistence", "yesterday")],
                     f"{v['local_beats_better_baseline']} of {v['ponds']}",
                     f"{v['transfer_beats_better_baseline']} of {v['ponds']}",
                     f"{v['transfer_beats_local']} of {v['ponds']}"])
    table(["Task", "Local LSTM", "LSTM from other 5 ponds", "Persistence", "Same hour yesterday",
           "Local beats better baseline", "Transferred beats better baseline", "Transferred beats local"], rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Markdown summary of the last run")
    ap.add_argument("--out", type=Path, help="also write the Markdown to this file")
    args = ap.parse_args()
    out("# Aquabot results summary\n")
    out("Generated by `tools/summarize_results.py` from `reports/<dataset>/metrics.json` and the CSV reports.\n")
    for dataset in ("synthetic", "real"):
        summarize(dataset)
    summarize_ponds()
    text = "\n".join(LINES)
    sys.stdout.reconfigure(encoding="utf-8")              # Windows consoles default to cp1252
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
