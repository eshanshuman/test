"""06 - LSTM that names the oxygen state H hours ahead (Anoxia / Hypoxia / Normoxia).

Purpose
    A farmer acts on "will the pond be short of oxygen at dawn?", not on a
    number. This script classifies the oxygen state H hours ahead (default 6;
    also run with --horizon 1) from the last 24 h of readings: Anoxia < 2,
    Hypoxia 2-4, Normoxia >= 4 mg/L.

    Two ways to split, because they answer different questions:
      * random (default): whole calendar days are dealt at random to
        train/validation/test. Optimistic: a test day's neighbours are in the
        training set and share its weather.
      * --split time: chronological 70/15/15. Stricter: the test period is
        later than anything the model saw.
    Low-oxygen hours are rare, so the loss uses square-root inverse-frequency
    class weights (a gentler correction than full inverse frequency; no
    resampling). One split proves little, so the experiment is repeated 10
    times with a new split and seed and reported as mean ± sd, together with
    how many runs the LSTM beat each baseline:
      * persistence: the state stays what it is now;
      * same hour yesterday: the state 24 h before the target hour.
    Macro F1 is computed over the states that actually occur in that run's
    test set, with the identical label list for every method.

Paper link
    The project brief quotes 92.4 % for a quantum-LSTM oxygen-state
    classifier. That paper was not supplied, so its data, split and metric are
    unknown and the figure CANNOT be reproduced or compared here; it is a
    reference point only. Nothing below is a quantum model.

Outputs (tagged by horizon and split, e.g. h6_random)
    figures/<dataset>/fig14_do_state_confusion_<tag>.png;
    reports/<dataset>/06_do_state_runs_<tag>.csv; block "06_do_state_<tag>"
    in metrics.json.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import config as cfg
import ts_lib as ts

cfg.apply_style()
NOTE = f"{cfg.DATASET} dataset ({cfg.DATASET_NOTE})"
WINDOW, UNITS, N_RUNS = 24, 32, 10
ALL_STATES = [0, 1, 2]


def split_samples(series, t, horizon, mode, rng):
    """Return train/val/test sample indices and the boolean mask of training rows."""
    n = len(series["time"])
    if mode == "time":
        train, val, test, train_end = ts.chrono_split(t, horizon, n)
        return train, val, test, np.arange(n) < train_end
    day = series["time"].normalize()                     # calendar day of every hour
    days = rng.permutation(day.unique())
    n_tr, n_va = int(0.70 * len(days)), int(0.15 * len(days))
    which = {"train": days[:n_tr], "val": days[n_tr:n_tr + n_va], "test": days[n_tr + n_va:]}
    target_day = day[t + horizon]                        # a sample belongs to its TARGET's day
    parts = [t[target_day.isin(which[k])] for k in ("train", "val", "test")]
    return parts[0], parts[1], parts[2], np.asarray(day.isin(which["train"]))


def one_run(series, horizon, mode, seed, epochs) -> dict:
    values, do = series["values"], series["values"][:, ts.DO]
    state = cfg.oxygen_state(do)
    rng = np.random.default_rng(seed)
    t = ts.valid_ends(series, WINDOW, horizon)
    train_t, val_t, test_t, train_rows = split_samples(series, t, horizon, mode, rng)

    rows = values[train_rows]                            # scaler: training rows only
    mean, sd = np.nanmean(rows, axis=0), np.nanstd(rows, axis=0)
    scaled = (values - mean) / np.where(sd > 0, sd, 1.0)
    X = {k: ts.windows(scaled, idx, WINDOW) for k, idx in (("tr", train_t), ("va", val_t), ("te", test_t))}
    y = {k: state[idx + horizon].astype(int) for k, idx in (("tr", train_t), ("va", val_t), ("te", test_t))}

    counts = np.bincount(y["tr"], minlength=3)
    present_tr = counts > 0
    weights = np.zeros(3)
    weights[present_tr] = np.sqrt(counts.sum() / (present_tr.sum() * counts[present_tr]))
    weights[present_tr] /= weights[present_tr].mean()    # square-root inverse frequency, mean 1
    class_weight = {i: float(weights[i]) for i in ALL_STATES if present_tr[i]}

    ts.seed_everything(seed)
    model = ts.build_lstm(WINDOW, UNITS, n_out=3, activation="softmax")
    epochs_run, _ = ts.train(model, X["tr"], y["tr"], X["va"], y["va"], epochs,
                             loss="sparse_categorical_crossentropy", patience=6,
                             class_weight=class_weight)
    preds = {"lstm": model.predict(X["te"], verbose=0).argmax(axis=1),
             "persistence": state[test_t].astype(int),
             "yesterday": state[test_t + horizon - 24].astype(int)}
    present = sorted(np.unique(y["te"]).tolist())        # same label list for every method
    out = {"seed": seed, "n_train": len(train_t), "n_test": len(test_t), "epochs_run": epochs_run,
           "states_in_test": "/".join(cfg.DO_STATES[i] for i in present),
           "cm": confusion_matrix(y["te"], preds["lstm"], labels=ALL_STATES)}
    for name, p in preds.items():
        out[f"{name}_f1"] = f1_score(y["te"], p, labels=present, average="macro", zero_division=0)
        out[f"{name}_acc"] = accuracy_score(y["te"], p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Oxygen-state classifier H hours ahead")
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--split", choices=["random", "time"], default="random")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--runs", type=int, default=N_RUNS)
    args = ap.parse_args()
    if not 1 <= args.horizon <= 24:
        raise SystemExit("--horizon must be between 1 and 24")
    tag = f"h{args.horizon}_{args.split}"
    series = ts.prepare_series(cfg.load_dataset())

    cfg.section(f"06 Oxygen state +{args.horizon} h, split = {args.split}, on the {NOTE}")
    state_now = cfg.oxygen_state(series["values"][:, ts.DO])
    seen = state_now[~np.isnan(state_now)].astype(int)
    print("  hours in each state: " + ", ".join(
        f"{cfg.DO_STATES[i]} {int((seen == i).sum()):,} ({100 * (seen == i).mean():.1f} %)" for i in ALL_STATES))
    print(f"  window {WINDOW} h, {UNITS} LSTM units, {args.runs} runs "
          + ("(new random day split and seed each run)" if args.split == "random"
             else "(same chronological split, new seed each run)"))

    cfg.section("Runs")
    runs = []
    print(f"  {'run':>3} {'test n':>7} {'LSTM F1':>8} {'persist':>8} {'yesterday':>10}  states in test")
    for r in range(args.runs):
        res = one_run(series, args.horizon, args.split, cfg.SEED + r, args.epochs)
        runs.append(res)
        print(f"  {r + 1:>3} {res['n_test']:>7,} {res['lstm_f1']:>8.3f} {res['persistence_f1']:>8.3f} "
              f"{res['yesterday_f1']:>10.3f}  {res['states_in_test']}")
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "cm"} for r in runs])
    table.round(4).to_csv(cfg.REPORT_DIR / f"06_do_state_runs_{tag}.csv", index=False)

    cfg.section(f"Summary over {args.runs} runs (macro F1 over the states present in each test set)")
    summary = {}
    print(f"  {'method':<22}{'macro F1':>16}{'accuracy':>18}")
    for key, name in (("lstm", "LSTM"), ("persistence", "Persistence"), ("yesterday", "Same hour yesterday")):
        f1, acc = table[f"{key}_f1"], table[f"{key}_acc"]
        summary[key] = {"macro_f1_mean": round(float(f1.mean()), 4), "macro_f1_sd": round(float(f1.std(ddof=0)), 4),
                        "accuracy_mean": round(float(acc.mean()), 4), "accuracy_sd": round(float(acc.std(ddof=0)), 4)}
        print(f"  {name:<22}{f1.mean():>9.3f} ± {f1.std(ddof=0):.3f}{acc.mean():>11.3f} ± {acc.std(ddof=0):.3f}")
    wins_p = int((table["lstm_f1"] > table["persistence_f1"]).sum())
    wins_y = int((table["lstm_f1"] > table["yesterday_f1"]).sum())
    print(f"\n  LSTM beat persistence in {wins_p} of {args.runs} runs, "
          f"and same-hour-yesterday in {wins_y} of {args.runs} runs (macro F1).")
    best_base = max(summary["persistence"]["macro_f1_mean"], summary["yesterday"]["macro_f1_mean"])
    ahead = summary["lstm"]["macro_f1_mean"] > best_base and min(wins_p, wins_y) > args.runs / 2
    print("  VERDICT: the LSTM " + ("beats both baselines" if ahead else "does NOT clearly beat both baselines")
          + f" at +{args.horizon} h with the {args.split} split.")

    cfg.section("Pooled confusion matrix (all runs)")
    pooled = np.sum([r["cm"] for r in runs], axis=0)
    recall = pooled.diagonal() / np.maximum(pooled.sum(axis=1), 1)
    for i in ALL_STATES:
        print(f"  {cfg.DO_STATES[i]:<9} test hours {int(pooled[i].sum()):>6,}   recall {recall[i]:.3f}")
    cfg.plot_confusion(pooled, cfg.DO_STATES,
                       f"Oxygen state +{args.horizon} h, {args.split} split, {args.runs} runs pooled\n{NOTE}",
                       f"fig14_do_state_confusion_{tag}")
    cfg.record_metric(f"06_do_state_{tag}", {
        "horizon_h": args.horizon, "split": args.split, "runs": args.runs, "window_h": WINDOW,
        **summary, "lstm_wins_vs_persistence": wins_p, "lstm_wins_vs_yesterday": wins_y,
        "lstm_beats_both": bool(ahead), "pooled_confusion_matrix": pooled,
        "pooled_recall": {cfg.DO_STATES[i]: round(float(recall[i]), 4) for i in ALL_STATES}})


if __name__ == "__main__":
    main()
