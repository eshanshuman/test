"""09 - DO forecasting and oxygen-state warning on real stocked ponds.

Purpose
    The only test in this project on real STOCKED ponds: six African-catfish
    ponds from Udanor et al. 2022 (prepared by tools/prepare_ponds.py). Every
    result is "real catfish ponds, freshwater": not vannamei, no salinity, a
    noisy DO sensor. Three experiments, 3 seeds each, 32 LSTM units:
      A. Per pond: LSTM DO forecast at +1, +3, +6 h vs persistence and
         same-hour-yesterday on the same test hours.
      B. Leave-one-pond-out: one LSTM trained on the training sets of the
         other five ponds (scaler from their training hours, early stopping on
         their validation sets), tested on the held-out pond's test hours,
         the same hours as A.
      C. Oxygen state 6 h ahead (Anoxia / Hypoxia / Normoxia), local and
         leave-one-pond-out, square-root class weights, macro F1 over the
         states present with the same label list for every method.
    Gap handling, valid windows, the measured-target rule, the change-in-DO
    target and the LSTM all come from ts_lib, unchanged.

    Two deviations from 05/06/08, made because of this dataset's long gaps:
      * Input window 24 h instead of 168 h: too few gap-free weeks exist.
      * The chronological 70/15/15 split is taken over each pond's ORDERED
        USABLE SAMPLES, not its calendar span. Valid hours cluster early, so a
        calendar split leaves some ponds with no test hours. The periods are
        still disjoint in time: a sample belongs to a set only if both its
        last input and its target fall inside that set's period, and the
        scaler is fitted on hours of the training period only.
    Nothing here was tuned to beat the baselines.

Paper link
    Data: Udanor et al. 2022 (Data in Brief 43:108400). A leakage-aware
    re-analysis of the same dataset (github.com/MohamedElminur1/
    DL_Models_Fishweight) reports that simple baselines matched deep models;
    this script asks the same question for DO.

Outputs
    figures/ponds/fig18_pond_forecast.png; reports/ponds/09_forecast.csv,
    09_state.csv; block "09_real_ponds" in reports/ponds/metrics.json.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import config as cfg
import ts_lib as ts

cfg.apply_style()
FIG_DIR, REPORT_DIR = cfg.ROOT / "figures" / "ponds", cfg.ROOT / "reports" / "ponds"
WINDOW, UNITS, SEEDS, EPOCHS, PATIENCE = 24, 32, (0, 1, 2), 40, 6
HORIZONS, STATE_HORIZON = (1, 3, 6), 6
METHODS = {"local": ("Local LSTM", cfg.SERIES_COLOURS[0], "o"),
           "transfer": ("LSTM from the other 5 ponds", cfg.SERIES_COLOURS[1], "s"),
           "persistence": ("Persistence", cfg.SERIES_COLOURS[2], "^"),
           "yesterday": ("Same hour yesterday", "#777777", "D")}


def split(series: dict, horizon: int) -> dict:
    """Chronological 70/15/15 over the ordered usable samples (see docstring)."""
    t = ts.valid_ends(series, WINDOW, horizon)
    t1, t2 = t[int(0.70 * len(t))], t[int(0.85 * len(t))]     # first hour of the val / test period
    return {"train": t[t + horizon < t1], "val": t[(t >= t1) & (t + horizon < t2)], "test": t[t >= t2],
            "train_rows": series["values"][:t1]}                # hours of the training period


def scaler(rows: np.ndarray):
    mean, sd = np.nanmean(rows, axis=0), np.nanstd(rows, axis=0)
    return mean, np.where(sd > 0, sd, 1.0)


def build_xy(parts: list, mean, sd, horizon: int, which: str, task: str, delta_sd: float = 1.0):
    """Stack windows and targets of one set ('train'/'val'/'test') over one or more ponds."""
    X, y = [], []
    for series, sp in parts:
        t, do = sp[which], series["values"][:, ts.DO]
        X.append(ts.windows((series["values"] - mean) / sd, t, WINDOW))
        y.append(cfg.oxygen_state(do[t + horizon]).astype(int) if task == "state"
                 else ((do[t + horizon] - do[t]) / delta_sd).astype(np.float32))
    return np.concatenate(X), np.concatenate(y)


def fit_predict(train_parts: list, test_part: tuple, horizon: int, task: str, seed: int) -> np.ndarray:
    """Train on train_parts (one pond = local, five ponds = transfer); predict the test pond's test hours."""
    mean, sd = scaler(np.concatenate([sp["train_rows"] for _, sp in train_parts]))
    deltas = np.concatenate([s["values"][sp["train"] + horizon, ts.DO] - s["values"][sp["train"], ts.DO]
                             for s, sp in train_parts])
    delta_sd = float(np.std(deltas))
    X_tr, y_tr = build_xy(train_parts, mean, sd, horizon, "train", task, delta_sd)
    X_va, y_va = build_xy(train_parts, mean, sd, horizon, "val", task, delta_sd)
    X_te, _ = build_xy([test_part], mean, sd, horizon, "test", task, delta_sd)
    ts.seed_everything(seed)
    if task == "state":
        model = ts.build_lstm(WINDOW, UNITS, n_out=3, activation="softmax")
        ts.train(model, X_tr, y_tr, X_va, y_va, EPOCHS, loss="sparse_categorical_crossentropy",
                 patience=PATIENCE, class_weight=ts.sqrt_class_weights(y_tr))
        return model.predict(X_te, verbose=0).argmax(axis=1)
    model = ts.build_lstm(WINDOW, UNITS)
    ts.train(model, X_tr, y_tr, X_va, y_va, EPOCHS, patience=PATIENCE)
    series, sp = test_part
    return series["values"][sp["test"], ts.DO] + model.predict(X_te, verbose=0).ravel() * delta_sd


def run_task(ponds: dict, horizon: int, task: str) -> list[dict]:
    """Experiments A+B (task='forecast') or C (task='state') for one horizon."""
    splits = {p: split(s, horizon) for p, s in ponds.items()}
    rows = []
    for p, series in ponds.items():
        sp, do = splits[p], series["values"][:, ts.DO]
        test_t = sp["test"]
        truth = do[test_t + horizon]
        others = [(ponds[q], splits[q]) for q in ponds if q != p]
        preds = {"local": [fit_predict([(series, sp)], (series, sp), horizon, task, s) for s in SEEDS],
                 "transfer": [fit_predict(others, (series, sp), horizon, task, s) for s in SEEDS],
                 "persistence": [do[test_t]], "yesterday": [do[test_t + horizon - 24]]}
        row = {"pond": p, "horizon": horizon, "n_train": len(sp["train"]), "n_val": len(sp["val"]),
               "n_test": len(test_t)}
        if task == "state":
            y = cfg.oxygen_state(truth).astype(int)
            present = sorted(np.unique(y).tolist())               # same label list for every method
            row["states_in_test"] = "/".join(cfg.DO_STATES[i] for i in present)
            score = lambda pr: f1_score(y, pr if pr.dtype.kind == "i" else cfg.oxygen_state(pr).astype(int),
                                        labels=present, average="macro", zero_division=0)
            key = "f1"
        else:
            score = lambda pr: ts.regression_scores(truth, pr)["rmse"]
            key = "rmse"
            for m in preds:
                row[f"{m}_r2"] = float(np.mean([ts.regression_scores(truth, pr)["r2"] for pr in preds[m]]))
        for m, plist in preds.items():
            vals = np.array([score(pr) for pr in plist])
            row[f"{m}_{key}"] = float(vals.mean())
            if len(plist) > 1:
                row[f"{m}_{key}_sd"] = float(vals.std())
        rows.append(row)
        base = min(row["persistence_rmse"], row["yesterday_rmse"]) if key == "rmse" else \
            max(row["persistence_f1"], row["yesterday_f1"])
        print(f"  pond {p}  +{horizon} h  test n {len(test_t):>4}   local {row[f'local_{key}']:.3f}   "
              f"transfer {row[f'transfer_{key}']:.3f}   persistence {row[f'persistence_{key}']:.3f}   "
              f"yesterday {row[f'yesterday_{key}']:.3f}   better baseline {base:.3f}", flush=True)
    return rows


def verdicts(table: pd.DataFrame, key: str) -> dict:
    """Counts over ponds: who beats the better baseline, and transfer vs local."""
    lower = key == "rmse"
    base = table[[f"persistence_{key}", f"yesterday_{key}"]].min(axis=1) if lower else \
        table[[f"persistence_{key}", f"yesterday_{key}"]].max(axis=1)
    better = (lambda a, b: a < b) if lower else (lambda a, b: a > b)
    return {"ponds": len(table),
            "local_beats_better_baseline": int(better(table[f"local_{key}"], base).sum()),
            "transfer_beats_better_baseline": int(better(table[f"transfer_{key}"], base).sum()),
            "transfer_beats_local": int(better(table[f"transfer_{key}"], table[f"local_{key}"]).sum()),
            **{f"mean_{m}_{key}": round(float(table[f"{m}_{key}"].mean()), 4) for m in METHODS}}


def fig_forecast(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    offsets = np.linspace(-0.27, 0.27, len(METHODS))
    for off, (m, (name, colour, marker)) in zip(offsets, METHODS.items()):
        for i, h in enumerate(HORIZONS):
            vals = table.loc[table["horizon"] == h, f"{m}_rmse"]
            ax.scatter(np.full(len(vals), i + off), vals, s=22, color=colour, marker=marker, alpha=0.45, linewidths=0)
            ax.scatter(i + off, vals.mean(), s=120, color=colour, marker=marker, edgecolor="#222222",
                       linewidths=1.2, zorder=3, label=name if i == 0 else None)
            ax.annotate(f"{vals.mean():.2f}", (i + off, vals.mean()), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=7)
    ax.set_xticks(range(len(HORIZONS)), [f"+{h} h" for h in HORIZONS])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Test RMSE (mg/L), lower is better")
    ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.set_title(f"DO forecast error on six ponds: small dots = one pond, large = mean over ponds\n"
                 f"{cfg.PONDS_NOTE} (Udanor et al. 2022); LSTMs averaged over {len(SEEDS)} seeds")
    cfg.save_fig(fig, "fig18_pond_forecast", FIG_DIR)


def main() -> None:
    cfg.section(f"09 Real ponds: {cfg.PONDS_NOTE} (Udanor et al. 2022)")
    missing = [p for p in cfg.POND_IDS if not (cfg.PONDS_DIR / f"pond_{p:02d}.csv").exists()]
    if missing:
        raise SystemExit(f"data/ponds/ files missing for ponds {missing}: run tools/prepare_ponds.py first.")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ponds = {p: ts.prepare_series(cfg.load_dataset(cfg.PONDS_DIR / f"pond_{p:02d}.csv")) for p in cfg.POND_IDS}
    print(f"  window {WINDOW} h, {UNITS} units, {len(SEEDS)} seeds, split over ordered usable samples")
    print(f"  {'pond':>4} {'grid h':>7} {'DO h':>6} {'usable +1 h samples':>20}  train / val / test    test period starts")
    for p, s in ponds.items():
        sp = split(s, 1)
        print(f"  {p:>4} {len(s['time']):>7,} {int(s['do_measured'].sum()):>6,} "
              f"{len(ts.valid_ends(s, WINDOW, 1)):>20,}  {len(sp['train']):>5} / {len(sp['val']):>3} / {len(sp['test']):>4}"
              f"    {str(s['time'][sp['test'][0]])[:16]}")

    cfg.section("A + B. DO forecast, test RMSE in mg/L (LSTMs: mean over seeds)")
    forecast = pd.DataFrame([r for h in HORIZONS for r in run_task(ponds, h, "forecast")])
    forecast.round(4).to_csv(REPORT_DIR / "09_forecast.csv", index=False)

    cfg.section(f"C. Oxygen state +{STATE_HORIZON} h, macro F1 over the states present (higher is better)")
    state = pd.DataFrame(run_task(ponds, STATE_HORIZON, "state"))
    state.round(4).to_csv(REPORT_DIR / "09_state.csv", index=False)

    cfg.section("Summary over the six ponds")
    summary = {}
    print(f"  {'task':<14}{'local':>8}{'transfer':>10}{'persist.':>10}{'yesterday':>11}   "
          f"local > base   transfer > base   transfer > local")
    for h in HORIZONS:
        v = verdicts(forecast[forecast["horizon"] == h], "rmse")
        summary[f"forecast_h{h}"] = v
        print(f"  {'RMSE +' + str(h) + ' h':<14}{v['mean_local_rmse']:>8.3f}{v['mean_transfer_rmse']:>10.3f}"
              f"{v['mean_persistence_rmse']:>10.3f}{v['mean_yesterday_rmse']:>11.3f}   "
              f"{v['local_beats_better_baseline']:>5} of {v['ponds']}   {v['transfer_beats_better_baseline']:>8} of {v['ponds']}"
              f"   {v['transfer_beats_local']:>9} of {v['ponds']}")
    v = verdicts(state, "f1")
    summary[f"state_h{STATE_HORIZON}"] = v
    print(f"  {'F1 state +' + str(STATE_HORIZON) + ' h':<14}{v['mean_local_f1']:>8.3f}{v['mean_transfer_f1']:>10.3f}"
          f"{v['mean_persistence_f1']:>10.3f}{v['mean_yesterday_f1']:>11.3f}   "
          f"{v['local_beats_better_baseline']:>5} of {v['ponds']}   {v['transfer_beats_better_baseline']:>8} of {v['ponds']}"
          f"   {v['transfer_beats_local']:>9} of {v['ponds']}")
    print("  ('beats' = better than the better of the two baselines in that pond, on the seed-mean score)")

    cfg.section("Figure")
    fig_forecast(forecast)
    cfg.record_metric("09_real_ponds", {
        "note": cfg.PONDS_NOTE, "ponds": list(cfg.POND_IDS), "window_h": WINDOW, "units": UNITS,
        "seeds": len(SEEDS), "summary": summary,
        "forecast": forecast.round(4).to_dict(orient="records"),
        "state": state.round(4).to_dict(orient="records")}, REPORT_DIR / "metrics.json")


if __name__ == "__main__":
    main()
