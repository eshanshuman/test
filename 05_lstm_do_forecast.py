"""05 - LSTM forecast of dissolved oxygen, scored against two baselines.

Purpose
    Forecast DO some hours ahead from the last 168 h (one week) of readings.
    Channels: DO, temperature, pH, salinity and the hour of day as sin/cos.
    By default the network predicts the CHANGE in DO over the horizon and the
    level is rebuilt as "last reading + predicted change" (--target level
    predicts the level directly, for comparison). Predicting the change makes
    "no change" the starting point, so the network has to earn any improvement
    over simply repeating the last reading.

    A forecast only "works" where it beats the baselines, scored on exactly
    the same test hours:
      * persistence: DO stays at the last reading;
      * same hour yesterday: DO at the target hour minus 24 h. (At +24 h this
        is the same number as persistence; they differ at shorter horizons.)

    Leakage guards: chronological 70/15/15 split (no shuffling of time),
    scaler fitted on the training period only, windows that touch a sensor gap
    longer than MAX_GAP_H skipped, targets must be real measurements.

Paper link
    LSTM DO forecasting is the headline model of the reviewed papers (Chen et
    al. 2022; Huang & Khabusi 2025). The brief's "Vietnam LSTM" paper was not
    supplied; its numbers are reference points only and are not reproduced.

Outputs
    figures/<dataset>/fig11_lstm_forecast_vs_measured.png, fig12_lstm_scatter.png,
    fig13_lstm_horizon_sweep.png; models/<dataset>/lstm_do_forecast.keras and
    lstm_do_forecast_meta.json; reports/<dataset>/05_horizon_sweep.csv; block
    "05_lstm_do_forecast" in metrics.json ("..._level" with --target level,
    which records numbers only).
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
import ts_lib as ts

cfg.apply_style()
NOTE = f"{cfg.DATASET} dataset ({cfg.DATASET_NOTE})"
WINDOW, MAIN_UNITS, MAIN_HORIZON = 168, 48, 24
SWEEP_HORIZONS, SWEEP_UNITS, SWEEP_SEEDS = (1, 3, 6, 12, 24), 32, (0, 1, 2)
C_LSTM, C_PERSIST, C_YESTERDAY = cfg.SERIES_COLOURS


def run_once(series, horizon, units, seed, epochs, target="delta", keep_model=False) -> dict:
    """Train one model for one horizon and score it and both baselines on the test hours."""
    values, n = series["values"], len(series["time"])
    train_t, val_t, test_t, train_end = ts.chrono_split(ts.valid_ends(series, WINDOW, horizon), horizon, n)
    mean, sd = ts.fit_scaler(values, train_end)                  # training period only
    scaled = (values - mean) / sd
    do = values[:, ts.DO]
    delta_sd = float(np.std(do[train_t + horizon] - do[train_t]))   # training samples only

    def xy(t):
        y = ((do[t + horizon] - do[t]) / delta_sd if target == "delta"
             else scaled[t + horizon, ts.DO])
        return ts.windows(scaled, t, WINDOW), y.astype(np.float32)

    (X_tr, y_tr), (X_va, y_va), (X_te, _) = xy(train_t), xy(val_t), xy(test_t)
    ts.seed_everything(seed)
    model = ts.build_lstm(WINDOW, units)
    n_epochs, _ = ts.train(model, X_tr, y_tr, X_va, y_va, epochs)
    out = model.predict(X_te, verbose=0).ravel()
    pred = (do[test_t] + out * delta_sd if target == "delta"
            else out * sd[ts.DO] + mean[ts.DO])
    truth = do[test_t + horizon]
    result = {"horizon": horizon, "seed": seed, "epochs_run": n_epochs,
              "n_train": len(train_t), "n_val": len(val_t), "n_test": len(test_t),
              "lstm": ts.regression_scores(truth, pred),
              "persistence": ts.regression_scores(truth, do[test_t]),
              "yesterday": ts.regression_scores(truth, do[test_t + horizon - 24]),
              "test_t": test_t, "pred": pred, "truth": truth,
              "split_ok": bool(train_t.max() + horizon < val_t.min() and val_t.max() + horizon < test_t.min())}
    if keep_model:
        result.update(model=model, mean=mean, sd=sd, delta_sd=delta_sd)
    return result


def fig_forecast(series, res) -> None:
    target_time = series["time"][res["test_t"] + res["horizon"]]
    frame = pd.DataFrame({"measured": res["truth"], "lstm": res["pred"]}, index=target_time)
    last = frame[frame.index > frame.index.max() - pd.Timedelta(days=14)]
    last = last.reindex(pd.date_range(last.index.min(), last.index.max(), freq="h"))   # NaN = gap -> line break
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(last.index, last["measured"], color="#333333", linewidth=1.3, label="Measured")
    ax.plot(last.index, last["lstm"], color=C_LSTM, linewidth=1.3,
            label=f"LSTM forecast made {res['horizon']} h earlier")
    ax.set_ylabel("DO (mg/L)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.set_title(f"+{res['horizon']} h DO forecast vs measurement, last 14 test days "
                 f"(breaks = sensor gaps)\n{NOTE}")
    cfg.save_fig(fig, "fig11_lstm_forecast_vs_measured")


def fig_scatter(res) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    lim = [min(res["truth"].min(), res["pred"].min()) - 0.3, max(res["truth"].max(), res["pred"].max()) + 0.3]
    ax.scatter(res["truth"], res["pred"], s=8, alpha=0.4, color=C_LSTM, linewidths=0)
    ax.plot(lim, lim, color="#333333", linestyle="--", linewidth=1, label="perfect forecast")
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("Measured DO (mg/L)")
    ax.set_ylabel(f"Forecast DO, +{res['horizon']} h (mg/L)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"LSTM test forecasts (R² = {res['lstm']['r2']:.3f}, "
                 f"persistence R² = {res['persistence']['r2']:.3f})\n{NOTE}")
    cfg.save_fig(fig, "fig12_lstm_scatter")


def fig_sweep(sweep: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4))
    x = np.arange(len(sweep))
    ax.errorbar(x, sweep["lstm_rmse_mean"], yerr=sweep["lstm_rmse_sd"], color=C_LSTM, marker="o",
                markersize=7, linewidth=2, capsize=4, label=f"LSTM (mean ± sd, {len(SWEEP_SEEDS)} seeds)")
    ax.plot(x, sweep["persistence_rmse"], color=C_PERSIST, marker="s", markersize=7, linewidth=2,
            linestyle="--", label="Persistence (last reading)")
    ax.plot(x, sweep["yesterday_rmse"], color=C_YESTERDAY, marker="^", markersize=7, linewidth=2,
            linestyle=":", label="Same hour yesterday")
    ax.set_xticks(x, [f"+{h} h" for h in sweep["horizon"]])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Test RMSE (mg/L), lower is better")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.set_title(f"DO forecast error by horizon: LSTM vs baselines\n{NOTE}")
    cfg.save_fig(fig, "fig13_lstm_horizon_sweep")


def main() -> None:
    ap = argparse.ArgumentParser(description="LSTM DO forecast vs baselines")
    ap.add_argument("--epochs", type=int, default=60, help="maximum epochs (early stopping applies)")
    ap.add_argument("--target", choices=["delta", "level"], default="delta")
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()
    series = ts.prepare_series(cfg.load_dataset())

    cfg.section(f"05 LSTM DO forecast on the {NOTE}")
    print(f"  {len(series['time']):,} hourly steps; {int(series['bad'].sum())} hours inside gaps "
          f"> {cfg.MAX_GAP_H} h (windows touching them are skipped)")
    print(f"  window {WINDOW} h, target = {args.target}, chronological 70/15/15 split")

    cfg.section(f"Main model: {MAIN_UNITS} units, +{MAIN_HORIZON} h")
    res = run_once(series, MAIN_HORIZON, MAIN_UNITS, cfg.SEED, args.epochs, args.target, keep_model=True)
    print(f"  samples: train {res['n_train']:,} / val {res['n_val']:,} / test {res['n_test']:,}; "
          f"stopped after {res['epochs_run']} epochs")
    print(f"  split check (train targets < val inputs < test inputs in time): {res['split_ok']}")
    print(f"\n  {'method':<24}{'RMSE':>8}{'MAE':>8}{'R²':>8}")
    for name, key in (("LSTM", "lstm"), ("Persistence", "persistence"), ("Same hour yesterday", "yesterday")):
        s = res[key]
        print(f"  {name:<24}{s['rmse']:>8.3f}{s['mae']:>8.3f}{s['r2']:>8.3f}")
    skill = 1 - (res["lstm"]["rmse"] / res["persistence"]["rmse"]) ** 2
    print(f"\n  skill vs persistence (1 - MSE ratio): {skill:+.3f}  (> 0 means the LSTM beats it)")
    print(f"  (at +{MAIN_HORIZON} h 'same hour yesterday' is the last reading, i.e. identical to persistence)")
    verdict = "beats" if skill > 0 else "does NOT beat"
    print(f"  VERDICT: the LSTM {verdict} persistence at +{MAIN_HORIZON} h on this dataset.")
    if res["lstm"]["r2"] > 0.5:
        print("  NOTE: R² > 0.5 at +24 h. Persistence R² on the same hours is "
              f"{res['persistence']['r2']:.3f}; the daily cycle alone explains that much, so judge by skill.")

    tag = "05_lstm_do_forecast" if args.target == "delta" else "05_lstm_do_forecast_level"
    block = {"target": args.target, "window_h": WINDOW, "horizon_h": MAIN_HORIZON, "units": MAIN_UNITS,
             "n_train": res["n_train"], "n_val": res["n_val"], "n_test": res["n_test"],
             "epochs_run": res["epochs_run"], "lstm": res["lstm"], "persistence": res["persistence"],
             "same_hour_yesterday": res["yesterday"], "skill_vs_persistence": round(skill, 4),
             "split_ok": res["split_ok"]}
    if args.target == "level":
        cfg.record_metric(tag, block)
        print("\n  --target level: numbers recorded; figures, model and sweep are left untouched.")
        return

    cfg.section("Figures and model")
    fig_forecast(series, res)
    fig_scatter(res)
    res["model"].save(cfg.MODEL_DIR / "lstm_do_forecast.keras")
    meta = {"window_h": WINDOW, "horizon_h": MAIN_HORIZON, "channels": ts.CHANNELS, "target": "delta",
            "mean": res["mean"].tolist(), "sd": res["sd"].tolist(), "delta_sd": res["delta_sd"],
            "test_r2": res["lstm"]["r2"], "test_rmse": res["lstm"]["rmse"],
            "persistence_r2": res["persistence"]["r2"], "persistence_rmse": res["persistence"]["rmse"]}
    (cfg.MODEL_DIR / "lstm_do_forecast_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("  saved model and scaler (lstm_do_forecast.keras, lstm_do_forecast_meta.json)")

    if not args.skip_sweep:
        cfg.section(f"Horizon sweep: {SWEEP_UNITS} units, {len(SWEEP_SEEDS)} seeds each")
        rows = []
        for h in SWEEP_HORIZONS:
            runs = [run_once(series, h, SWEEP_UNITS, s, args.epochs) for s in SWEEP_SEEDS]
            rmse = np.array([r["lstm"]["rmse"] for r in runs])
            r2 = np.array([r["lstm"]["r2"] for r in runs])
            rows.append({"horizon": h, "n_test": runs[0]["n_test"],
                         "lstm_rmse_mean": rmse.mean(), "lstm_rmse_sd": rmse.std(),
                         "lstm_r2_mean": r2.mean(), "lstm_r2_sd": r2.std(),
                         "persistence_rmse": runs[0]["persistence"]["rmse"],
                         "persistence_r2": runs[0]["persistence"]["r2"],
                         "yesterday_rmse": runs[0]["yesterday"]["rmse"],
                         "yesterday_r2": runs[0]["yesterday"]["r2"]})
            best_base = min(rows[-1]["persistence_rmse"], rows[-1]["yesterday_rmse"])
            rows[-1]["lstm_beats_best_baseline"] = bool(rmse.mean() < best_base)
            print(f"  +{h:>2} h  LSTM RMSE {rmse.mean():.3f} ± {rmse.std():.3f} (R² {r2.mean():.3f})   "
                  f"persistence {rows[-1]['persistence_rmse']:.3f}   yesterday {rows[-1]['yesterday_rmse']:.3f}   "
                  f"{'LSTM ahead' if rows[-1]['lstm_beats_best_baseline'] else 'baseline ahead'}")
        sweep = pd.DataFrame(rows).round(4)
        sweep.to_csv(cfg.REPORT_DIR / "05_horizon_sweep.csv", index=False)
        fig_sweep(sweep)
        block["sweep"] = sweep.to_dict(orient="records")
    cfg.record_metric(tag, block)


if __name__ == "__main__":
    main()
