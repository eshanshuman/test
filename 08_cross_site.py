"""08 - Cross-site test: does a DO model trained at one site work at another?

Purpose
    The real question behind any deployed model: a farmer's pond is not the
    pond the model was trained on. With the two NERRS sites (estuarine
    monitoring data, real data only) the script compares, on the SAME Cat
    Point test hours:
      * transfer : LSTM trained at Dry Bar, tested at Cat Point, with inputs
                   scaled by DRY BAR's scaler (Cat Point contributes nothing
                   to training or scaling);
      * local    : LSTM trained and tested at Cat Point;
      * persistence at Cat Point (DO stays at the last reading).
    Horizons +1, +3 and +6 h, 3 seeds each. Same windowing rules as 05:
    168-h window, the change in DO as target, windows touching a gap longer
    than MAX_GAP_H skipped. Both sites use their first 70 % of the period for
    training and the next 15 % for early stopping, so all training data is
    earlier in time than the Cat Point test period.

Paper link
    None of the reviewed papers tests a model away from the site it was
    trained on; this is the generalisation check a reviewer will ask for.
    Both sites are in the same bay, so this is a mild test: a different farm
    in a different climate would be harder.

Outputs
    figures/real/fig16_cross_site.png; reports/real/08_cross_site.csv; block
    "08_cross_site" in reports/real/metrics.json.
"""
from __future__ import annotations

import os

os.environ["AQUABOT_DATASET"] = "real"                   # this script only exists for the real data

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
import ts_lib as ts

cfg.apply_style()
NOTE = "real dataset (estuarine monitoring data)"
WINDOW, UNITS, HORIZONS, SEEDS, EPOCHS = 168, 32, (1, 3, 6), (0, 1, 2), 40
COLOURS = dict(zip(("transfer", "local", "persistence"), cfg.SERIES_COLOURS))
NAMES = {"transfer": "Trained at Dry Bar", "local": "Trained at Cat Point", "persistence": "Persistence"}


def fit_at(site: dict, horizon: int, seed: int):
    """Train on one site's training period; return the model and that site's scaler."""
    values, do = site["values"], site["values"][:, ts.DO]
    n = len(site["time"])
    train_t, val_t, _, train_end = ts.chrono_split(ts.valid_ends(site, WINDOW, horizon), horizon, n)
    mean, sd = ts.fit_scaler(values, train_end)
    scaled = (values - mean) / sd
    delta_sd = float(np.std(do[train_t + horizon] - do[train_t]))
    y = lambda t: ((do[t + horizon] - do[t]) / delta_sd).astype(np.float32)
    ts.seed_everything(seed)
    model = ts.build_lstm(WINDOW, UNITS)
    ts.train(model, ts.windows(scaled, train_t, WINDOW), y(train_t),
             ts.windows(scaled, val_t, WINDOW), y(val_t), EPOCHS, patience=6)
    return model, mean, sd, delta_sd, len(train_t)


def score_at(site: dict, test_t, horizon, model, mean, sd, delta_sd) -> dict:
    """Score a model on given test hours, scaling inputs with the scaler it was trained with."""
    do = site["values"][:, ts.DO]
    scaled = (site["values"] - mean) / sd
    pred = do[test_t] + model.predict(ts.windows(scaled, test_t, WINDOW), verbose=0).ravel() * delta_sd
    return ts.regression_scores(do[test_t + horizon], pred)


def main() -> None:
    cfg.section(f"08 Cross-site DO forecast: Dry Bar -> Cat Point, {NOTE}")
    if not (cfg.REAL_CSV.exists() and cfg.REAL_SITE2_CSV.exists()):
        raise SystemExit("Needs data/pond_dataset.csv and data/pond_dataset_site2.csv "
                         "(run tools/fetch_nerrs.py and 00_prepare_dataset.py for both sites).")
    dry = ts.prepare_series(cfg.load_dataset(cfg.REAL_CSV))
    cat = ts.prepare_series(cfg.load_dataset(cfg.REAL_SITE2_CSV))
    for name, site in (("Dry Bar", dry), ("Cat Point", cat)):
        v = site["values"]
        print(f"  {name:<10} {len(site['time']):,} h, {int(site['bad'].sum())} h in long gaps; mean DO "
              f"{np.nanmean(v[:, 0]):.2f} mg/L, temperature {np.nanmean(v[:, 1]):.2f} °C, "
              f"pH {np.nanmean(v[:, 2]):.2f}, salinity {np.nanmean(v[:, 3]):.2f} ppt")
    print(f"  window {WINDOW} h, {UNITS} units, {len(SEEDS)} seeds, horizons {HORIZONS}")

    cfg.section("Test RMSE at Cat Point (mg/L), mean ± sd over seeds")
    rows = []
    print(f"  {'horizon':>8} {'test n':>7}  {'Dry Bar -> Cat Point':>22}  {'Cat Point -> Cat Point':>24}  {'persistence':>12}")
    for h in HORIZONS:
        do = cat["values"][:, ts.DO]
        _, _, test_t, _ = ts.chrono_split(ts.valid_ends(cat, WINDOW, h), h, len(cat["time"]))
        persistence = ts.regression_scores(do[test_t + h], do[test_t])
        res = {"transfer": [], "local": []}
        for seed in SEEDS:
            res["transfer"].append(score_at(cat, test_t, h, *fit_at(dry, h, seed)[:4]))
            res["local"].append(score_at(cat, test_t, h, *fit_at(cat, h, seed)[:4]))
        row = {"horizon": h, "n_test": len(test_t),
               "persistence_rmse": persistence["rmse"], "persistence_r2": persistence["r2"]}
        for key in ("transfer", "local"):
            rm = np.array([r["rmse"] for r in res[key]])
            row.update({f"{key}_rmse_mean": rm.mean(), f"{key}_rmse_sd": rm.std(),
                        f"{key}_r2_mean": float(np.mean([r["r2"] for r in res[key]]))})
        row["transfer_beats_persistence"] = bool(row["transfer_rmse_mean"] < row["persistence_rmse"])
        row["transfer_penalty_pct"] = 100 * (row["transfer_rmse_mean"] / row["local_rmse_mean"] - 1)
        rows.append(row)
        print(f"  {'+' + str(h) + ' h':>8} {len(test_t):>7}  "
              f"{row['transfer_rmse_mean']:>14.3f} ± {row['transfer_rmse_sd']:.3f}  "
              f"{row['local_rmse_mean']:>16.3f} ± {row['local_rmse_sd']:.3f}  {row['persistence_rmse']:>12.3f}")
    table = pd.DataFrame(rows)
    table.round(4).to_csv(cfg.REPORT_DIR / "08_cross_site.csv", index=False)

    cfg.section("Result")
    for _, r in table.iterrows():
        verdict = "beats" if r["transfer_beats_persistence"] else "does NOT beat"
        print(f"  +{int(r['horizon'])} h: the Dry Bar model {verdict} persistence at Cat Point; its RMSE is "
              f"{r['transfer_penalty_pct']:+.1f} % relative to the locally trained model "
              f"(R² transfer {r['transfer_r2_mean']:.3f}, local {r['local_r2_mean']:.3f}).")

    fig, ax = plt.subplots(figsize=(6.6, 4))
    x, width = np.arange(len(table)), 0.26
    for k, key in enumerate(("transfer", "local", "persistence")):
        heights = table[f"{key}_rmse_mean"] if key != "persistence" else table["persistence_rmse"]
        err = table[f"{key}_rmse_sd"] if key != "persistence" else None
        bars = ax.bar(x + (k - 1) * width, heights, width * 0.92, yerr=err, color=COLOURS[key],
                      label=NAMES[key], error_kw={"ecolor": "#222222", "capsize": 3})
        for bar, v in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, v + (0 if err is None else err.max()) + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [f"+{h} h" for h in table["horizon"]])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Test RMSE at Cat Point (mg/L), lower is better")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"DO forecast at Cat Point: model from another site vs local model\n"
                 f"error bars = sd over {len(SEEDS)} seeds, {NOTE}")
    cfg.section("Figure")
    cfg.save_fig(fig, "fig16_cross_site")
    cfg.record_metric("08_cross_site", {"window_h": WINDOW, "units": UNITS, "seeds": len(SEEDS),
                                        "results": table.round(4).to_dict(orient="records")})


if __name__ == "__main__":
    main()
