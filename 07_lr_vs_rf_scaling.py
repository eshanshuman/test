"""07 - Linear Regression vs Random Forest as the training set grows.

Purpose
    A "soft sensor" question: can DO be estimated from the cheaper probes (pH,
    temperature, salinity) and the hour of day, and how much data does a
    flexible model need before it is worth using instead of a straight line?
    For each training size the script draws that many rows at random from the
    training pool 25 times, fits both models, and scores them on ONE fixed
    test set, giving a mean and a 95 % confidence interval per size. It then
    reports the smallest size from which the forest stays ahead at every
    larger size.

    Notes for a reviewer: the hour is given as sin/cos so the straight-line
    model can represent a daily cycle at all; this is estimation of DO at the
    same moment from other probes, NOT forecasting, so the test set is a
    random 20 % of the complete rows (fixed by the seed).

Paper link
    The reviewed papers compare simple and ensemble regressors for DO
    (Hemal et al. 2024; Huang & Khabusi 2025); the brief's de França et al.
    comparison was not supplied and is not reproduced. This script adds what
    those comparisons usually omit: the dependence on training-set size.

Outputs
    figures/<dataset>/fig15_lr_vs_rf_scaling.png;
    reports/<dataset>/07_lr_vs_rf_scaling.csv; block "07_lr_vs_rf_scaling"
    in metrics.json.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import config as cfg

cfg.apply_style()
NOTE = f"{cfg.DATASET} dataset ({cfg.DATASET_NOTE})"
SIZES = [25, 50, 100, 200, 400, 800, 1600, 3200]
N_DRAWS = 25
INPUTS = ["ph", "temperature_c", "salinity_ppt", "hour_sin", "hour_cos"]
C_LR, C_RF = cfg.SERIES_COLOURS[1], cfg.SERIES_COLOURS[0]


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mean_ci(values) -> tuple[float, float]:
    """Mean and half-width of the 95 % t-interval over the draws."""
    v = np.asarray(values, float)
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(half)


def main() -> None:
    cfg.section(f"07 Linear Regression vs Random Forest: DO from the other probes, {NOTE}")
    df = cfg.load_dataset().dropna(subset=cfg.FEATURES)
    hour = df[cfg.TIME_COL].dt.hour
    df = df.assign(hour_sin=np.sin(2 * np.pi * hour / 24), hour_cos=np.cos(2 * np.pi * hour / 24))
    pool, test = train_test_split(df, test_size=0.2, random_state=cfg.SEED)
    sizes = sorted({min(s, len(pool)) for s in SIZES})           # capped at the pool size
    print(f"  complete rows {len(df):,} -> training pool {len(pool):,} / fixed test set {len(test):,}")
    print(f"  inputs: {', '.join(INPUTS)}; target: do_mg_l")
    print(f"  sizes: {sizes}; {N_DRAWS} random draws per size")
    baseline = rmse(test["do_mg_l"], np.full(len(test), pool["do_mg_l"].mean()))
    print(f"  reference: always predicting the pool mean DO gives test RMSE {baseline:.3f} mg/L")

    cfg.section("Learning curves (test RMSE in mg/L, mean ± 95 % CI)")
    rng = np.random.default_rng(cfg.SEED)
    rows = []
    print(f"  {'n':>5}  {'Linear Regression':>20}  {'Random Forest':>20}  {'RF R²':>7}  ahead")
    for n in sizes:
        lr_err, rf_err, lr_r2, rf_r2 = [], [], [], []
        for _ in range(N_DRAWS):
            sample = pool.iloc[rng.choice(len(pool), size=n, replace=False)]
            lr = LinearRegression().fit(sample[INPUTS], sample["do_mg_l"])
            rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1,
                                       random_state=int(rng.integers(1_000_000)))
            rf.fit(sample[INPUTS], sample["do_mg_l"])
            p_lr, p_rf = lr.predict(test[INPUTS]), rf.predict(test[INPUTS])
            lr_err.append(rmse(test["do_mg_l"], p_lr)), rf_err.append(rmse(test["do_mg_l"], p_rf))
            lr_r2.append(r2_score(test["do_mg_l"], p_lr)), rf_r2.append(r2_score(test["do_mg_l"], p_rf))
        (lr_m, lr_ci), (rf_m, rf_ci) = mean_ci(lr_err), mean_ci(rf_err)
        rows.append({"n_train": n, "lr_rmse_mean": lr_m, "lr_rmse_ci95": lr_ci,
                     "rf_rmse_mean": rf_m, "rf_rmse_ci95": rf_ci,
                     "lr_r2_mean": float(np.mean(lr_r2)), "rf_r2_mean": float(np.mean(rf_r2)),
                     "rf_ahead": bool(rf_m < lr_m),
                     "ci_separate": bool(abs(rf_m - lr_m) > rf_ci + lr_ci)})
        who = ("forest" if rf_m < lr_m else "linear") + ("" if rows[-1]["ci_separate"] else " (CIs overlap)")
        print(f"  {n:>5}  {lr_m:>12.3f} ± {lr_ci:.3f}  {rf_m:>12.3f} ± {rf_ci:.3f}  "
              f"{rows[-1]['rf_r2_mean']:>7.3f}  {who}")
    table = pd.DataFrame(rows)
    table.round(4).to_csv(cfg.REPORT_DIR / "07_lr_vs_rf_scaling.csv", index=False)

    # smallest size from which the forest is ahead at that size AND every larger one
    def stays_ahead_from(flags) -> int | None:
        return next((int(n) for i, n in enumerate(sizes) if flags.iloc[i:].all()), None)

    crossover = stays_ahead_from(table["rf_ahead"])
    clear_from = stays_ahead_from(table["rf_ahead"] & table["ci_separate"])   # and the CIs do not overlap
    cfg.section("Result")
    if crossover is None:
        print("  The forest does NOT stay ahead: Linear Regression is ahead at the largest size.")
    else:
        print(f"  The forest stays ahead of Linear Regression from n = {crossover} training rows upward.")
        print("  With non-overlapping 95 % CIs at every larger size as well: "
              + (f"from n = {clear_from}." if clear_from is not None else "never."))
    last = table.iloc[-1]
    print(f"  At n = {int(last['n_train'])}: forest RMSE {last['rf_rmse_mean']:.3f} (R² {last['rf_r2_mean']:.3f}), "
          f"linear RMSE {last['lr_rmse_mean']:.3f} (R² {last['lr_r2_mean']:.3f}).")

    fig, ax = plt.subplots(figsize=(6.4, 4))
    for key, colour, marker, name in (("lr", C_LR, "s", "Linear Regression"), ("rf", C_RF, "o", "Random Forest")):
        m, ci = table[f"{key}_rmse_mean"], table[f"{key}_rmse_ci95"]
        ax.fill_between(table["n_train"], m - ci, m + ci, color=colour, alpha=0.2, linewidth=0)
        ax.plot(table["n_train"], m, color=colour, marker=marker, markersize=7, linewidth=2, label=name)
    ax.axhline(baseline, color="#888888", linestyle=":", linewidth=1.2)
    ax.annotate("always predict the mean", xy=(sizes[0], baseline), xytext=(0, 4),
                textcoords="offset points", fontsize=8, color="#555555")
    if clear_from is not None:
        ax.axvline(clear_from, color="#555555", linestyle="--", linewidth=1)
        ax.annotate(f"forest clearly ahead from n = {clear_from}", xy=(clear_from, ax.get_ylim()[0]), xytext=(4, 6),
                    textcoords="offset points", fontsize=8, color="#333333")
    ax.set_xscale("log")
    ax.set_xticks(sizes, [str(s) for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("Training rows (log scale)")
    ax.set_ylabel("Test RMSE (mg/L), lower is better")
    ax.legend(fontsize=8)
    ax.set_title(f"Estimating DO from pH, temperature, salinity and hour\n"
                 f"mean ± 95 % CI over {N_DRAWS} draws, {NOTE}")
    cfg.section("Figure")
    cfg.save_fig(fig, "fig15_lr_vs_rf_scaling")
    cfg.record_metric("07_lr_vs_rf_scaling", {
        "pool": len(pool), "n_test": len(test), "draws": N_DRAWS, "inputs": INPUTS,
        "mean_baseline_rmse": round(baseline, 4), "forest_ahead_from_n": crossover, "forest_clearly_ahead_from_n": clear_from,
        "curve": table.round(4).to_dict(orient="records")})


if __name__ == "__main__":
    main()
