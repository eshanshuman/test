"""02 - Exploratory data analysis of the active dataset.

Purpose
    Look at the data before modelling it: how the four parameters relate, how
    they differ between Good / Moderate / Critical, how unbalanced the classes
    are, what the daily DO cycle looks like, and which parameter carries the
    most information about the label. This is the evidence for later design
    choices (macro F1 instead of accuracy, DO as the forecasting target).

Paper link
    The descriptive step that precedes the classifiers in the reviewed
    IoT-aquaculture papers (Chen et al. 2022; Hemal et al. 2024; Huang &
    Khabusi 2025). Nothing here reproduces a published number.

Outputs (figures/<dataset>/)
    fig01_correlation_heatmap.png   fig02_distributions_by_class.png
    fig03_class_balance.png         fig04_do_last_30_days.png
    fig05_mutual_information.png
    plus block "02_eda" in reports/<dataset>/metrics.json.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

import config as cfg

cfg.apply_style()
NOTE = f"{cfg.DATASET} dataset ({cfg.DATASET_NOTE})"


def fig_correlation(df: pd.DataFrame) -> pd.DataFrame:
    corr = df[cfg.FEATURES].corr()                       # pairwise, ignores NaN
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)  # diverging, white at 0
    names = [cfg.PRETTY[c] for c in cfg.FEATURES]
    ax.set_xticks(range(4), names, rotation=30, ha="right")
    ax.set_yticks(range(4), names)
    for i in range(4):
        for j in range(4):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.6 else "#222222")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85, label="Pearson r")
    ax.set_title(f"Correlation between parameters\n{NOTE}")
    cfg.save_fig(fig, "fig01_correlation_heatmap")
    return corr


def fig_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2))
    present = [c for c in cfg.CLASSES if (df[cfg.LABEL_COL] == c).any()]
    for ax, col in zip(axes.ravel(), cfg.FEATURES):
        groups = [df.loc[df[cfg.LABEL_COL] == c, col].dropna() for c in present]
        box = ax.boxplot(groups, patch_artist=True, widths=0.55, showfliers=False,
                         medianprops={"color": "#222222", "linewidth": 1.4})
        for patch, c in zip(box["boxes"], present):
            patch.set(facecolor=cfg.STATUS_COLOURS[c], edgecolor="#222222", alpha=0.85)
        ax.set_xticks(range(1, len(present) + 1), present)
        # dashed lines = edges of the Good band, dotted = edges of the Moderate band
        for style, key in (("--", "good"), (":", "moderate")):
            for edge in cfg.BANDS[col][key]:
                if np.isfinite(edge):
                    ax.axhline(edge, color="#555555", linestyle=style, linewidth=0.9)
        ax.set_ylabel(cfg.PRETTY[col])
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"Each parameter by pond status (whiskers 1.5 IQR; dashed = Good band, "
                 f"dotted = Moderate band)\n{NOTE}", fontsize=10, fontweight="bold")
    fig.tight_layout()
    cfg.save_fig(fig, "fig02_distributions_by_class")


def fig_class_balance(df: pd.DataFrame) -> pd.Series:
    counts = df[cfg.LABEL_COL].value_counts().reindex(cfg.CLASSES, fill_value=0)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    bars = ax.bar(counts.index, counts.values, width=0.6,
                  color=[cfg.STATUS_COLOURS[c] for c in counts.index])
    total = counts.sum()
    for bar, c in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, c, f"{c:,}\n{100 * c / total:.1f} %",
                ha="center", va="bottom")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.set_ylabel("Labelled hours")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Class balance\n{NOTE}")
    cfg.save_fig(fig, "fig03_class_balance")
    return counts


def fig_do_trace(df: pd.DataFrame) -> None:
    end = df[cfg.TIME_COL].max()
    last = df[df[cfg.TIME_COL] > end - pd.Timedelta(days=30)]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(last[cfg.TIME_COL], last["do_mg_l"], color=cfg.SERIES_COLOURS[0], linewidth=1.2)
    # a line takes the colour of the status the pond falls into BELOW it
    for level, name in ((5.0, "Moderate"), (3.0, "Critical")):
        ax.axhline(level, color=cfg.STATUS_COLOURS[name], linestyle="--", linewidth=1.4)
        ax.annotate(f"below {level:g} mg/L: {name}", xy=(1.01, level), va="center", fontsize=8,
                    xycoords=("axes fraction", "data"), color="#333333")
    ax.set_ylabel("DO (mg/L)")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_title(f"Dissolved oxygen, last 30 days (line breaks = missing readings)\n{NOTE}")
    cfg.save_fig(fig, "fig04_do_last_30_days")


def fig_mutual_information(df: pd.DataFrame) -> pd.Series:
    full = df.dropna(subset=cfg.FEATURES + [cfg.LABEL_COL])
    mi = mutual_info_classif(full[cfg.FEATURES], full[cfg.LABEL_COL], random_state=cfg.SEED)
    mi = pd.Series(mi, index=cfg.FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.barh([cfg.PRETTY[c] for c in mi.index], mi.values,
                   color=cfg.SERIES_COLOURS[0], height=0.55)
    for bar, v in zip(bars, mi.values):
        ax.text(v, bar.get_y() + bar.get_height() / 2, f" {v:.3f}", va="center")
    ax.set_xlim(0, mi.max() * 1.15)
    ax.set_xlabel("Mutual information with the status label (nats)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Which parameter says most about the label\n{NOTE}")
    cfg.save_fig(fig, "fig05_mutual_information")
    return mi.sort_values(ascending=False)


def main() -> None:
    df = cfg.load_dataset()
    cfg.section(f"02 EDA on the {NOTE}")
    print(f"  {len(df):,} hourly rows, {df[cfg.TIME_COL].min()} to {df[cfg.TIME_COL].max()}")
    print(f"  labelled rows: {int(df[cfg.LABEL_COL].notna().sum()):,}; "
          f"rows with all four readings: {int(df[cfg.FEATURES].notna().all(axis=1).sum()):,}")
    print(df[cfg.FEATURES].describe().loc[["min", "25%", "50%", "75%", "max"]].round(2).to_string())

    cfg.section("Figures")
    corr = fig_correlation(df)
    fig_distributions(df)
    counts = fig_class_balance(df)
    fig_do_trace(df)
    mi = fig_mutual_information(df)

    cfg.section("Correlation (Pearson r)")
    print(corr.round(2).to_string())

    cfg.section("Class balance")
    for name, c in counts.items():
        print(f"  {name:<9}{c:>6,}  {100 * c / counts.sum():5.1f} %")
    majority = counts.max() / counts.sum()
    print(f"  always guessing the largest class would score {100 * majority:.1f} % accuracy,")
    print("  which is why later scripts report macro F1 alongside accuracy.")

    cfg.section("Mutual information with the label (nats)")
    for col, v in mi.items():
        print(f"  {cfg.PRETTY[col]:<18}{v:.3f}")

    cfg.section("Daily DO cycle")
    by_hour = df.groupby(df[cfg.TIME_COL].dt.hour)["do_mg_l"].mean()
    h_min, h_max = int(by_hour.idxmin()), int(by_hour.idxmax())
    print(f"  mean daily DO minimum at {h_min:02d}:00 ({by_hour.min():.2f} mg/L)")
    print(f"  mean daily DO maximum at {h_max:02d}:00 ({by_hour.max():.2f} mg/L)")
    print(f"  mean day-night swing: {by_hour.max() - by_hour.min():.2f} mg/L")

    cfg.record_metric("02_eda", {
        "rows": len(df), "labelled_rows": int(df[cfg.LABEL_COL].notna().sum()),
        "class_counts": counts.to_dict(), "majority_class_share": round(float(majority), 4),
        "mutual_information_nats": {k: round(float(v), 4) for k, v in mi.items()},
        "do_min_hour": h_min, "do_max_hour": h_max,
        "do_mean_at_min_hour": round(float(by_hour.min()), 3),
        "do_mean_at_max_hour": round(float(by_hour.max()), 3),
        "correlation": corr.round(3).to_dict()})
    print(f"\n  metrics -> {cfg.METRICS_JSON.relative_to(cfg.ROOT).as_posix()}")


if __name__ == "__main__":
    main()
