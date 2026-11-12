"""03 - Random Forest classifier for pond status (Good / Moderate / Critical).

Purpose
    Learn the pond status from the four sensor readings and measure honestly
    how well that works. The forest is tuned with a 5-fold grid search (macro
    F1, so the rare Critical class counts as much as the common ones) on the
    training 80 % only; the test 20 % is touched once, at the end.

    The reference row: the labelling rule itself (config.label_frame) is
    scored on the same test readings.
      * Synthetic data: labels come from the noise-free pond, so the rule on
        noisy readings is NOT perfect. Its error is pure sensor noise, and it
        is the ceiling a model can honestly be expected to approach.
      * Real data: the labels were produced by that rule from these readings,
        so the rule scores 1.0 by construction and the classifier is only
        re-learning thresholds. A high score there is not a discovery.

Paper link
    Random Forest is the usual baseline classifier in the reviewed
    IoT-aquaculture literature (Chen et al. 2022; Hemal et al. 2024; Huang &
    Khabusi 2025). Their accuracies are reference points only; nothing here
    reproduces their data.

Outputs
    figures/<dataset>/fig06_rf_confusion_matrix.png, fig07_rf_roc.png,
    fig08_rf_permutation_importance.png; models/<dataset>/random_forest.joblib;
    reports/<dataset>/03_rf_classification_report.csv; block
    "03_random_forest" in metrics.json.
"""
from __future__ import annotations

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import config as cfg

cfg.apply_style()
NOTE = f"{cfg.DATASET} dataset ({cfg.DATASET_NOTE})"
GRID = {"n_estimators": [200, 400], "max_depth": [None, 12, 20],
        "min_samples_leaf": [1, 2, 4], "class_weight": [None, "balanced"]}
LABELS = list(range(len(cfg.CLASSES)))


def scores(y_true, y_pred) -> dict:
    return {"accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", labels=LABELS,
                                             zero_division=0)), 4)}


def fig_roc(y_test, proba) -> dict:
    fig, ax = plt.subplots(figsize=(5, 4.4))
    aucs = {}
    for i, name in enumerate(cfg.CLASSES):
        truth = (y_test == i).astype(int)
        if truth.min() == truth.max():          # class absent from the test set
            continue
        fpr, tpr, _ = roc_curve(truth, proba[:, i])
        aucs[name] = round(float(roc_auc_score(truth, proba[:, i])), 4)
        ax.plot(fpr, tpr, color=cfg.STATUS_COLOURS[name], linewidth=2,
                label=f"{name} vs rest (AUC {aucs[name]:.3f})")
    ax.plot([0, 1], [0, 1], color="#888888", linestyle=":", linewidth=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"Random Forest: one-vs-rest ROC\n{NOTE}")
    cfg.save_fig(fig, "fig07_rf_roc")
    return aucs


def fig_importance(model, X_test, y_test) -> pd.Series:
    result = permutation_importance(model, X_test, y_test, scoring="f1_macro",
                                    n_repeats=20, random_state=cfg.SEED, n_jobs=-1)
    imp = pd.Series(result.importances_mean, index=cfg.FEATURES)
    sd = pd.Series(result.importances_std, index=cfg.FEATURES)
    order = imp.sort_values().index
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh([cfg.PRETTY[c] for c in order], imp[order], xerr=sd[order], height=0.55,
            color=cfg.SERIES_COLOURS[0], error_kw={"ecolor": "#333333", "capsize": 3})
    for y_pos, c in enumerate(order):
        ax.text(imp[c] + sd[c], y_pos, f"  {imp[c]:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, (imp + sd).max() * 1.2)
    ax.set_xlabel("Drop in test macro F1 when the column is shuffled (mean ± sd, 20 repeats)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Random Forest: permutation importance\n{NOTE}")
    cfg.save_fig(fig, "fig08_rf_permutation_importance")
    return imp.sort_values(ascending=False)


def main() -> None:
    cfg.section(f"03 Random Forest on the {NOTE}")
    X_train, X_test, y_train, y_test = cfg.classification_split()
    print(f"  complete labelled rows: {len(X_train) + len(X_test):,} "
          f"-> train {len(X_train):,} / test {len(X_test):,} (stratified 80/20, seed {cfg.SEED})")
    print("  test class counts: " + ", ".join(
        f"{cfg.CLASSES[i]} {int((y_test == i).sum())}" for i in LABELS))

    cfg.section("Grid search (5-fold, macro F1, training rows only)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.SEED)
    search = GridSearchCV(RandomForestClassifier(random_state=cfg.SEED, n_jobs=1), GRID,
                          scoring="f1_macro", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print(f"  {len(search.cv_results_['params'])} combinations tried")
    print(f"  best params : {search.best_params_}")
    print(f"  best CV macro F1: {search.best_score_:.4f}")

    cfg.section("Held-out test set")
    pred, proba = model.predict(X_test), model.predict_proba(X_test)
    full_proba = np.zeros((len(X_test), len(cfg.CLASSES)))
    full_proba[:, model.classes_] = proba                # in case a class is missing
    rf = scores(y_test, pred)
    rule_labels = cfg.label_frame(X_test).map({c: i for i, c in enumerate(cfg.CLASSES)})
    rule = scores(y_test, rule_labels)
    majority = scores(y_test, np.full(len(y_test), y_train.mode()[0]))
    print(f"  {'method':<34}{'accuracy':>10}{'macro F1':>10}")
    for name, s in (("Random Forest", rf), ("Labelling rule on same readings", rule),
                    ("Always the majority class", majority)):
        print(f"  {name:<34}{s['accuracy']:>10.4f}{s['macro_f1']:>10.4f}")
    gap = rf["accuracy"] - rule["accuracy"]
    print(f"\n  Random Forest minus rule reference: {gap:+.4f} accuracy")
    if cfg.DATASET == "real":
        print("  The rule is 1.0 by construction here (labels were made by it from these")
        print("  readings), so the forest is re-learning thresholds: estuarine monitoring data.")
    else:
        print("  The rule's error is pure sensor noise: labels come from the noise-free pond.")

    report = classification_report(y_test, pred, labels=LABELS, target_names=cfg.CLASSES,
                                   output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T.round(4)
    print("\n" + report_df.loc[cfg.CLASSES, ["precision", "recall", "f1-score", "support"]].to_string())
    report_df.to_csv(cfg.REPORT_DIR / "03_rf_classification_report.csv")

    cfg.section("Figures and model")
    cm = confusion_matrix(y_test, pred, labels=LABELS)
    cfg.plot_confusion(cm, cfg.CLASSES, f"Random Forest: test confusion matrix\n{NOTE}",
                       "fig06_rf_confusion_matrix")
    aucs = fig_roc(y_test.to_numpy(), full_proba)
    imp = fig_importance(model, X_test, y_test)
    print("  permutation importance: " + ", ".join(f"{cfg.PRETTY[c]} {v:.3f}" for c, v in imp.items()))
    joblib.dump(model, cfg.MODEL_DIR / "random_forest.joblib", compress=3)
    print(f"  saved {(cfg.MODEL_DIR / 'random_forest.joblib').relative_to(cfg.ROOT).as_posix()}")

    cfg.record_metric("03_random_forest", {
        "n_train": len(X_train), "n_test": len(X_test), "best_params": search.best_params_,
        "cv_macro_f1": round(float(search.best_score_), 4), "test": rf,
        "rule_reference": rule, "majority_baseline": majority,
        "accuracy_minus_rule": round(gap, 4), "roc_auc_ovr": aucs,
        "per_class_f1": {c: report[c]["f1-score"] for c in cfg.CLASSES},
        "permutation_importance": {k: round(float(v), 4) for k, v in imp.items()},
        "confusion_matrix": cm})


if __name__ == "__main__":
    main()
